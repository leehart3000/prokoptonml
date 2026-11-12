"""A serving bundle: everything needed to identify sequences, in one folder.

Serving should not need MLflow, the split data or minutes of indexing, so
``build_bundle`` gathers, once:

- ``config.json``: the app's settings;
- ``marker.joblib`` and ``router.joblib``: the registered models (the
  versions their training aliases point at);
- ``specialists/<name>/``: each serving specialist's reference set, drawn
  from **every** split (more species than the training-only set used to
  evaluate it), with its barcode-gap table and a pre-built k-mer index;
- ``requirements.txt``: the exact versions of the packages the models were
  saved with, for the serving image to install;
- ``manifest.json``: model versions, the SHA-256 of every file, and when
  and with what the bundle was made.

A pickled scikit-learn model is only safe to load with the scikit-learn (and
NumPy and SciPy) it was saved with, so ``load_bundle`` refuses to load a
bundle when those versions differ.

The models are saved with joblib, which uses pickle: loading a pickle can
run code, so ``load_bundle`` only loads files whose SHA-256 matches the
manifest, and bundles should only come from our own build.
"""

import hashlib
import json
import logging
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.identify import Identifier
from prokopton.ml.inference.load import Loaded
from prokopton.ml.pipeline.reference import (
    REFERENCE_NAME,
    build_reference,
    serving_reference_path,
)
from prokopton.ml.specialists.identify import NEIGHBOURS_NAME, Specialist
from prokopton.ml.training.tracking import Tracker, mlflow_tracker

__all__ = [
    "MANIFEST_NAME",
    "BundleError",
    "BundleManifest",
    "build_bundle",
    "bundle_path",
    "load_bundle",
]

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
INDEX_NAME = "index.npz"
RUNTIME_PACKAGES = (
    "prokopton",
    "numpy",
    "scipy",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "pyarrow",
    "pandas",
    "pydantic",
    "fastapi",
    "uvicorn",
)
"""Packages serving needs, pinned in the bundle's ``requirements.txt``."""
MUST_MATCH = ("numpy", "scipy", "scikit-learn")
"""Packages that must be the same version to load the pickled models."""


class BundleError(RuntimeError):
    """Raised when a bundle is missing, incomplete or altered."""


class BundleManifest(BaseModel):
    """What a bundle holds, and a checksum for every file in it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app: str
    release: str
    prokopton_ml_version: str
    created_at: datetime
    versions: dict[str, str]
    specialists: tuple[str, ...]
    files: dict[str, str]
    """Path within the bundle -> SHA-256."""
    runtime: dict[str, str] = {}
    """Package -> the version installed when the bundle was built."""


def _installed(names: tuple[str, ...]) -> dict[str, str]:
    found = {}
    for name in names:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return found


def bundle_path(config: AppConfig) -> Path:
    """Where an app's serving bundle is written."""
    return config.data.paths.models_dir / "bundle"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle(
    config: AppConfig,
    tracker_factory: Callable[[AppConfig, str], Tracker] = mlflow_tracker,
) -> BundleManifest:
    """Build the serving bundle (see the module docstring).

    Each serving specialist's reference set is rebuilt from every split,
    which takes a few minutes. The bundle is written beside its final place
    and moved there at the end, so a failed build leaves the old one.

    Raises:
        BundleError: If the app has no marker classifier.
    """
    if config.marker is None:
        msg = "This app has no marker classifier configured"
        raise BundleError(msg)
    final = bundle_path(config)
    folder = final.with_name(f"{final.name}.part")
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(config.model_dump_json(indent=2), "utf-8")
    runtime = _installed(RUNTIME_PACKAGES)
    (folder / "requirements.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in runtime.items()),
        "utf-8",
    )

    versions = {}
    for name in ("marker", "router"):
        training = config.classifier(name).training
        model, versions[name] = tracker_factory(config, name).load_model(
            training.registered_model, training.alias
        )
        joblib.dump(model, folder / f"{name}.joblib")

    serving = [name for name, rules in config.specialists.items() if rules.router_label]
    for name in serving:
        report = build_reference(config, name, serving=True)
        source = serving_reference_path(config, name).parent
        target = folder / "specialists" / name
        target.mkdir(parents=True)
        for file_name in (REFERENCE_NAME, NEIGHBOURS_NAME):
            shutil.copy2(source / file_name, target / file_name)
        specialist = Specialist.from_parquet(
            target / REFERENCE_NAME, config.specialist(name)
        )
        specialist.save_index(target / INDEX_NAME)
        versions[name] = (
            f"{config.data.source.release}, {report.target_records} target records"
        )

    files = {
        str(path.relative_to(folder)): _sha256(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }
    manifest = BundleManifest(
        app=config.name,
        release=config.data.source.release,
        prokopton_ml_version=__version__,
        created_at=datetime.now(UTC),
        versions=versions,
        specialists=tuple(serving),
        files=files,
        runtime=runtime,
    )
    (folder / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2), "utf-8")
    shutil.rmtree(final, ignore_errors=True)
    folder.replace(final)
    logger.info("Wrote the serving bundle to %s: %s", final, versions)
    return manifest


def load_bundle(folder: Path) -> Loaded:
    """Load a serving bundle, checking every file against its manifest.

    Raises:
        BundleError: If the manifest is missing, a file is missing or does
            not match its checksum, or NumPy, SciPy or scikit-learn are not
            the versions the models were saved with.
    """
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.exists():
        msg = f"{manifest_path} not found; build the bundle first"
        raise BundleError(msg)
    manifest = BundleManifest.model_validate_json(manifest_path.read_text("utf-8"))
    for name, expected in manifest.files.items():
        path = folder / name
        if not path.exists() or _sha256(path) != expected:
            msg = f"{path} is missing or does not match the bundle's manifest"
            raise BundleError(msg)
    installed = _installed(MUST_MATCH)
    for name in MUST_MATCH:
        built = manifest.runtime.get(name)
        if built is not None and installed.get(name) != built:
            msg = (
                f"The bundle's models were saved with {name} {built}, but "
                f"{installed.get(name)} is installed; install the bundle's "
                "requirements.txt"
            )
            raise BundleError(msg)
    config = AppConfig.model_validate(
        json.loads((folder / "config.json").read_text("utf-8"))
    )
    marker = joblib.load(folder / "marker.joblib")
    router = joblib.load(folder / "router.joblib")
    specialists = {}
    for name in manifest.specialists:
        target = folder / "specialists" / name
        specialists[name] = Specialist.from_parquet(
            target / REFERENCE_NAME, config.specialist(name), index=target / INDEX_NAME
        )
    return Loaded(
        identifier=Identifier(config, marker, router, specialists),
        versions=manifest.versions,
    )
