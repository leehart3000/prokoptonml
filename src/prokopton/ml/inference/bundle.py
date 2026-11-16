"""A serving bundle: everything needed to answer requests, in one folder.

Serving should not need MLflow, the split data or minutes of indexing, so
``build_bundle`` gathers, once:

- ``config.json``: the app's settings;
- the files of each model the serving steps use, written by its kind (see
  ``models``; for sequences: ``<name>.joblib`` for a classifier, and
  ``specialists/<name>/`` for a specialist's reference set, drawn from
  **every** split, with its barcode-gap table and a pre-built k-mer index);
- ``requirements.txt``: the exact versions of the packages the models were
  saved with, for the serving image to install;
- ``manifest.json``: the bundle format (2), each model's kind and version,
  which files each serving step uses, the SHA-256 of every file, and when
  and with what the bundle was made.

Bundles of format 1 (with ``marker.joblib``, ``router.joblib`` and the
specialists named in the manifest) are still read, until prokopton-ml 0.2.0.

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
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.identify import Identifier, model_kinds
from prokopton.ml.inference.load import Loaded
from prokopton.ml.inference.models import TrackerFactory, serving_models
from prokopton.ml.training.tracking import mlflow_tracker

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
FORMAT = 2
"""The bundle format written; formats 1 and 2 are read."""
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
    format: int = 1
    """1 for bundles made before the format was recorded."""
    models: dict[str, str] = {}
    """Model -> its kind (format 2)."""
    steps: dict[str, tuple[str, ...]] = {}
    """Serving step -> the files of the model it uses (format 2)."""
    specialists: tuple[str, ...] = ()
    """The specialists (format 1)."""
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
    config: AppConfig, tracker_factory: TrackerFactory = mlflow_tracker
) -> BundleManifest:
    """Build the serving bundle (see the module docstring).

    A specialist's reference set is rebuilt from every split, which takes a
    few minutes. The bundle is written beside its final place and moved
    there at the end, so a failed build leaves the old one.

    Raises:
        BundleError: If the app has no serving steps.
    """
    if not config.serving.chain:
        msg = f"The {config.name!r} app has no serving steps ([serving] chain)"
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

    kinds = model_kinds()
    versions: dict[str, str] = {}
    model_files: dict[str, list[str]] = {}
    for name in serving_models(config):
        kind = kinds[config.models[name].kind]
        versions[name], model_files[name] = kind.bundle(
            config, name, folder, tracker_factory
        )
    steps = {
        name: tuple(model_files[step.model]) if step.model else ()
        for name, step in config.serving.steps.items()
        if name in config.serving.chain
    }

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
        format=FORMAT,
        models={name: config.models[name].kind for name in model_files},
        steps=steps,
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
        BundleError: If the manifest is missing or of an unknown format, a
            file is missing or does not match its checksum, or NumPy, SciPy
            or scikit-learn are not the versions the models were saved with.
    """
    manifest_path = folder / MANIFEST_NAME
    if not manifest_path.exists():
        msg = f"{manifest_path} not found; build the bundle first"
        raise BundleError(msg)
    manifest = BundleManifest.model_validate_json(manifest_path.read_text("utf-8"))
    if manifest.format not in (1, FORMAT):
        msg = (
            f"{manifest_path} is bundle format {manifest.format}; this version of "
            f"prokopton-ml reads formats 1 and {FORMAT}"
        )
        raise BundleError(msg)
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
    stored = manifest.models or {
        "marker": "bio.kmer-classifier",
        "router": "bio.kmer-classifier",
        **dict.fromkeys(manifest.specialists, "bio.specialist"),
    }
    kinds = model_kinds()
    models = {
        name: kinds[kind].load_bundled(config, name, folder)
        for name, kind in stored.items()
    }
    return Loaded(identifier=Identifier(config, models), versions=manifest.versions)
