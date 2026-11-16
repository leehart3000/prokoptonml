"""Model kinds for sequences: k-mer classifiers and specialists.

- ``bio.kmer-classifier``: the version the model's training alias points at
  in the MLflow registry, saved in a bundle as ``<name>.joblib``.
- ``bio.specialist``: a reference set. Loaded directly, it is the one built
  from the training split (as evaluated); in a bundle, it is rebuilt from
  **every** split (more species), with its barcode-gap table and a pre-built
  k-mer index, under ``specialists/<name>/``.
"""

import shutil
from pathlib import Path
from typing import Any

import joblib

from prokopton.ml.bio.pipeline.reference import (
    REFERENCE_NAME,
    build_reference,
    reference_path,
    serving_reference_path,
)
from prokopton.ml.bio.specialists.identify import NEIGHBOURS_NAME, Specialist
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.models import LoadError, ModelKind, TrackerFactory

__all__ = ["INDEX_NAME", "bio_model_kinds"]

INDEX_NAME = "index.npz"


def _bundle_classifier(
    config: AppConfig, name: str, folder: Path, tracker_factory: TrackerFactory
) -> tuple[str, list[str]]:
    model, version = _load_classifier(config, name, tracker_factory)
    joblib.dump(model, folder / f"{name}.joblib")
    return version, [f"{name}.joblib"]


def _load_bundled_classifier(config: AppConfig, name: str, folder: Path) -> Any:
    return joblib.load(folder / f"{name}.joblib")


def _load_classifier(
    config: AppConfig, name: str, tracker_factory: TrackerFactory
) -> tuple[Any, str]:
    training = config.classifier(name).training
    model, version = tracker_factory(config, name).load_model(
        training.registered_model, training.alias
    )
    return model, version


def _bundle_specialist(
    config: AppConfig, name: str, folder: Path, _: TrackerFactory
) -> tuple[str, list[str]]:
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
    files = [
        f"specialists/{name}/{f}" for f in (REFERENCE_NAME, NEIGHBOURS_NAME, INDEX_NAME)
    ]
    return (
        f"{config.data.source.release}, {report.target_records} target records",
        files,
    )


def _load_bundled_specialist(config: AppConfig, name: str, folder: Path) -> Specialist:
    target = folder / "specialists" / name
    return Specialist.from_parquet(
        target / REFERENCE_NAME, config.specialist(name), index=target / INDEX_NAME
    )


def _load_specialist(
    config: AppConfig, name: str, _: TrackerFactory
) -> tuple[Specialist, str]:
    path = reference_path(config, name)
    if not path.exists():
        msg = f"{path} not found; run the reference stage for {name!r} first"
        raise LoadError(msg)
    specialist = Specialist.from_parquet(path, config.specialist(name))
    return specialist, f"reference {config.data.source.release}"


def bio_model_kinds() -> list[ModelKind]:
    """The model kinds for sequences."""
    return [
        ModelKind(
            "bio.kmer-classifier",
            _bundle_classifier,
            _load_bundled_classifier,
            _load_classifier,
        ),
        ModelKind(
            "bio.specialist",
            _bundle_specialist,
            _load_bundled_specialist,
            _load_specialist,
        ),
    ]
