"""The ``core.tabular-classifier`` model kind: the version the training alias
points at in the MLflow registry, saved in a bundle as ``<name>.joblib``."""

from pathlib import Path
from typing import Any

import joblib

from prokopton.ml.config import AppConfig
from prokopton.ml.inference.models import ModelKind, TrackerFactory
from prokopton.ml.tabular.config import KIND, registered_model, table_classifier

__all__ = ["tabular_model_kinds"]


def _load_current(
    config: AppConfig, name: str, tracker_factory: TrackerFactory
) -> tuple[Any, str]:
    alias = table_classifier(config, name).training.alias
    model, version = tracker_factory(config, name).load_model(
        registered_model(config, name), alias
    )
    return model, version


def _bundle(
    config: AppConfig, name: str, folder: Path, tracker_factory: TrackerFactory
) -> tuple[str, list[str]]:
    model, version = _load_current(config, name, tracker_factory)
    joblib.dump(model, folder / f"{name}.joblib")
    return version, [f"{name}.joblib"]


def _load_bundled(config: AppConfig, name: str, folder: Path) -> Any:
    return joblib.load(folder / f"{name}.joblib")


def tabular_model_kinds() -> list[ModelKind]:
    """The tabular model kinds (the ``prokopton.ml.model_kinds`` entry point)."""
    return [ModelKind(KIND, _bundle, _load_bundled, _load_current)]
