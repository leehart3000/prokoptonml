"""Load an app's models, ready to identify sequences.

Each model a serving step uses is loaded from where the pipeline keeps it
(see ``models``): for sequences, the classifiers from the MLflow registry
(the version each one's training alias points at), and the specialists from
their reference sets.
"""

from dataclasses import dataclass
from typing import Any

from prokopton.ml.config import AppConfig
from prokopton.ml.inference.identify import Identifier, model_kinds
from prokopton.ml.inference.models import LoadError, TrackerFactory, serving_models
from prokopton.ml.training.tracking import mlflow_tracker

__all__ = ["LoadError", "Loaded", "load_identifier"]


@dataclass(frozen=True, slots=True)
class Loaded:
    """An identifier, and the model versions it uses."""

    identifier: Identifier
    versions: dict[str, str]


def load_identifier(
    config: AppConfig, tracker_factory: TrackerFactory = mlflow_tracker
) -> Loaded:
    """Load the models the app's serving steps use.

    Raises:
        LoadError: If a model is not available (for example, a specialist's
            reference set has not been built).
    """
    kinds = model_kinds()
    models: dict[str, Any] = {}
    versions: dict[str, str] = {}
    for name in serving_models(config):
        kind = kinds[config.models[name].kind]
        models[name], versions[name] = kind.load_current(config, name, tracker_factory)
    return Loaded(identifier=Identifier(config, models), versions=versions)
