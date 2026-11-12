"""Load an app's models and specialists, ready to identify sequences.

The marker classifier and the router are loaded from the MLflow registry
(the version each one's training alias points at); each specialist that is
consulted at serving (it has a ``router_label``) is loaded from its
reference set.
"""

from collections.abc import Callable
from dataclasses import dataclass

from prokopton.ml.config import AppConfig
from prokopton.ml.inference.identify import Identifier
from prokopton.ml.pipeline.reference import reference_path
from prokopton.ml.specialists.identify import Specialist
from prokopton.ml.training.tracking import Tracker, mlflow_tracker

__all__ = ["LoadError", "Loaded", "load_identifier"]


class LoadError(RuntimeError):
    """Raised when a model or reference set is not available."""


@dataclass(frozen=True, slots=True)
class Loaded:
    """An identifier, and the model versions it uses."""

    identifier: Identifier
    versions: dict[str, str]


def load_identifier(
    config: AppConfig,
    tracker_factory: Callable[[AppConfig, str], Tracker] = mlflow_tracker,
) -> Loaded:
    """Load the marker classifier, router and specialists.

    Raises:
        LoadError: If the app has no marker classifier, or a specialist's
            reference set has not been built.
    """
    if config.marker is None:
        msg = "This app has no marker classifier configured"
        raise LoadError(msg)
    models = {}
    versions = {}
    for name in ("marker", "router"):
        training = config.classifier(name).training
        tracker = tracker_factory(config, name)
        models[name], versions[name] = tracker.load_model(
            training.registered_model, training.alias
        )
    specialists = {}
    for name, rules in config.specialists.items():
        if not rules.router_label:
            continue
        path = reference_path(config, name)
        if not path.exists():
            msg = f"{path} not found; run the reference stage for {name!r} first"
            raise LoadError(msg)
        specialists[name] = Specialist.from_parquet(path, rules)
        versions[name] = f"reference {config.data.source.release}"
    identifier = Identifier(config, models["marker"], models["router"], specialists)
    return Loaded(identifier=identifier, versions=versions)
