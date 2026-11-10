"""Experiment tracking: a small interface, with an MLflow implementation.

Training code talks to a ``Tracker`` rather than to MLflow directly, so tests
can use a simple in-memory tracker, and another tracking service could be
swapped in later.
"""

import inspect
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any, Protocol

__all__ = ["MlflowTracker", "Tracker", "own_types"]


class Tracker(Protocol):
    """What training needs from an experiment tracker."""

    def run(self, name: str, *, nested: bool = False) -> AbstractContextManager[str]:
        """Start a run (yielding its id) and end it afterwards."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record settings for the current run."""
        ...

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record results for the current run."""
        ...

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Record descriptive labels for the current run."""
        ...

    def log_dict(self, data: Mapping[str, Any], path: str) -> None:
        """Save a JSON document with the current run."""
        ...

    def log_model(
        self, model: Any, input_example: Any, registered_name: str, alias: str
    ) -> str:
        """Save and register a model, point ``alias`` at it, return its version."""
        ...

    def load_model(self, registered_name: str, alias: str) -> tuple[Any, str]:
        """Load the model version that ``alias`` points at, and its version."""
        ...


def own_types(model: Any) -> list[str]:
    """Name the ProkoptonML classes in a model or pipeline.

    MLflow saves scikit-learn models in the skops format, which only rebuilds
    types it has been told to trust; scikit-learn's own are trusted already.
    """
    steps = getattr(model, "steps", [("model", model)])
    names = {f"{type(step).__module__}.{type(step).__qualname__}" for _, step in steps}
    return sorted(name for name in names if name.startswith("prokopton."))


class MlflowTracker:
    """A ``Tracker`` backed by MLflow.

    Args:
        tracking_uri: Where MLflow stores runs and the model registry, e.g.
            ``sqlite:///data/mlflow/mlflow.db``.
        artifact_dir: Folder for model files and other artifacts, used when
            the experiment is first created.
        experiment: The experiment's name.
    """

    def __init__(self, tracking_uri: str, artifact_dir: Path, experiment: str) -> None:
        import mlflow

        if tracking_uri.startswith("sqlite:///"):
            database = Path(tracking_uri.removeprefix("sqlite:///"))
            database.parent.mkdir(parents=True, exist_ok=True)
        self._mlflow: Any = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            mlflow.create_experiment(
                experiment, artifact_location=artifact_dir.resolve().as_uri()
            )
        mlflow.set_experiment(experiment)

    @contextmanager
    def _run(self, name: str, nested: bool) -> Iterator[str]:
        with self._mlflow.start_run(run_name=name, nested=nested) as active:
            yield str(active.info.run_id)

    def run(self, name: str, *, nested: bool = False) -> AbstractContextManager[str]:
        return self._run(name, nested)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self._mlflow.log_metrics(dict(metrics))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._mlflow.set_tags(dict(tags))

    def log_dict(self, data: Mapping[str, Any], path: str) -> None:
        self._mlflow.log_dict(dict(data), path)

    def log_model(
        self, model: Any, input_example: Any, registered_name: str, alias: str
    ) -> str:
        options: dict[str, Any] = {}
        accepted = inspect.signature(self._mlflow.sklearn.log_model).parameters
        if "skops_trusted_types" in accepted:
            options["skops_trusted_types"] = own_types(model)
        info = self._mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=input_example,
            registered_model_name=registered_name,
            **options,
        )
        version = str(info.registered_model_version)
        self._mlflow.MlflowClient().set_registered_model_alias(
            registered_name, alias, version
        )
        return version

    def load_model(self, registered_name: str, alias: str) -> tuple[Any, str]:
        found = self._mlflow.MlflowClient().get_model_version_by_alias(
            registered_name, alias
        )
        # Load by version, so the model matches the version reported
        model = self._mlflow.sklearn.load_model(
            f"models:/{registered_name}/{found.version}"
        )
        return model, str(found.version)
