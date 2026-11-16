"""Model kinds: how each kind of model is put into a serving bundle and
loaded again, or loaded straight from where the pipeline keeps it.

A serving chain's steps say which models they use (``model`` in each
``[serving.steps.<name>]``); each model's kind (``[models.<name>] kind``)
says how it is stored. Kinds are provided by the platform
(``prokopton.ml.bio.models`` for sequence models) and by any installed
package that declares an entry point in the ``prokopton.ml.model_kinds``
group: a function that returns a list of ``ModelKind``.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from prokopton.ml.config import AppConfig
from prokopton.ml.training.tracking import Tracker

__all__ = [
    "ENTRY_POINT_GROUP",
    "LoadError",
    "ModelKind",
    "ModelKinds",
    "TrackerFactory",
    "model_entry_point_kinds",
    "serving_models",
]

ENTRY_POINT_GROUP = "prokopton.ml.model_kinds"

TrackerFactory = Callable[[AppConfig, str], Tracker]
"""Makes the MLflow tracker for one model (replaced in tests)."""


class LoadError(RuntimeError):
    """Raised when a model or reference set is not available."""


@dataclass(frozen=True)
class ModelKind:
    """How a kind of model is bundled and loaded."""

    kind: str
    bundle: Callable[[AppConfig, str, Path, TrackerFactory], tuple[str, list[str]]]
    """Writes the model's files into the bundle folder; returns its version
    and the files written (relative to the folder)."""
    load_bundled: Callable[[AppConfig, str, Path], Any]
    """Loads the model from a bundle folder."""
    load_current: Callable[[AppConfig, str, TrackerFactory], tuple[Any, str]]
    """Loads the model from where the pipeline keeps it (the MLflow registry
    or the pipeline's data); returns it and its version."""


class ModelKinds:
    """The model kinds available, by kind."""

    def __init__(self, kinds: Iterable[ModelKind]) -> None:
        self._kinds: dict[str, ModelKind] = {}
        for kind in kinds:
            if kind.kind in self._kinds:
                msg = f"Model kind {kind.kind!r} is provided twice"
                raise ValueError(msg)
            self._kinds[kind.kind] = kind

    def __getitem__(self, kind: str) -> ModelKind:
        if kind not in self._kinds:
            known = ", ".join(sorted(self._kinds))
            msg = f"Unknown model kind {kind!r} (known kinds: {known})"
            raise ValueError(msg)
        return self._kinds[kind]


def model_entry_point_kinds() -> list[ModelKind]:
    """The model kinds provided by installed packages' entry points."""
    kinds: list[ModelKind] = []
    for point in entry_points(group=ENTRY_POINT_GROUP):
        kinds.extend(point.load()())
    return kinds


def serving_models(config: AppConfig) -> list[str]:
    """The models the app's serving steps use, in the order of ``[models]``."""
    used = {
        step.model
        for name, step in config.serving.steps.items()
        if name in config.serving.chain and step.model is not None
    }
    return [name for name in config.models if name in used]
