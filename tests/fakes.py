"""Test helpers shared across test folders: an in-memory experiment tracker."""

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeRun:
    name: str
    nested: bool
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    documents: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeTracker:
    runs: list[FakeRun] = field(default_factory=list)
    models: list[tuple[Any, Any, str, str]] = field(default_factory=list)
    registry: dict[tuple[str, str], tuple[Any, str]] = field(default_factory=dict)
    _active: list[FakeRun] = field(default_factory=list)

    @contextmanager
    def _run(self, name: str, nested: bool) -> Iterator[str]:
        run = FakeRun(name, nested)
        self.runs.append(run)
        self._active.append(run)
        try:
            yield f"run-{len(self.runs)}"
        finally:
            self._active.pop()

    def run(self, name: str, *, nested: bool = False) -> AbstractContextManager[str]:
        return self._run(name, nested)

    @property
    def current(self) -> FakeRun:
        return self._active[-1]

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.current.params.update(params)

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        self.current.metrics.update(metrics)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self.current.tags.update(tags)

    def log_dict(self, data: Mapping[str, Any], path: str) -> None:
        self.current.documents[path] = dict(data)

    def log_model(
        self, model: Any, input_example: Any, registered_name: str, alias: str
    ) -> str:
        self.models.append((model, input_example, registered_name, alias))
        version = str(len(self.models))
        self.registry[registered_name, alias] = (model, version)
        return version

    def load_model(self, registered_name: str, alias: str) -> tuple[Any, str]:
        return self.registry[registered_name, alias]
