"""Serving chains: an answer built by an app's serving steps, in order.

The app's config lists its serving steps (``[serving]``: ``chain`` and
``[serving.steps.<name>]``). Each step has a *kind*, such as
``bio.sequence-input`` or ``bio.router``, provided by the platform or by any
installed package that declares an entry point in the
``prokopton.ml.serving_steps`` group: a function that returns a list of
``ServingStepKind``.

A step gets the work so far and either carries on (perhaps with an answer
that a later step may replace), gives the final answer, or declines. Every
step records what it did in the decision trace, so an answer can be checked.
The first step's kind also says what result the chain returns.
"""

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from prokopton.ml.config import AppConfig

__all__ = [
    "ENTRY_POINT_GROUP",
    "FINAL",
    "Answer",
    "Chain",
    "Decline",
    "RequestSpec",
    "Result",
    "ServingKinds",
    "ServingStepKind",
    "Step",
    "StepRunner",
    "Trace",
    "Work",
    "serving_entry_point_kinds",
]

ENTRY_POINT_GROUP = "prokopton.ml.serving_steps"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Step(_Model):
    """One step of the decision trace, and how long it took."""

    name: str
    summary: str
    details: dict[str, Any] = {}
    seconds: float = 0.0


class Answer(_Model):
    """The answer to show."""

    level: str
    """How specific the answer is (for sequences: species, complex,
    species_group, genus, family, or the router's group, such as order)."""
    name: str | None
    text: str
    confident: bool
    species: tuple[str, ...] = ()


class Result(_Model):
    """The outcome of one request. A step's kind can add fields (for
    sequences: the length, strand and region)."""

    status: Literal["identified", "declined"]
    answer: Answer | None = None
    reason: str | None = None
    """For a declined request: a short code, such as ``too_short``."""
    message: str | None = None
    notes: tuple[str, ...] = ()
    trace: tuple[Step, ...] = ()
    seconds: float = 0.0
    """Time spent answering (not loading the models)."""


class Trace:
    """The steps so far, each timed from the end of the one before."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.started = self._last = time.perf_counter()

    def add(self, step: Step) -> None:
        now = time.perf_counter()
        self.steps.append(
            step.model_copy(update={"seconds": round(now - self._last, 4)})
        )
        self._last = now

    @property
    def seconds(self) -> float:
        return round(self._last - self.started, 4)


class Work:
    """A request as it goes through the chain."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.answer: Answer | None = None
        self.notes: list[str] = []
        self.fields: dict[str, Any] = {}
        """Extra result fields, set by steps (such as ``length``)."""
        self.found: dict[str, Any] = {}
        """What steps found, for later steps (such as the router's scores)."""
        self.trace = Trace()


@dataclass(frozen=True)
class Decline:
    """A step's decision to decline the request."""

    reason: str
    message: str


FINAL: Literal["final"] = "final"
"""Returned by a step whose answer is final: the chain stops there."""

StepRunner = Callable[[Work], Decline | Literal["final"] | None]
"""A built step: returns ``None`` to carry on, ``FINAL`` or a ``Decline``."""


@dataclass(frozen=True)
class RequestSpec:
    """What a chain's first step accepts: the request body, and the text in
    it that the chain answers."""

    model: Callable[[AppConfig, str], type[BaseModel]]
    """Makes the request model for the step (named in the config), with
    its example."""
    text: Callable[[BaseModel], str]
    """The text to answer, from a validated request."""


@dataclass(frozen=True)
class ServingStepKind:
    """A kind of serving step: what it does, and how to build it."""

    kind: str
    description: str
    build: Callable[[AppConfig, str, Mapping[str, Any]], StepRunner]
    """Builds the step from the config, its name and the loaded models."""
    result: type[Result] = Result
    """The result the chain returns when this kind is the first step."""
    request: RequestSpec | None = None
    """The request it accepts, when it is the first step."""
    log: Callable[[Result], dict[str, Any]] | None = None
    """Fields it adds to each request's log line (never the request itself)."""


class ServingKinds:
    """The serving step kinds available, by kind."""

    def __init__(self, kinds: Iterable[ServingStepKind]) -> None:
        self._kinds: dict[str, ServingStepKind] = {}
        for kind in kinds:
            if kind.kind in self._kinds:
                msg = f"Serving step kind {kind.kind!r} is provided twice"
                raise ValueError(msg)
            self._kinds[kind.kind] = kind

    def __getitem__(self, kind: str) -> ServingStepKind:
        if kind not in self._kinds:
            known = ", ".join(sorted(self._kinds))
            msg = f"Unknown serving step kind {kind!r} (known kinds: {known})"
            raise ValueError(msg)
        return self._kinds[kind]


def serving_entry_point_kinds() -> list[ServingStepKind]:
    """The serving step kinds provided by installed packages' entry points."""
    kinds: list[ServingStepKind] = []
    for point in entry_points(group=ENTRY_POINT_GROUP):
        kinds.extend(point.load()())
    return kinds


class Chain:
    """An app's serving steps, built from its config and loaded models."""

    def __init__(
        self, config: AppConfig, kinds: ServingKinds, models: Mapping[str, Any]
    ) -> None:
        if not config.serving.chain:
            msg = f"The {config.name!r} app has no serving steps ([serving] chain)"
            raise ValueError(msg)
        self.config = config
        self.steps: list[tuple[str, StepRunner]] = []
        for name in config.serving.chain:
            settings = config.serving.steps[name]
            if settings.model is not None and settings.model not in models:
                msg = (
                    f"Serving step {name!r} uses the {settings.model!r} model, "
                    "which is not loaded"
                )
                raise ValueError(msg)
            kind = kinds[settings.kind]
            self.steps.append((name, kind.build(config, name, models)))
        first = config.serving.chain[0]
        start = kinds[config.serving.steps[first].kind]
        self.result_type = start.result
        self._request = start.request
        self._first = first
        self._loggers: list[Callable[[Result], dict[str, Any]]] = []
        for name in config.serving.chain:
            log = kinds[config.serving.steps[name].kind].log
            if log is not None and log not in self._loggers:
                self._loggers.append(log)

    def request_model(self) -> type[BaseModel]:
        """The request body the chain accepts (from its first step's kind).

        Raises:
            ValueError: If the first step's kind does not take requests.
        """
        if self._request is None:
            msg = f"The first serving step, {self._first!r}, does not take requests"
            raise ValueError(msg)
        return self._request.model(self.config, self._first)

    def text_of(self, request: BaseModel) -> str:
        """The text to answer, from a request made with ``request_model``."""
        assert self._request is not None  # request_model made the request type
        return self._request.text(request)

    def log_fields(self, result: Result) -> dict[str, Any]:
        """What the steps' kinds log about a result (never the request)."""
        fields: dict[str, Any] = {}
        for log in self._loggers:
            fields.update(log(result))
        return fields

    def identify(self, text: str) -> Result:
        """Answer one request (see the module docstring)."""
        work = Work(text)
        for _, run in self.steps:
            outcome = run(work)
            if isinstance(outcome, Decline):
                return self.result_type(
                    status="declined",
                    reason=outcome.reason,
                    message=outcome.message,
                    notes=tuple(work.notes),
                    trace=tuple(work.trace.steps),
                    seconds=work.trace.seconds,
                    **work.fields,
                )
            if outcome == FINAL:
                break
        return self.result_type(
            status="identified",
            answer=work.answer,
            notes=tuple(work.notes),
            trace=tuple(work.trace.steps),
            seconds=work.trace.seconds,
            **work.fields,
        )
