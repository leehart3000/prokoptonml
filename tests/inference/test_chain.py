"""Tests for serving chains: kinds, building, and how steps end a chain."""

from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from bio_app import minimal
from prokopton.ml.config import AppConfig
from prokopton.ml.inference import chain
from prokopton.ml.inference.chain import (
    FINAL,
    Answer,
    Chain,
    Decline,
    RequestSpec,
    Result,
    ServingKinds,
    ServingStepKind,
    Step,
    StepRunner,
    Work,
    serving_entry_point_kinds,
)


def config(*steps: str) -> AppConfig:
    return AppConfig.model_validate(
        minimal(
            serving={
                "chain": list(steps),
                "steps": {name: {"kind": f"test.{name}"} for name in steps},
            },
        )
    )


def answers(text: str) -> Callable[[AppConfig, str, Mapping[str, Any]], StepRunner]:
    def build(app: AppConfig, name: str, models: Mapping[str, Any]) -> StepRunner:
        def run(work: Work) -> None:
            work.trace.add(Step(name=name, summary=text))
            work.answer = Answer(level="x", name=text, text=text, confident=True)
            work.notes.append(f"{name} ran")

        return run

    return build


def final(app: AppConfig, name: str, models: Mapping[str, Any]) -> StepRunner:
    def run(work: Work) -> Literal["final"]:
        work.answer = Answer(level="y", name="final", text="final", confident=True)
        return FINAL

    return run


def declines(app: AppConfig, name: str, models: Mapping[str, Any]) -> StepRunner:
    def run(work: Work) -> Decline:
        work.trace.add(Step(name=name, summary="no"))
        return Decline("nope", "Not this.")

    return run


KINDS = ServingKinds(
    [
        ServingStepKind("test.first", "Answers.", answers("first")),
        ServingStepKind("test.second", "Answers again.", answers("second")),
        ServingStepKind("test.final", "A final answer.", final),
        ServingStepKind("test.declines", "Declines.", declines),
    ]
)


def test_later_steps_can_replace_the_answer() -> None:
    result = Chain(config("first", "second"), KINDS, {}).identify("x")

    assert isinstance(result, Result)
    assert result.status == "identified"
    assert result.answer is not None
    assert result.answer.name == "second"
    assert result.notes == ("first ran", "second ran")
    assert [step.name for step in result.trace] == ["first", "second"]


def test_a_final_answer_stops_the_chain() -> None:
    result = Chain(config("first", "final", "second"), KINDS, {}).identify("x")

    assert result.answer is not None
    assert result.answer.name == "final"
    assert result.notes == ("first ran",)


def test_a_decline_stops_the_chain() -> None:
    result = Chain(config("first", "declines", "second"), KINDS, {}).identify("x")

    assert (result.status, result.reason, result.message) == (
        "declined",
        "nope",
        "Not this.",
    )
    assert result.notes == ("first ran",)


def test_an_app_needs_serving_steps() -> None:
    with pytest.raises(ValueError, match="has no serving steps"):
        Chain(config(), KINDS, {})


def test_kinds_are_unique_and_known() -> None:
    with pytest.raises(ValueError, match="provided twice"):
        ServingKinds(
            [ServingStepKind("a", "A.", final), ServingStepKind("a", "A.", final)]
        )
    with pytest.raises(ValueError, match=r"kind 'x' \(known kinds: test\.declines"):
        KINDS["x"]


def test_installed_packages_can_add_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    kind = ServingStepKind("shop.step", "A step.", final)
    point = SimpleNamespace(load=lambda: lambda: [kind])
    monkeypatch.setattr(chain, "entry_points", lambda group: [point])

    assert serving_entry_point_kinds() == [kind]


def test_a_steps_model_must_be_loaded() -> None:
    app = AppConfig.model_validate(
        minimal(
            models={"router": {"kind": "bio.kmer-classifier"}},
            serving={
                "chain": ["first"],
                "steps": {"first": {"kind": "test.first", "model": "router"}},
            },
        )
    )

    with pytest.raises(ValueError, match="uses the 'router' model, which is not"):
        Chain(app, KINDS, {})


class Text(BaseModel):
    text: str


def test_the_first_step_says_what_requests_it_takes() -> None:
    with pytest.raises(ValueError, match="'first', does not take requests"):
        Chain(config("first"), KINDS, {}).request_model()

    spec = RequestSpec(lambda app, name: Text, lambda body: f"[{body.text}]")  # type: ignore[attr-defined]
    kinds = ServingKinds(
        [
            ServingStepKind("test.asks", "Takes text.", answers("asks"), request=spec),
            ServingStepKind(
                "test.first",
                "Logs.",
                answers("first"),
                log=lambda r: {"n": len(r.notes)},
            ),
        ]
    )
    built = Chain(config("asks", "first", "first"), kinds, {})

    assert built.request_model() is Text
    assert built.text_of(Text(text="x")) == "[x]"
    assert built.log_fields(built.identify("x")) == {"n": 3}
