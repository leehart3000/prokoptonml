"""Tests for serving a tabular classifier: its model kind (bundled and loaded)
and its serving steps, on the made-up table of test_tabular_stages."""

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fakes import FakeTracker
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.bundle import build_bundle, bundle_path, load_bundle
from prokopton.ml.inference.chain import Chain
from prokopton.ml.inference.load import load_identifier
from prokopton.ml.tabular.config import registered_model
from prokopton.ml.tabular.stages import split, train
from tabular_world import app_data

EXAMPLE = {"size": 9.2, "weight": 18.4, "colour": "blue"}


def serving_app(tmp_path: Path, threshold: float = 0.9) -> AppConfig:
    data: dict[str, Any] = app_data(tmp_path)
    data["serving"] = {
        "endpoint": "classify",
        "chain": ["record", "kind"],
        "steps": {
            "record": {
                "kind": "core.record-input",
                "model": "kind",
                "example": EXAMPLE,
            },
            "kind": {
                "kind": "core.classifier",
                "model": "kind",
                "threshold": threshold,
            },
        },
    }
    return AppConfig.model_validate(data)


def trained(tmp_path: Path, threshold: float = 0.9) -> tuple[AppConfig, FakeTracker]:
    config = serving_app(tmp_path, threshold)
    split(config, "kind")
    tracker = FakeTracker()
    train(config, "kind", lambda _: tracker)
    return config, tracker


def ask(chain: Chain, record: dict[str, Any]) -> Any:
    request = chain.request_model().model_validate(record)
    return chain.identify(chain.text_of(request))


def test_a_bundled_classifier_answers_a_record(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path)

    manifest = build_bundle(config, lambda _config, _name: tracker)
    chain = load_bundle(bundle_path(config)).identifier

    assert manifest.models == {"kind": "core.tabular-classifier"}
    assert manifest.steps == {"record": ("kind.joblib",), "kind": ("kind.joblib",)}
    result = ask(chain, EXAMPLE)
    assert result.status == "identified"
    assert result.answer.name == "large"
    assert result.answer.text.startswith("Most likely large (probability ")
    assert result.answer.confident
    assert [step.name for step in result.trace] == ["record", "kind"]
    assert result.trace[0].summary == "One record with 3 of 3 values."
    assert result.trace[1].details["top"][0]["label"] == "large"


def test_missing_values_are_filled_in(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path)
    chain = load_identifier(config, lambda _config, _name: tracker).identifier

    result = ask(chain, {"size": 1.2})

    assert result.answer.name == "small"
    assert result.trace[0].details["missing"] == ["weight", "colour"]


def test_an_unsure_answer_says_so(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path, threshold=1.0)
    chain = load_identifier(config, lambda _config, _name: tracker).identifier

    result = ask(chain, {"size": 5.0, "colour": "green"})

    assert not result.answer.confident
    assert result.answer.text.startswith("Possibly medium, but the model is unsure")


def test_a_record_without_values_is_declined(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path)
    chain = load_identifier(config, lambda _config, _name: tracker).identifier

    result = ask(chain, {})

    assert (result.status, result.reason) == ("declined", "empty_record")
    assert result.message == "The record has no values."


def test_the_request_is_the_models_columns(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path)
    chain = load_identifier(config, lambda _config, _name: tracker).identifier
    request = chain.request_model()

    assert request.__name__ == "ClassifyRequest"
    schema = request.model_json_schema()
    assert set(schema["properties"]) == {"size", "weight", "colour"}
    assert schema["examples"] == [EXAMPLE]
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        request.model_validate({"height": 2.0})
    with pytest.raises(ValidationError, match="valid number"):
        request.model_validate({"size": "big"})


def test_the_model_is_loaded_from_its_registered_name(tmp_path: Path) -> None:
    config, tracker = trained(tmp_path)
    asked: list[str] = []

    def factory(found: AppConfig, name: str) -> FakeTracker:
        asked.append(name)
        return tracker

    loaded = load_identifier(config, factory)

    assert loaded.versions == {"kind": "1"}
    assert asked == ["kind"]
    assert registered_model(config, "kind") == "things-kind"
