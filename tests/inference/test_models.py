"""Tests for model kinds: the registry, and which models serving uses."""

from types import SimpleNamespace

import pytest

from prokopton.ml.config import AppConfig
from prokopton.ml.inference import models
from prokopton.ml.inference.models import (
    ModelKind,
    ModelKinds,
    model_entry_point_kinds,
    serving_models,
)
from test_schema_2 import minimal


def kind(name: str) -> ModelKind:
    return ModelKind(
        name,
        bundle=lambda config, model, folder, factory: ("1", []),
        load_bundled=lambda config, model, folder: None,
        load_current=lambda config, model, factory: (None, "1"),
    )


def test_kinds_are_unique_and_known() -> None:
    with pytest.raises(ValueError, match="provided twice"):
        ModelKinds([kind("a"), kind("a")])
    with pytest.raises(ValueError, match=r"kind 'x' \(known kinds: a, b\)"):
        ModelKinds([kind("b"), kind("a")])["x"]


def test_installed_packages_can_add_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    added = kind("shop.model")
    point = SimpleNamespace(load=lambda: lambda: [added])
    monkeypatch.setattr(models, "entry_points", lambda group: [point])

    assert model_entry_point_kinds() == [added]


def test_serving_uses_the_models_of_the_steps_in_its_chain() -> None:
    config = AppConfig.model_validate(
        minimal(
            schema=3,
            models={
                "a": {"kind": "bio.kmer-classifier"},
                "b": {"kind": "bio.kmer-classifier"},
                "c": {"kind": "bio.kmer-classifier"},
            },
            serving={
                "chain": ["second", "first", "plain"],
                "steps": {
                    "first": {"kind": "bio.router", "model": "a"},
                    "second": {"kind": "bio.router", "model": "b"},
                    "plain": {"kind": "shop.step"},
                    "unused": {"kind": "bio.router", "model": "c"},
                },
            },
        )
    )

    assert serving_models(config) == ["a", "b"]
