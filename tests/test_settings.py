"""Tests for settings that packages declare (models, serving steps and
sections), and for reading schema 3 files as schema 4."""

from importlib import metadata
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from bio_app import minimal
from prokopton.ml import settings as module
from prokopton.ml.bio.config import BioConfig, bio_section, bio_settings
from prokopton.ml.config import AppConfig, convert_schema_3
from prokopton.ml.settings import Declared, Settings, declared, entry_point_settings
from prokopton.ml.training.tracking import _experiment


class Thing(BaseModel):
    size: int


def test_the_bio_settings_are_declared() -> None:
    found = declared()

    assert found.sections["bio"].model is BioConfig
    assert found.models["bio.specialist"].name == "bio.specialist"
    assert found.steps["bio.router"].needs_model == "bio.kmer-classifier"
    assert {s.name for s in bio_settings()} <= {
        *found.models,
        *found.steps,
        *found.sections,
    }


def test_settings_cannot_be_declared_twice() -> None:
    with pytest.raises(ValueError, match="model 'x' are declared twice"):
        Declared([Settings("model", "x", Thing), Settings("model", "x", Thing)])


def test_settings_come_from_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    point = metadata.EntryPoint(
        "shop", "test_settings:shop_settings", module.ENTRY_POINT_GROUP
    )
    monkeypatch.setattr(module, "entry_points", lambda group: [point])

    assert entry_point_settings() == shop_settings()


def shop_settings() -> list[Settings]:
    return [Settings("section", "shop", Thing)]


def app(**extra: Any) -> dict[str, Any]:
    return minimal(models={"router": {"kind": "bio.kmer-classifier"}}, **extra)


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ValidationError, match=r"unknown section \[shop\]"):
        AppConfig.model_validate(app(shop={"size": 1}))


def test_a_section_is_checked_against_its_settings() -> None:
    data = app()
    config = AppConfig.model_validate(data)
    assert isinstance(bio_section(config), BioConfig)
    assert config.section("shop") is None

    broken = AppConfig.model_validate(data).model_dump(mode="json", by_alias=True)
    broken["bio"]["split"] = {"salt": ""}

    with pytest.raises(ValidationError, match=r"\[bio\]: split\.salt"):
        AppConfig.model_validate(broken)


def test_a_models_settings_are_checked_against_its_kind() -> None:
    data = app()
    data["models"]["router"]["training"] = {"k_values": "eight"}

    with pytest.raises(ValidationError, match=r"\[models\.router\]: training"):
        AppConfig.model_validate(data)


def test_an_app_without_a_bio_section_is_not_a_sequence_app() -> None:
    data = {
        "schema": 4,
        "name": "penguins",
        "title": "Penguins",
        "data": {"release": "0.1"},
        "paths": {"models_dir": "m"},
    }

    config = AppConfig.model_validate(data)

    with pytest.raises(ValueError, match=r"'penguins' app has no \[bio\] section"):
        bio_section(config)


def test_schema_3_tables_no_section_claims_are_left_to_be_reported() -> None:
    data = {
        "schema": 3,
        "data": {"source": {"release": "GB1", "doi": "d"}, "paths": {"raw_dir": "r"}},
        "validation": {"min_length": 1},
    }

    converted = convert_schema_3(data, Declared([]))

    assert converted["schema"] == 4
    assert converted["data"] == {
        "release": "GB1",
        "source": {"doi": "d"},
        "paths": {"raw_dir": "r"},
    }
    assert converted["validation"] == {"min_length": 1}
    assert "paths" not in converted


def test_a_sections_schema_3_tables_move_into_it() -> None:
    found = Declared(
        [
            Settings(
                "section",
                "shop",
                Thing,
                schema_3_tables={"x.y": "shop.y", "z": "shop.z"},
            )
        ]
    )

    converted = convert_schema_3({"schema": 3, "z": {"size": 2}}, found)

    assert converted["shop"] == {"z": {"size": 2}}
    assert "z" not in converted


def test_a_models_experiment_comes_from_its_training_settings() -> None:
    config = AppConfig.model_validate(
        minimal(
            models={
                "router": {"kind": "bio.kmer-classifier"},
                "other": {"kind": "shop.thing"},
            }
        )
    )

    assert _experiment(config, "router") == "router"
    assert _experiment(config, "other") == f"{config.name}-other"
