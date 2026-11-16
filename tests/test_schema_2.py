"""Tests for config schema 2: models, the pipeline, and reading schema 1."""

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from prokopton.ml.config import (
    AppConfig,
    ClassifierConfig,
    SpecialistConfig,
    UnknownClassifierError,
    convert_schema_1,
    default_stages,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DNABARCODE_STAGES = (
    "ingest",
    "validate",
    "split",
    "leakage",
    *(f"router-{s}" for s in ("sample", "train", "calibrate", "evaluate", "card")),
    *(f"marker-{s}" for s in ("sample", "train", "calibrate", "evaluate", "card")),
    *(f"culicidae-{s}" for s in ("anchors", "reference", "evaluate", "card")),
    "bundle",
)


def schema_1() -> dict[str, Any]:
    with (FIXTURES / "dnabarcode.toml").open("rb") as handle:
        return tomllib.load(handle)


def minimal(**extra: Any) -> dict[str, Any]:
    data = schema_1()
    kept = {key: data[key] for key in ("name", "title", "data", "validation")}
    return {"schema": 2, **kept, **extra}


def test_schema_1_is_converted_with_every_stage_it_ran() -> None:
    config = AppConfig.model_validate(schema_1())

    assert config.schema_version == 2
    assert list(config.models) == ["router", "marker", "culicidae"]
    assert isinstance(config.models["router"], ClassifierConfig)
    assert isinstance(config.models["culicidae"], SpecialistConfig)
    assert config.pipeline.stages == DNABARCODE_STAGES
    assert config.stages["router-train"].kind == "bio.classifier-train"
    assert config.stages["router-train"].model == "router"
    assert config.stages["culicidae-anchors"].kind == "bio.specialist-anchors"
    assert config.stages["bundle"].model is None


def test_settings_are_unchanged_by_the_conversion() -> None:
    data = schema_1()
    config = AppConfig.model_validate(data)

    assert config.router.training.k_values == tuple(
        data["router"]["training"]["k_values"]
    )
    assert config.marker is not None
    assert config.marker.label == data["marker"]["label"]
    culicidae = config.specialist("culicidae")
    assert (
        culicidae.species_identity
        == data["specialists"]["culicidae"]["species_identity"]
    )
    assert list(config.specialists) == ["culicidae"]


def test_a_dumped_config_reads_back_the_same() -> None:
    config = AppConfig.model_validate(schema_1())

    assert AppConfig.model_validate_json(config.model_dump_json()) == config
    assert json.loads(config.model_dump_json(by_alias=True))["schema"] == 2


def test_a_schema_1_app_without_marker_or_anchors() -> None:
    data = schema_1()
    del data["marker"]
    data["specialists"]["culicidae"].pop("anchors", None)

    config = AppConfig.model_validate(data)

    assert config.marker is None
    assert "marker-train" not in config.stages
    assert "culicidae-anchors" not in config.stages
    assert "culicidae-reference" in config.stages


def test_schema_1_without_a_router_gets_the_default_one() -> None:
    data = schema_1()
    del data["router"]

    converted = convert_schema_1(data)

    assert converted["models"]["router"] == {"kind": "bio.kmer-classifier"}


def test_default_stages_cover_classifiers_then_specialists() -> None:
    stages = default_stages(
        {"s": {"kind": "bio.specialist"}, "c": {"kind": "bio.kmer-classifier"}}
    )

    assert list(stages)[4:9] == [
        f"c-{s}" for s in ("sample", "train", "calibrate", "evaluate", "card")
    ]
    assert list(stages)[-1] == "bundle"


def test_schema_2_is_read_as_written() -> None:
    config = AppConfig.model_validate(
        minimal(
            models={"router": {"kind": "bio.kmer-classifier"}},
            pipeline={"stages": ["ingest", "router-train"]},
            stages={
                "ingest": {"kind": "bio.ingest"},
                "router-train": {"kind": "bio.classifier-train", "model": "router"},
            },
        )
    )

    assert config.pipeline.stages == ("ingest", "router-train")
    assert config.model("router") is config.router


def test_a_pipeline_stage_must_be_defined() -> None:
    with pytest.raises(ValidationError, match=r"has no \[stages.train\] section"):
        AppConfig.model_validate(minimal(pipeline={"stages": ["train"]}))


def test_a_stage_must_act_on_a_model_that_exists() -> None:
    with pytest.raises(ValidationError, match="'sharks', which is not a model"):
        AppConfig.model_validate(
            minimal(stages={"t": {"kind": "bio.classifier-train", "model": "sharks"}})
        )


def test_a_model_needs_a_known_kind() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(minimal(models={"x": {"kind": "bio.unknown"}}))


def test_models_are_found_by_name_and_kind() -> None:
    config = AppConfig.model_validate(schema_1())

    with pytest.raises(UnknownClassifierError, match="no 'sharks' model"):
        config.model("sharks")
    with pytest.raises(UnknownClassifierError, match="no 'culicidae' classifier"):
        config.classifier("culicidae")
    with pytest.raises(UnknownClassifierError, match="no 'router' specialist"):
        config.specialist("router")
