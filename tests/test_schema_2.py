"""Tests for config schema 2: models, the pipeline, and reading schema 1."""

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from prokopton.ml.bio.config import (
    ClassifierConfig,
    MarkerStepConfig,
    SpecialistConfig,
    SpecialistStepConfig,
)
from prokopton.ml.config import (
    AppConfig,
    OtherStepConfig,
    UnknownClassifierError,
    convert_schema_1,
    convert_schema_2,
    default_stages,
    upgrade,
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

    assert config.schema_version == 3
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
    assert json.loads(config.model_dump_json(by_alias=True))["schema"] == 3


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


def test_inference_settings_move_into_the_serving_steps() -> None:
    data = schema_1()
    data["inference"]["route_threshold"] = 0.2

    serving = AppConfig.model_validate(data).serving

    assert serving.chain == ("input", "marker", "router", "culicidae")
    assert serving.endpoint == "identify"  # What older apps served
    marker = serving.steps["marker"]
    assert isinstance(marker, MarkerStepConfig)
    assert marker.unsupported_note == data["inference"]["unsupported_note"]
    specialist = serving.steps["culicidae"]
    assert isinstance(specialist, SpecialistStepConfig)
    assert specialist.route_threshold == 0.2


def test_schema_2_without_marker_or_inference_gets_default_steps() -> None:
    data = convert_schema_1(schema_1())
    del data["models"]["marker"], data["inference"]
    data["stages"] = {
        k: v for k, v in data["stages"].items() if v.get("model") != "marker"
    }
    data["pipeline"]["stages"] = list(data["stages"])

    steps = convert_schema_2(data)["serving"]["steps"]

    assert list(steps) == ["input", "router", "culicidae"]
    assert steps["input"] == {"kind": "bio.sequence-input", "model": "router"}
    empty = convert_schema_2({"models": {}})["serving"]
    assert (empty["chain"], empty["steps"]) == ([], {})


def test_upgrade_leaves_schema_3_alone() -> None:
    data = minimal(serving={"endpoint": "identify"})
    data["schema"] = 3

    assert upgrade(data) is data


def serving(**fields: Any) -> dict[str, Any]:
    return minimal(
        schema=3,
        models={
            "router": {"kind": "bio.kmer-classifier"},
            "f": {"kind": "bio.specialist", "target": "F", "background": "B"},
        },
        serving=fields,
    )


def test_serving_steps_are_checked() -> None:
    with pytest.raises(ValidationError, match=r"no \[serving.steps.input\] section"):
        AppConfig.model_validate(serving(chain=["input"]))
    with pytest.raises(ValidationError, match="'sharks', which is not a model"):
        AppConfig.model_validate(
            serving(steps={"r": {"kind": "bio.router", "model": "sharks"}})
        )
    with pytest.raises(ValidationError, match=r"needs a bio\.kmer-classifier model"):
        AppConfig.model_validate(
            serving(steps={"r": {"kind": "bio.router", "model": "f"}})
        )


def test_other_packages_steps_are_accepted() -> None:
    config = AppConfig.model_validate(
        serving(
            chain=["x", "y"],
            steps={
                "x": {"kind": "shop.thing"},
                "y": {"kind": "shop.other", "model": "f"},
            },
        )
    )

    assert isinstance(config.serving.steps["x"], OtherStepConfig)


def test_a_dumped_schema_2_config_is_read() -> None:
    # As in the config.json of a bundle made by prokopton-ml 0.2.0a2
    data = convert_schema_1(schema_1())
    data["schema_version"] = data.pop("schema")

    config = AppConfig.model_validate(data)

    assert config.schema_version == 3
    assert config.serving.chain == ("input", "marker", "router", "culicidae")
