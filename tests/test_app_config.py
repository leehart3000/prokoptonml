"""Tests for an app's models, pipeline and serving steps (config schema 4),
and for refusing schemas that are no longer read."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from bio_app import dnabarcode, minimal
from prokopton.ml.bio.config import (
    ClassifierConfig,
    MarkerStepConfig,
    SpecialistConfig,
    SpecialistStepConfig,
    classifier_settings,
    specialist_settings,
)
from prokopton.ml.config import (
    AppConfig,
    StepEntry,
    UnknownClassifierError,
    upgrade,
)

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


def test_the_dnabarcode_config_has_its_models_and_every_stage() -> None:
    config = AppConfig.model_validate(dnabarcode())

    assert config.schema_version == 4
    assert list(config.models) == ["router", "marker", "culicidae"]
    assert isinstance(config.model_settings("router"), ClassifierConfig)
    assert isinstance(config.model_settings("culicidae"), SpecialistConfig)
    assert config.pipeline.stages == DNABARCODE_STAGES
    assert config.stages["router-train"].kind == "bio.classifier-train"
    assert config.stages["router-train"].model == "router"
    assert config.stages["culicidae-anchors"].kind == "bio.specialist-anchors"
    assert config.stages["bundle"].model is None


def test_settings_are_read_as_written() -> None:
    data = dnabarcode()
    config = AppConfig.model_validate(data)

    assert classifier_settings(config, "router").training.k_values == tuple(
        data["models"]["router"]["training"]["k_values"]
    )
    assert (
        classifier_settings(config, "marker").label
        == (data["models"]["marker"]["label"])
    )
    culicidae = specialist_settings(config, "culicidae")
    assert (
        culicidae.species_identity == (data["models"]["culicidae"]["species_identity"])
    )


def test_a_dumped_config_reads_back_the_same() -> None:
    config = AppConfig.model_validate(dnabarcode())

    assert AppConfig.model_validate_json(config.model_dump_json()) == config
    assert json.loads(config.model_dump_json(by_alias=True))["schema"] == 4


@pytest.mark.parametrize("schema", [1, 2])
def test_schemas_1_and_2_are_no_longer_read(schema: int) -> None:
    data = {**minimal(), "schema": schema}
    if schema == 1:
        del data["schema"]  # Schema 1 files didn't say

    with pytest.raises(ValidationError, match=f"schema {schema}, which is no longer"):
        AppConfig.model_validate(data)
    with pytest.raises(ValueError, match=r"with prokopton-ml 0\.2\.0a7"):
        upgrade(data)


def test_a_dumped_schema_2_config_is_refused() -> None:
    # As in the config.json of a bundle made by prokopton-ml 0.2.0a2
    data = {**minimal(), "schema_version": 2}
    del data["schema"]

    with pytest.raises(ValidationError, match="schema 2, which is no longer read"):
        AppConfig.model_validate(data)


def test_upgrade_leaves_schema_4_alone() -> None:
    data = {"schema": 4, "name": "x"}

    assert upgrade(data) is data


def test_a_pipeline_is_read_as_written() -> None:
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
    assert config.model_settings("router") is classifier_settings(config, "router")
    assert isinstance(config.model_settings("router"), ClassifierConfig)


def test_a_pipeline_stage_must_be_defined() -> None:
    with pytest.raises(ValidationError, match=r"has no \[stages.train\] section"):
        AppConfig.model_validate(minimal(pipeline={"stages": ["train"]}))


def test_a_stage_must_act_on_a_model_that_exists() -> None:
    with pytest.raises(ValidationError, match="'sharks', which is not a model"):
        AppConfig.model_validate(
            minimal(stages={"t": {"kind": "bio.classifier-train", "model": "sharks"}})
        )


def test_a_model_with_settings_needs_a_kind_that_declares_them() -> None:
    with pytest.raises(ValidationError, match="no settings are known for kind"):
        AppConfig.model_validate(minimal(models={"x": {"kind": "bio.unknown", "k": 8}}))
    # A kind without settings is checked when it is used (by its registry)
    config = AppConfig.model_validate(minimal(models={"x": {"kind": "shop.x"}}))
    assert config.model_settings("x") is config.models["x"]


def test_models_are_found_by_name_and_kind() -> None:
    config = AppConfig.model_validate(dnabarcode())

    with pytest.raises(UnknownClassifierError, match="no 'sharks' model"):
        config.model_settings("sharks")
    with pytest.raises(UnknownClassifierError, match="no 'culicidae' classifier"):
        classifier_settings(config, "culicidae")
    with pytest.raises(UnknownClassifierError, match="no 'router' specialist"):
        specialist_settings(config, "router")


def test_the_serving_steps_have_their_kinds_settings() -> None:
    data = dnabarcode()
    data["serving"]["steps"]["culicidae"]["route_threshold"] = 0.2

    config = AppConfig.model_validate(data)

    assert config.serving.chain == ("input", "marker", "router", "culicidae")
    assert config.serving.endpoint == "identify"
    marker = config.step_settings("marker")
    assert isinstance(marker, MarkerStepConfig)
    assert (
        marker.unsupported_note
        == (data["serving"]["steps"]["marker"]["unsupported_note"])
    )
    specialist = config.step_settings("culicidae")
    assert isinstance(specialist, SpecialistStepConfig)
    assert specialist.route_threshold == 0.2


def serving(**fields: Any) -> dict[str, Any]:
    return minimal(
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

    assert isinstance(config.serving.steps["x"], StepEntry)
    assert config.step_settings("x") is config.serving.steps["x"]
