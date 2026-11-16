"""Tests for pipeline stages: the registry, plans, settings hashes and records."""

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from prokopton.ml import __version__, stages
from prokopton.ml.bio.sources.refseq import AnchorError
from prokopton.ml.bio.stages import bio_kinds
from prokopton.ml.config import AppConfig
from prokopton.ml.stages import (
    Registry,
    StageContext,
    StageError,
    StageKind,
    core_kinds,
    entry_point_kinds,
    plan,
    read_record,
    record_path,
    run_stage,
    settings_hash,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def app_data(tmp_path: Path) -> dict[str, Any]:
    with (FIXTURES / "dnabarcode.toml").open("rb") as handle:
        config = AppConfig.model_validate(tomllib.load(handle))
    data = config.model_dump(mode="json")
    data["data"]["paths"]["models_dir"] = str(tmp_path)
    return data


def app(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(app_data(tmp_path))


def fake(kind: str = "core.fake", **fields: Any) -> StageKind:
    return StageKind(kind, "A fake stage.", lambda c: f"ran {c.name}", **fields)


def platform() -> Registry:
    return Registry([*core_kinds(), *bio_kinds()])


def test_every_dnabarcode_stage_has_a_kind() -> None:
    config = app(Path("unused"))

    platform().check(config)

    assert platform()["bio.ingest"].description.startswith("Download")


def test_kinds_are_unique_and_known() -> None:
    with pytest.raises(StageError, match="provided twice"):
        Registry([fake(), fake()])
    with pytest.raises(StageError, match=r"kind 'x' \(known kinds: bio\."):
        platform()["x"]


def test_the_registry_collects_expected_errors() -> None:
    registry = Registry([fake(errors=(KeyError,)), fake("core.other", errors=())])

    assert registry.errors == (KeyError,)


@pytest.mark.parametrize(
    ("stage", "message"),
    [
        ({"kind": "core.bundle", "model": "router"}, "does not act on a model"),
        ({"kind": "bio.classifier-train"}, "needs a model"),
        (
            {"kind": "bio.classifier-train", "model": "culicidae"},
            "acts on a bio.kmer-classifier model, but 'culicidae' is a bio.specialist",
        ),
    ],
)
def test_stages_must_fit_their_kind(
    tmp_path: Path, stage: dict[str, str], message: str
) -> None:
    data = app_data(tmp_path)
    data["stages"]["odd"] = stage

    with pytest.raises(StageError, match=message):
        platform().check(AppConfig.model_validate(data))


def test_installed_packages_can_add_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    point = SimpleNamespace(load=lambda: lambda: [fake("shop.thing")])
    monkeypatch.setattr(stages, "entry_points", lambda group: [point])

    assert [kind.kind for kind in entry_point_kinds()] == ["shop.thing"]


def test_plans(tmp_path: Path) -> None:
    config = app(tmp_path)
    order = list(config.pipeline.stages)

    assert plan(config, ["split", "ingest"]) == ["split", "ingest"]
    assert plan(config, run_all=True) == order
    assert plan(config, start="bundle") == ["bundle"]
    with pytest.raises(StageError, match="no stage named 'sharks'"):
        plan(config, ["sharks"])
    with pytest.raises(StageError, match="'sharks' is not in the pipeline"):
        plan(config, start="sharks")
    with pytest.raises(StageError, match="only one of them"):
        plan(config)
    with pytest.raises(StageError, match="only one of them"):
        plan(config, ["split"], run_all=True)


def test_settings_hashes_follow_the_settings_a_stage_uses(tmp_path: Path) -> None:
    before = app(tmp_path)
    data = app_data(tmp_path)
    data["models"]["router"]["training"]["threads"] = 3
    data["validation"]["min_length"] = 101
    after = AppConfig.model_validate(data)
    registry = platform()

    def changed(name: str) -> bool:
        kind = registry[before.stages[name].kind]
        return settings_hash(before, name, kind) != settings_hash(after, name, kind)

    assert changed("router-train")
    assert changed("validate")
    assert not changed("ingest")
    assert not changed("marker-train")


def test_a_run_is_recorded(tmp_path: Path) -> None:
    data = app_data(tmp_path)
    data["stages"]["fake"] = {"kind": "core.fake"}
    config = AppConfig.model_validate(data)
    assert read_record(config, "fake") is None

    assert run_stage(Registry([fake()]), config, "fake") == "ran fake"

    record = read_record(config, "fake")
    assert record is not None
    assert (record.summary, record.platform_version) == ("ran fake", __version__)
    record_path(config, "fake").write_text("{}")
    assert read_record(config, "fake") is None


def test_the_bundle_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = app(tmp_path)
    built = SimpleNamespace(versions={"router": "5"})
    monkeypatch.setattr(stages, "build_bundle", lambda c: built)

    summary = platform()["core.bundle"].run(StageContext(config, "bundle", None))

    assert summary.startswith("serving bundle {'router': '5'} written to")


def test_anchors_need_settings(tmp_path: Path) -> None:
    data = app_data(tmp_path)
    data["models"]["culicidae"]["anchors"] = None
    config = AppConfig.model_validate(data)
    anchors = platform()["bio.specialist-anchors"]

    with pytest.raises(AnchorError, match="Set anchors and anchor_taxon"):
        anchors.run(StageContext(config, "culicidae-anchors", "culicidae"))
