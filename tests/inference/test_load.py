"""Tests for prokopton.ml.inference.load and the identify command."""

import json
from pathlib import Path
from typing import Any

import pytest

from bio_app import CLASSIFIER, app_config
from fakes import FakeTracker
from inference_world import AEGYPTI, FakeMarker, FakeRouter
from prokopton.ml import cli
from prokopton.ml.bio.specialists.identify import Specialist
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.load import LoadError, load_identifier
from specialist_world import config_dict, prepared


def app(tmp_path: Path, **specialist: Any) -> AppConfig:
    data = config_dict(tmp_path)
    data["models"]["marker"] = {
        "kind": CLASSIFIER,
        "label": "gene",
        "training": {"reverse_label": "reverse", "registered_model": "marker"},
    }
    data["models"]["culicidae"].update({"router_label": "Diptera", **specialist})
    return app_config(data)


def factory(tracker: FakeTracker) -> Any:
    return lambda _config, _name: tracker


def models() -> FakeTracker:
    tracker = FakeTracker()
    tracker.registry["router", "candidate"] = (FakeRouter(Diptera=0.9, Other=0.1), "5")
    tracker.registry["marker", "candidate"] = (FakeMarker(), "6")
    return tracker


def test_loads_models_and_specialists(tmp_path: Path) -> None:
    prepared(tmp_path)
    config = app(tmp_path)

    loaded = load_identifier(config, factory(models()))

    assert loaded.versions == {
        "marker": "6",
        "router": "5",
        "culicidae": "reference GB999",
    }
    assert isinstance(loaded.identifier.models["culicidae"], Specialist)


def test_skips_specialists_not_used_at_serving(tmp_path: Path) -> None:
    loaded = load_identifier(app(tmp_path, router_label=""), factory(models()))

    assert "culicidae" not in loaded.identifier.models


def test_loads_only_what_the_serving_steps_use(tmp_path: Path) -> None:
    config = app_config(config_dict(tmp_path))  # No marker

    loaded = load_identifier(config, factory(models()))

    assert "marker" not in loaded.identifier.models
    assert "router" in loaded.versions


def test_needs_reference_sets(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="run the reference stage"):
        load_identifier(app(tmp_path), factory(models()))


def test_cli_identify_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared(tmp_path)
    config = app(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *_, **__: config)
    monkeypatch.setattr(
        cli,
        "load_identifier",
        lambda c: load_identifier(c, factory(models())),
    )
    sequence = tmp_path / "query.fasta"
    sequence.write_text(f">q\n{AEGYPTI}\n")

    assert cli.main(["--config", "app.toml", "identify", "--input", str(sequence)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "identified"
    assert output["versions"]["router"] == "5"

    monkeypatch.setattr("sys.stdin.read", lambda: "ACGT")
    assert cli.main(["--config", "app.toml", "identify", "--input", "-"]) == 0
    assert json.loads(capsys.readouterr().out)["reason"] == "too_short"

    assert (
        cli.main(["--config", "app.toml", "identify", "--input", str(tmp_path / "no")])
        == 1
    )
