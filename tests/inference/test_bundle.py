"""Tests for prokopton.ml.inference.bundle and the bundle command."""

import json
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeTracker
from inference_world import FakeMarker, FakeRouter
from prokopton.ml import cli
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.bundle import (
    MANIFEST_NAME,
    BundleError,
    build_bundle,
    bundle_path,
    load_bundle,
)
from specialist_world import AEGYPTI, config_dict, prepared


def app(tmp_path: Path) -> AppConfig:
    data = config_dict(tmp_path)
    data["marker"] = {
        "label": "gene",
        "training": {"reverse_label": "reverse", "registered_model": "marker"},
    }
    data["specialists"]["culicidae"]["router_label"] = "Diptera"
    return AppConfig.model_validate(data)


def factory() -> Any:
    tracker = FakeTracker()
    tracker.registry["router", "candidate"] = (FakeRouter(Diptera=0.9, Other=0.1), "5")
    tracker.registry["marker", "candidate"] = (FakeMarker(), "6")
    return lambda _config, _name: tracker


def built(tmp_path: Path) -> tuple[AppConfig, Path]:
    prepared(tmp_path)
    config = app(tmp_path)
    build_bundle(config, factory())
    return config, bundle_path(config)


def test_builds_a_bundle_with_checksums(tmp_path: Path) -> None:
    prepared(tmp_path)
    config = app(tmp_path)
    build_bundle(config, factory())

    # Building again replaces the bundle
    manifest = build_bundle(config, factory())

    assert set(manifest.files) == {
        "config.json",
        "marker.joblib",
        "router.joblib",
        "specialists/culicidae/reference.parquet",
        "specialists/culicidae/neighbours.json",
        "specialists/culicidae/index.npz",
    }
    assert manifest.versions["marker"] == "6"
    assert manifest.versions["culicidae"].startswith("GB999, ")
    assert manifest.specialists == ("culicidae",)
    assert not bundle_path(config).with_name("bundle.part").exists()


def test_loads_a_bundle_and_identifies(tmp_path: Path) -> None:
    _, folder = built(tmp_path)

    loaded = load_bundle(folder)

    assert loaded.versions["router"] == "5"
    result = loaded.identifier.identify(AEGYPTI)
    assert result.answer is not None
    assert result.answer.name == "Aedes aegypti"


def test_refuses_a_missing_or_altered_bundle(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="build the bundle first"):
        load_bundle(tmp_path)

    _, folder = built(tmp_path)
    (folder / "router.joblib").write_bytes(b"not the model")
    with pytest.raises(BundleError, match="does not match"):
        load_bundle(folder)


def test_needs_a_marker_classifier(tmp_path: Path) -> None:
    with pytest.raises(BundleError, match="no marker classifier"):
        build_bundle(AppConfig.model_validate(config_dict(tmp_path)), factory())


def test_cli_builds_and_uses_a_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared(tmp_path)
    config = app(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *_, **__: config)
    monkeypatch.setattr(cli, "build_bundle", lambda c: build_bundle(c, factory()))

    assert cli.main(["bundle", "app.toml"]) == 0
    query = tmp_path / "query.txt"
    query.write_text(AEGYPTI)
    folder = str(bundle_path(config))
    assert (
        cli.main(["identify", "app.toml", "--input", str(query), "--bundle", folder])
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["name"] == "Aedes aegypti"
    assert output["versions"]["marker"] == "6"

    (bundle_path(config) / MANIFEST_NAME).unlink()
    assert (
        cli.main(["identify", "app.toml", "--input", str(query), "--bundle", folder])
        == 1
    )
