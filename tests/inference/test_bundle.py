"""Tests for prokopton.ml.inference.bundle and the bundle command."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bio_app import CLASSIFIER, app_config
from fakes import FakeTracker
from inference_world import FakeMarker, FakeRouter
from prokopton.ml import cli
from prokopton.ml.config import AppConfig
from prokopton.ml.inference import bundle
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
    data["models"]["marker"] = {
        "kind": CLASSIFIER,
        "label": "gene",
        "training": {"reverse_label": "reverse", "registered_model": "marker"},
    }
    data["models"]["culicidae"]["router_label"] = "Diptera"
    return app_config(data)


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
        "requirements.txt",
        "marker.joblib",
        "router.joblib",
        "specialists/culicidae/reference.parquet",
        "specialists/culicidae/neighbours.json",
        "specialists/culicidae/index.npz",
    }
    assert manifest.versions["marker"] == "6"
    assert manifest.versions["culicidae"].startswith("GB999, ")
    assert manifest.format == 2
    written = json.loads((bundle_path(config) / "config.json").read_text())
    assert written["schema"] == 4
    assert written["bio"]["source"]["genes"]  # The [bio] section is kept
    assert written["data"]["release"] == "GB999"
    assert manifest.models == {
        "router": "bio.kmer-classifier",
        "marker": "bio.kmer-classifier",
        "culicidae": "bio.specialist",
    }
    assert manifest.steps["input"] == ("router.joblib",)
    assert manifest.steps["culicidae"] == (
        "specialists/culicidae/reference.parquet",
        "specialists/culicidae/neighbours.json",
        "specialists/culicidae/index.npz",
    )
    assert not bundle_path(config).with_name("bundle.part").exists()
    assert manifest.runtime["numpy"] == np.__version__
    requirements = (bundle_path(config) / "requirements.txt").read_text()
    assert f"numpy=={np.__version__}\n" in requirements


def test_skips_packages_that_are_not_installed() -> None:
    assert bundle._installed(("numpy", "not-a-real-package")) == {
        "numpy": np.__version__
    }


def test_refuses_models_saved_with_other_package_versions(tmp_path: Path) -> None:
    _, folder = built(tmp_path)
    path = folder / MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest["runtime"]["numpy"] = "0.0.1"
    path.write_text(json.dumps(manifest))

    with pytest.raises(BundleError, match=r"saved with numpy 0\.0\.1"):
        load_bundle(folder)


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


def test_needs_serving_steps(tmp_path: Path) -> None:
    data = app_config(config_dict(tmp_path)).model_dump(mode="json")
    data["serving"] = {"chain": [], "steps": {}}

    with pytest.raises(BundleError, match="no serving steps"):
        build_bundle(AppConfig.model_validate(data), factory())


def test_refuses_format_1_bundles(tmp_path: Path) -> None:
    _, folder = built(tmp_path)
    manifest = json.loads((folder / MANIFEST_NAME).read_text())
    for key in ("format", "models", "steps"):
        del manifest[key]
    manifest["specialists"] = ["culicidae"]
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest))

    with pytest.raises(
        BundleError, match=r"bundle format 1; .* build the bundle again"
    ):
        load_bundle(folder)


def test_refuses_unknown_formats(tmp_path: Path) -> None:
    _, folder = built(tmp_path)
    manifest = json.loads((folder / MANIFEST_NAME).read_text())
    manifest["format"] = 99
    (folder / MANIFEST_NAME).write_text(json.dumps(manifest))

    with pytest.raises(BundleError, match="bundle format 99"):
        load_bundle(folder)


def test_cli_builds_and_uses_a_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared(tmp_path)
    config = app(tmp_path)
    monkeypatch.setattr(cli, "load_config", lambda *_, **__: config)
    monkeypatch.setattr(
        "prokopton.ml.stages.build_bundle", lambda c: build_bundle(c, factory())
    )

    assert cli.main(["--config", "app.toml", "run", "bundle"]) == 0
    query = tmp_path / "query.txt"
    query.write_text(AEGYPTI)
    folder = str(bundle_path(config))
    assert cli.main(["identify", "--input", str(query), "--bundle", folder]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["answer"]["name"] == "Aedes aegypti"
    assert output["versions"]["marker"] == "6"

    (bundle_path(config) / MANIFEST_NAME).unlink()
    assert cli.main(["identify", "--input", str(query), "--bundle", folder]) == 1
