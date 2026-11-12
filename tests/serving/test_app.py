"""Tests for prokopton.ml.serving.app."""

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import FakeTracker
from inference_world import AEGYPTI as BARCODE
from inference_world import CYTB, FakeMarker, FakeRouter
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.bundle import build_bundle, bundle_path
from prokopton.ml.inference.identify import Identifier
from prokopton.ml.inference.load import Loaded
from prokopton.ml.serving import app as serving
from prokopton.ml.serving.app import Settings, _Cache, create_app
from specialist_world import AEGYPTI, config_dict, prepared


def app_config(tmp_path: Path) -> AppConfig:
    data = config_dict(tmp_path)
    data["marker"] = {
        "label": "gene",
        "training": {"reverse_label": "reverse", "registered_model": "marker"},
    }
    data["specialists"]["culicidae"]["router_label"] = "Diptera"
    return AppConfig.model_validate(data)


def fake_loaded(tmp_path: Path) -> Loaded:
    config = app_config(tmp_path)
    identifier = Identifier(
        config, FakeMarker(), FakeRouter(Diptera=0.2, Other=0.8), {}
    )
    return Loaded(identifier=identifier, versions={"router": "5"})


def client(tmp_path: Path, api_key: str = "", cache_size: int = 256) -> TestClient:
    settings = Settings(
        bundle=tmp_path / "unused", api_key=api_key, cache_size=cache_size
    )
    return TestClient(create_app(settings, fake_loaded(tmp_path)))


def test_health_and_version(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        assert api.get("/health").json() == {"status": "ok"}
        found = api.get("/version").json()
    assert found["app"] == "testapp"
    assert found["versions"] == {"router": "5"}


def test_identifies_and_logs_without_the_sequence(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="prokopton.ml.serving")
    with client(tmp_path) as api:
        first = api.post("/identify", json={"sequence": BARCODE})
        again = api.post("/identify", json={"sequence": BARCODE})

    assert first.status_code == 200
    assert first.json()["answer"]["name"] == "Other"
    assert again.json() == first.json()
    lines = [json.loads(r.message) for r in caplog.records if '"identify"' in r.message]
    assert [line["cached"] for line in lines] == [False, True]
    assert lines[0]["router_label"] == "Other"
    assert lines[0]["specialist_outcome"] is None
    assert BARCODE not in caplog.text


def test_declined_sequences_are_answered_not_errors(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        found = api.post("/identify", json={"sequence": CYTB})

    assert found.status_code == 200
    assert found.json()["reason"] == "unsupported_marker"


def test_bad_requests_are_refused(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        assert api.post("/identify", json={"sequence": ""}).status_code == 422
        too_long = "A" * (serving.MAX_SEQUENCE_CHARACTERS + 1)
        assert api.post("/identify", json={"sequence": too_long}).status_code == 422
        assert api.post("/identify", json={"other": "x"}).status_code == 422


def test_the_api_key_is_required_when_set(tmp_path: Path) -> None:
    with client(tmp_path, api_key="secret") as api:
        assert api.post("/identify", json={"sequence": BARCODE}).status_code == 401
        wrong = {"X-Prokopton-Key": "guess"}
        assert (
            api.post("/identify", json={"sequence": BARCODE}, headers=wrong).status_code
            == 401
        )
        right = {"X-Prokopton-Key": "secret"}
        assert (
            api.post("/identify", json={"sequence": BARCODE}, headers=right).status_code
            == 200
        )
        assert api.get("/health").status_code == 200


def test_the_cache_keeps_the_most_recent(tmp_path: Path) -> None:
    result = fake_loaded(tmp_path).identifier.identify(BARCODE)
    cache = _Cache(2)
    for key in ("a", "b"):
        cache.put(key, result)
    cache.get("a")
    cache.put("c", result)

    assert cache.get("b") is None
    assert cache.get("a") is result
    off = _Cache(0)
    off.put("a", result)
    assert off.get("a") is None


def test_logs_the_specialist_outcome(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared(tmp_path)
    config = app_config(tmp_path)
    tracker = FakeTracker()
    tracker.registry["router", "candidate"] = (FakeRouter(Diptera=0.9, Other=0.1), "5")
    tracker.registry["marker", "candidate"] = (FakeMarker(), "6")
    build_bundle(config, lambda _c, _n: tracker)
    monkeypatch.setenv("PROKOPTON_BUNDLE", str(bundle_path(config)))
    monkeypatch.setenv("PROKOPTON_API_KEY", "")
    caplog.set_level(logging.INFO, logger="prokopton.ml.serving")

    # The app from the environment loads the bundle at start-up
    with TestClient(serving.create_app_from_env()) as api:
        found = api.post("/identify", json={"sequence": AEGYPTI}).json()

    assert found["answer"]["name"] == "Aedes aegypti"
    lines = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    assert lines[0]["event"] == "started"
    line: dict[str, Any] = lines[-1]
    assert (line["specialist_outcome"], line["identity"]) == ("species", 1.0)


def test_the_docs_show_a_real_example(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        schema = api.get("/openapi.json").json()

    request = schema["components"]["schemas"]["IdentifyRequest"]
    assert request["examples"] == [{"sequence": serving.EXAMPLE}]
    assert serving.EXAMPLE.startswith(">HQ945416.1")
