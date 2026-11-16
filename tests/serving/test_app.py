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
from prokopton.ml.bio.serving import MAX_SEQUENCE_CHARACTERS
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


EXAMPLE = ">An example\n" + BARCODE


def fake_loaded(tmp_path: Path) -> Loaded:
    # Without the specialist step: these tests don't load a specialist
    data = app_config(tmp_path).model_dump(mode="json")
    data["serving"]["chain"].remove("culicidae")
    del data["serving"]["steps"]["culicidae"]
    data["serving"]["steps"]["input"]["example"] = EXAMPLE
    config = AppConfig.model_validate(data)
    identifier = Identifier(
        config, {"marker": FakeMarker(), "router": FakeRouter(Diptera=0.2, Other=0.8)}
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
    # Each step logs its own fields: this chain has no specialist step
    assert "specialist_outcome" not in lines[0]
    assert "strand" in lines[0]
    assert BARCODE not in caplog.text


def test_declined_sequences_are_answered_not_errors(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        found = api.post("/identify", json={"sequence": CYTB})

    assert found.status_code == 200
    assert found.json()["reason"] == "unsupported_marker"


def test_bad_requests_are_refused(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        assert api.post("/identify", json={"sequence": ""}).status_code == 422
        too_long = "A" * (MAX_SEQUENCE_CHARACTERS + 1)
        assert api.post("/identify", json={"sequence": too_long}).status_code == 422
        found = api.post("/identify", json={"other": "x"})

    # Errors are "problem details" (RFC 9457), with a readable detail
    assert found.status_code == 422
    assert found.headers["content-type"] == "application/problem+json"
    problem = found.json()
    assert (problem["type"], problem["title"], problem["status"]) == (
        "about:blank",
        "Unprocessable Content",
        422,
    )
    assert problem["detail"].startswith("The request is not valid (")
    assert {e["field"] for e in problem["errors"]} == {"sequence", "other"}


def test_the_api_key_is_required_when_set(tmp_path: Path) -> None:
    with client(tmp_path, api_key="secret") as api:
        assert api.post("/identify", json={"sequence": BARCODE}).status_code == 401
        wrong = {"X-Prokopton-Key": "guess"}
        assert (
            api.post("/identify", json={"sequence": BARCODE}, headers=wrong).status_code
            == 401
        )
        problem = api.post("/identify", json={"sequence": BARCODE}).json()
        assert (problem["status"], problem["detail"]) == (
            401,
            "Missing or wrong API key",
        )
        right = {"X-Prokopton-Key": "secret"}
        assert (
            api.post("/identify", json={"sequence": BARCODE}, headers=right).status_code
            == 200
        )
        # The key is checked first, so any request without it gets 401 (the
        # deploy workflow relies on this, as it doesn't know the app's request)
        assert api.post("/identify", json={}).status_code == 401
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
    monkeypatch.setenv("PROKOPTON_API_TITLE", "DNAbarcode.org API")
    caplog.set_level(logging.INFO, logger="prokopton.ml.serving")

    # The app from the environment loads the bundle at start-up
    with TestClient(serving.create_app_from_env()) as api:
        found = api.post("/identify", json={"sequence": AEGYPTI}).json()

    assert found["answer"]["name"] == "Aedes aegypti"
    assert api.get("/openapi.json").json()["info"]["title"] == "DNAbarcode.org API"
    lines = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
    assert lines[0]["event"] == "started"
    line: dict[str, Any] = lines[-1]
    assert (line["specialist_outcome"], line["identity"]) == ("species", 1.0)


def test_the_docs_describe_the_apps_own_request_and_result(tmp_path: Path) -> None:
    with client(tmp_path) as api:
        schema = api.get("/openapi.json").json()

    assert schema["info"]["title"] == "Test app API"
    assert schema["info"]["description"].startswith("Identify a DNA barcode.")
    request = schema["components"]["schemas"]["IdentifyRequest"]
    assert request["examples"] == [{"sequence": EXAMPLE}]
    operation = schema["paths"]["/identify"]["post"]
    result = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert result == {"$ref": "#/components/schemas/SequenceResult"}
    assert "strand" in schema["components"]["schemas"]["SequenceResult"]["properties"]


def test_the_endpoint_is_the_apps(tmp_path: Path) -> None:
    loaded = fake_loaded(tmp_path)
    data = loaded.identifier.config.model_dump(mode="json")
    data["serving"]["endpoint"] = "predict"
    data["serving"]["steps"]["input"]["example"] = ""
    config = AppConfig.model_validate(data)
    identifier = Identifier(config, loaded.identifier.models)
    settings = Settings(bundle=tmp_path / "unused")
    app = create_app(settings, Loaded(identifier=identifier, versions={}))

    with TestClient(app) as api:
        assert api.post("/predict", json={"sequence": BARCODE}).status_code == 200
        assert api.post("/identify", json={"sequence": BARCODE}).status_code == 404
        schema = api.get("/openapi.json").json()

    assert schema["components"]["schemas"]["PredictRequest"]["examples"] == []
