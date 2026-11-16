"""Tests for known-answer checks: reading them, and answering them in this
process, from a service, and from the command line."""

import io
import json
import logging
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from prokopton.ml import checks, cli
from prokopton.ml.checks import (
    Check,
    CheckError,
    Outcome,
    bundle_config,
    read_checks,
    report,
    run_local,
    run_remote,
    summary,
)
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.chain import Answer, Result
from test_schema_2 import minimal


def write_check(folder: Path, name: str, request: object, expected: str) -> None:
    (folder / f"{name}.json").write_text(json.dumps(request), "utf-8")
    (folder / f"{name}.expected").write_text(expected, "utf-8")


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    found = tmp_path / "checks"
    found.mkdir()
    write_check(found, "b-declined", {"text": "x"}, "declined: too_short\n")
    write_check(found, "a-species", {"text": "ACGT"}, "Aedes aegypti (")
    return found


def identified(text: str) -> Result:
    answer = Answer(level="species", name=text, text=text, confident=True)
    return Result(status="identified", answer=answer)


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str


class FakeChain:
    """Answers "ACGT" with a species and declines anything else."""

    def request_model(self) -> type[BaseModel]:
        return Request

    def text_of(self, request: BaseModel) -> str:
        assert isinstance(request, Request)
        return request.text

    def identify(self, text: str) -> Result:
        if text == "ACGT":
            return identified("Aedes aegypti (100.0% identical).")
        return Result(status="declined", reason="too_short", message="Too short")


# --- Reading checks ---


def test_checks_are_read_in_name_order(folder: Path) -> None:
    found = read_checks(folder)

    assert [check.name for check in found] == ["a-species", "b-declined"]
    assert found[0] == Check("a-species", {"text": "ACGT"}, "Aedes aegypti (")
    assert found[1].expected == "declined: too_short"


def test_a_folder_without_checks_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(CheckError, match="No checks"):
        read_checks(tmp_path)


def test_a_request_without_an_expected_answer_is_an_error(folder: Path) -> None:
    (folder / "a-species.expected").unlink()

    with pytest.raises(CheckError, match=r"has no a-species\.expected"):
        read_checks(folder)


def test_a_request_that_is_not_json_is_an_error(folder: Path) -> None:
    (folder / "a-species.json").write_text("{not json", "utf-8")

    with pytest.raises(CheckError, match="not valid JSON"):
        read_checks(folder)


def test_a_request_that_is_not_an_object_is_an_error(folder: Path) -> None:
    (folder / "a-species.json").write_text("[1, 2]", "utf-8")

    with pytest.raises(CheckError, match="not a JSON object"):
        read_checks(folder)


def test_an_empty_expected_answer_is_an_error(folder: Path) -> None:
    (folder / "a-species.expected").write_text("\n", "utf-8")

    with pytest.raises(CheckError, match="is empty"):
        read_checks(folder)


# --- Summaries and reports ---


def test_a_summary_is_the_answer_text_or_the_reason_declined() -> None:
    assert summary({"status": "identified", "answer": {"text": "Culex"}}) == "Culex"
    assert summary({"status": "declined", "reason": "protein"}) == "declined: protein"
    assert summary({"status": "identified", "answer": None}) == ""


def test_an_answer_passes_when_it_starts_with_the_expected_text() -> None:
    assert Outcome("a", "Aedes aegypti (100.0% identical).", "Aedes aegypti (").passed
    assert not Outcome("a", "Aedes albopictus", "Aedes aegypti").passed


def test_the_report_shows_every_check_and_counts_failures() -> None:
    lines, failed = report(
        [Outcome("a", "Culex", "Culex"), Outcome("b", "Aedes", "Culex pipiens")]
    )

    assert failed == 1
    assert lines == [
        "ok   a: Culex",
        "FAIL b: Aedes",
        "     expected: Culex pipiens",
    ]


# --- Answering in this process ---


def test_checks_are_answered_by_a_chain(folder: Path) -> None:
    outcomes = run_local(FakeChain(), read_checks(folder))

    assert outcomes == [
        Outcome("a-species", "Aedes aegypti (100.0% identical).", "Aedes aegypti ("),
        Outcome("b-declined", "declined: too_short", "declined: too_short"),
    ]


def test_a_request_the_chain_rejects_is_reported(folder: Path) -> None:
    write_check(folder, "c-wrong", {"sequence": "ACGT"}, "Aedes")

    outcome = run_local(FakeChain(), read_checks(folder))[2]

    assert outcome.answer == "rejected: 2 problem(s) with the request"
    assert not outcome.passed


# --- Answering with a service ---


class FakeService:
    """Records what is sent, and gives back a status and body."""

    def __init__(self, code: int = 200, body: object = None) -> None:
        self.code = code
        self.body = body
        self.sent: list[urllib.request.Request] = []

    def __call__(self, request: urllib.request.Request) -> tuple[int, bytes]:
        self.sent.append(request)
        if isinstance(self.body, bytes):
            return self.code, self.body
        data = request.data
        assert isinstance(data, bytes)
        text = json.loads(data)["text"]
        result = FakeChain().identify(text).model_dump(mode="json")
        return self.code, json.dumps(self.body or result).encode("utf-8")


def test_checks_are_sent_to_the_endpoint_with_the_key(folder: Path) -> None:
    service = FakeService()

    outcomes = run_remote(
        "http://localhost:8080/", "identify", read_checks(folder), "k", service
    )

    assert [outcome.passed for outcome in outcomes] == [True, True]
    first = service.sent[0]
    assert first.full_url == "http://localhost:8080/identify"
    assert first.get_method() == "POST"
    assert first.get_header("X-prokopton-key") == "k"
    assert first.get_header("Content-type") == "application/json"


def test_no_key_header_is_sent_without_a_key(folder: Path) -> None:
    service = FakeService()

    run_remote("https://example.org", "predict", read_checks(folder), send=service)

    assert service.sent[0].get_header("X-prokopton-key") is None
    assert service.sent[0].full_url == "https://example.org/predict"


def test_an_error_from_the_service_is_reported_with_its_detail(
    folder: Path,
) -> None:
    service = FakeService(422, {"status": 422, "detail": "The request is not valid"})

    outcome = run_remote("http://x.test", "identify", read_checks(folder), send=service)

    assert outcome[0].answer == "rejected: HTTP 422 (The request is not valid)"


def test_an_error_without_a_detail_is_reported_by_its_status(folder: Path) -> None:
    for body in (b"<html>Bad gateway</html>", b"[1]"):
        service = FakeService(502, body)

        outcome = run_remote(
            "http://x.test", "identify", read_checks(folder), send=service
        )

        assert outcome[0].answer == "rejected: HTTP 502"


def test_a_service_that_cannot_be_reached_is_an_error(folder: Path) -> None:
    def refuse(_: urllib.request.Request) -> tuple[int, bytes]:
        raise ConnectionRefusedError("refused")

    with pytest.raises(CheckError, match=r"Could not reach http://x\.test/identify"):
        run_remote("http://x.test", "identify", read_checks(folder), send=refuse)


def test_only_http_and_https_urls_are_used(folder: Path) -> None:
    with pytest.raises(CheckError, match="Not an http or https URL"):
        run_remote("file:///etc/passwd", "identify", read_checks(folder))


class FakeResponse(io.BytesIO):
    status = 200


def test_requests_are_sent_with_the_standard_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[float] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        seen.append(timeout)
        if request.full_url.endswith("/bad"):
            raise urllib.error.HTTPError(
                request.full_url, 422, "Unprocessable", Message(), io.BytesIO(b"{}")
            )
        return FakeResponse(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    good = checks._send(urllib.request.Request("http://x.test/good"))
    bad = checks._send(urllib.request.Request("http://x.test/bad"))

    assert good == (200, b'{"ok": true}')
    assert bad == (422, b"{}")
    assert seen == [checks.TIMEOUT, checks.TIMEOUT]


# --- A bundle's config ---


def app_config(**extra: Any) -> AppConfig:
    return AppConfig.model_validate(minimal(schema=3, **extra))


def write_bundle(folder: Path, config: AppConfig) -> Path:
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text(config.model_dump_json(by_alias=True), "utf-8")
    return folder


def test_a_bundles_config_is_read_without_its_models(tmp_path: Path) -> None:
    serving = {"endpoint": "identify", "chain": [], "steps": {}}
    bundle = write_bundle(tmp_path / "bundle", app_config(serving=serving))

    assert bundle_config(bundle).serving.endpoint == "identify"


def test_a_folder_without_a_config_is_not_a_bundle(tmp_path: Path) -> None:
    with pytest.raises(CheckError, match=r"is .* a serving bundle"):
        bundle_config(tmp_path)


def test_a_bundle_with_a_broken_config_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"name": 1}', "utf-8")

    with pytest.raises(CheckError, match="not a valid app config"):
        bundle_config(tmp_path)


# --- The command line ---


class FakeLoaded:
    identifier = FakeChain()


def test_the_command_checks_a_bundle(
    folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loaded: list[Path] = []

    def load_bundle(bundle: Path) -> FakeLoaded:
        loaded.append(bundle)
        return FakeLoaded()

    monkeypatch.setattr(cli, "load_bundle", load_bundle)

    assert main(["check", str(tmp_path / "bundle"), str(folder)]) == 0
    assert loaded == [tmp_path / "bundle"]
    output = capsys.readouterr().out
    assert "ok   a-species: Aedes aegypti (100.0% identical)." in output
    assert "ok   b-declined: declined: too_short" in output


def test_the_command_fails_when_an_answer_differs(
    folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (folder / "a-species.expected").write_text("Culex pipiens", "utf-8")
    monkeypatch.setattr(cli, "load_bundle", lambda _: FakeLoaded())

    assert main(["check", str(tmp_path), str(folder)]) == 1
    assert "     expected: Culex pipiens" in capsys.readouterr().out
    assert "1 of 2 checks gave a different answer" in caplog.text


def test_the_command_checks_a_service_with_the_key_from_the_environment(
    folder: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    serving = {"endpoint": "identify", "chain": [], "steps": {}}
    bundle = write_bundle(tmp_path / "bundle", app_config(serving=serving))
    service = FakeService()
    calls: list[tuple[str, str, str]] = []

    def remote(
        url: str, endpoint: str, found: list[Check], key: str = ""
    ) -> list[Outcome]:
        calls.append((url, endpoint, key))
        return run_remote(url, endpoint, found, key, service)

    monkeypatch.setattr(cli, "run_remote", remote)
    monkeypatch.setenv("PROKOPTON_API_KEY", "secret-test-key")
    caplog.set_level(logging.INFO, logger="prokopton.ml")

    code = main(["check", str(bundle), str(folder), "--url", "http://x.test"])

    assert code == 0
    assert calls == [("http://x.test", "identify", "secret-test-key")]
    assert "all 2 checks gave their expected answers" in caplog.text
    assert "secret-test-key" not in caplog.text
