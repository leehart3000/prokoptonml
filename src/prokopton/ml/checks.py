"""Known-answer checks: requests whose answers must not change.

A product keeps its checks in a folder (such as ``tests/known-answers/``).
Each check is two files with the same name:

- ``<name>.json``: the request body, as sent to ``POST /<endpoint>``;
- ``<name>.expected``: the answer it must give, or just the start of it. An
  answer is the answer's text, or ``declined: <reason>`` for a declined
  request.

``prokopton-ml check BUNDLE CHECKS`` runs every check against a serving
bundle, in this process, or with ``--url`` against a running service (such
as a newly built image, before it is pushed). It reports every check and
fails if any answer differs. A service's API key is read from the
``PROKOPTON_API_KEY`` environment variable, never from the command line.
"""

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from prokopton.ml.config import AppConfig
from prokopton.ml.inference.chain import Result

__all__ = [
    "Answerer",
    "Check",
    "CheckError",
    "Outcome",
    "Sender",
    "bundle_config",
    "read_checks",
    "report",
    "run_local",
    "run_remote",
    "summary",
]

KEY_HEADER = "X-Prokopton-Key"
"""The header that carries the API key (see ``prokopton.ml.serving.app``)."""

TIMEOUT = 120
"""Seconds to wait for each answer from a service (a cold start can be slow)."""


class CheckError(RuntimeError):
    """Raised when the checks cannot be read or run."""


@dataclass(frozen=True)
class Check:
    """One known request and the answer it must give."""

    name: str
    request: dict[str, Any]
    expected: str


@dataclass(frozen=True)
class Outcome:
    """A check's answer."""

    name: str
    answer: str
    expected: str

    @property
    def passed(self) -> bool:
        """The answer starts with the expected text."""
        return self.answer.startswith(self.expected)


class Answerer(Protocol):
    """What answers checks in this process: a serving chain
    (``prokopton.ml.inference.chain.Chain``)."""

    def request_model(self) -> type[BaseModel]: ...

    def text_of(self, request: BaseModel) -> str: ...

    def identify(self, text: str) -> Result: ...


def read_checks(folder: Path) -> list[Check]:
    """Every check in a folder, by name.

    Raises:
        CheckError: If the folder has no checks, a request is not a JSON
            object, or a request has no ``.expected`` file.
    """
    requests = sorted(folder.glob("*.json"))
    if not requests:
        msg = f"No checks (<name>.json and <name>.expected) in {folder}"
        raise CheckError(msg)
    checks = []
    for request in requests:
        expected = request.with_suffix(".expected")
        if not expected.is_file():
            msg = f"{request} has no {expected.name}"
            raise CheckError(msg)
        try:
            body = json.loads(request.read_text("utf-8"))
        except json.JSONDecodeError as error:
            msg = f"{request} is not valid JSON: {error}"
            raise CheckError(msg) from error
        if not isinstance(body, dict):
            msg = f"{request} is not a JSON object"
            raise CheckError(msg)
        text = expected.read_text("utf-8").strip()
        if not text:
            msg = f"{expected} is empty"
            raise CheckError(msg)
        checks.append(Check(request.stem, body, text))
    return checks


def summary(result: dict[str, Any]) -> str:
    """A result as one line: the answer's text, or ``declined: <reason>``."""
    if result.get("status") == "declined":
        return f"declined: {result.get('reason')}"
    answer = result.get("answer") or {}
    return str(answer.get("text", ""))


def run_local(chain: Answerer, checks: list[Check]) -> list[Outcome]:
    """Answer every check with a loaded chain, in this process."""
    model = chain.request_model()
    outcomes = []
    for check in checks:
        try:
            request = model.model_validate(check.request)
        except ValidationError as error:
            answer = f"rejected: {error.error_count()} problem(s) with the request"
        else:
            result = chain.identify(chain.text_of(request))
            answer = summary(result.model_dump(mode="json"))
        outcomes.append(Outcome(check.name, answer, check.expected))
    return outcomes


Sender = Callable[[urllib.request.Request], tuple[int, bytes]]
"""Sends a request; returns the HTTP status and body."""


def _send(request: urllib.request.Request) -> tuple[int, bytes]:
    """Send with the standard library (``run_remote`` checks the scheme)."""
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def run_remote(
    url: str,
    endpoint: str,
    checks: list[Check],
    key: str = "",
    send: Sender = _send,
) -> list[Outcome]:
    """Answer every check with a running service.

    Raises:
        CheckError: If the URL is not http or https, or the service cannot
            be reached.
    """
    if not url.startswith(("http://", "https://")):
        msg = f"Not an http or https URL: {url}"
        raise CheckError(msg)
    target = f"{url.rstrip('/')}/{endpoint}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if key:
        headers[KEY_HEADER] = key
    outcomes = []
    for check in checks:
        request = urllib.request.Request(
            target,
            data=json.dumps(check.request).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            code, body = send(request)
        except OSError as error:
            msg = f"Could not reach {target}: {error}"
            raise CheckError(msg) from error
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {}
        if code == 200 and isinstance(data, dict):
            answer = summary(data)
        else:
            detail = data.get("detail") if isinstance(data, dict) else None
            answer = f"rejected: HTTP {code}" + (f" ({detail})" if detail else "")
        outcomes.append(Outcome(check.name, answer, check.expected))
    return outcomes


def bundle_config(folder: Path) -> AppConfig:
    """A bundle's app config, without loading its models.

    Raises:
        CheckError: If the bundle has no config, or its config is not valid.
    """
    path = folder / "config.json"
    if not path.is_file():
        msg = f"{path} not found; is {folder} a serving bundle?"
        raise CheckError(msg)
    try:
        return AppConfig.model_validate(json.loads(path.read_text("utf-8")))
    except ValueError as error:
        msg = f"{path} is not a valid app config: {error}"
        raise CheckError(msg) from error


def report(outcomes: list[Outcome]) -> tuple[list[str], int]:
    """Lines describing each check, and how many failed."""
    lines = []
    failed = 0
    for outcome in outcomes:
        mark = "ok  " if outcome.passed else "FAIL"
        lines.append(f"{mark} {outcome.name}: {outcome.answer}")
        if not outcome.passed:
            failed += 1
            lines.append(f"     expected: {outcome.expected}")
    return lines, failed
