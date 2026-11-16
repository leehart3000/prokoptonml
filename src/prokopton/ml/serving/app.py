"""The HTTP API: answer one request at a time, from a serving bundle.

Endpoints:

- ``GET /health``: the service is up and its models are loaded.
- ``GET /version``: the app, data release and model versions.
- ``POST /<endpoint>`` (``[serving] endpoint``, such as ``/identify``): the
  request its first serving step accepts (for sequences,
  ``{"sequence": "..."}``); returns the result with its decision trace.

The API's title, description and example request come from the app (its
config, and ``PROKOPTON_API_TITLE``), and its OpenAPI document describes the
app's own request and result. Errors are "problem details" (RFC 9457,
``application/problem+json``), with a readable ``detail``. A declined
request is an answer, not an error.

Settings come from environment variables (see ``Settings.from_env``). If an
API key is set, the endpoint requires it in the ``X-Prokopton-Key`` header:
the public website's Cloudflare Worker adds it (with rate limiting and a
Turnstile check in front), so the API cannot be used directly.

Each request is logged as one line of JSON (outcomes, scores and timings,
never the request itself), for monitoring. Identical requests are answered
from a small in-memory cache.
"""

import hashlib
import hmac
import inspect
import json
import logging
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from prokopton.ml import __version__
from prokopton.ml.inference.bundle import load_bundle
from prokopton.ml.inference.identify import Result
from prokopton.ml.inference.load import Loaded

__all__ = ["Settings", "create_app", "create_app_from_env"]

logger = logging.getLogger("prokopton.ml.serving")

DEFAULT_TITLE = ""
"""No title: the app's own title is used."""


@dataclass(frozen=True, slots=True)
class Settings:
    """How the service runs."""

    bundle: Path
    api_key: str = ""
    cache_size: int = 256
    title: str = DEFAULT_TITLE

    @classmethod
    def from_env(cls) -> "Settings":
        """Read ``PROKOPTON_BUNDLE`` (required), ``PROKOPTON_API_KEY``,
        ``PROKOPTON_CACHE_SIZE`` and ``PROKOPTON_API_TITLE`` (the name shown in
        the API docs, set by each product)."""
        return cls(
            bundle=Path(os.environ["PROKOPTON_BUNDLE"]),
            api_key=os.environ.get("PROKOPTON_API_KEY", ""),
            cache_size=int(os.environ.get("PROKOPTON_CACHE_SIZE", "256")),
            title=os.environ.get("PROKOPTON_API_TITLE", DEFAULT_TITLE),
        )


class Version(BaseModel):
    """What the service is running."""

    app: str
    prokopton_ml: str
    versions: dict[str, str]


class _Cache:
    """The most recent results, by the SHA-256 of the submitted text.

    Requests are served on several threads, so a lock guards the cache.
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._items: OrderedDict[str, Result] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Result | None:
        with self._lock:
            found = self._items.get(key)
            if found is not None:
                self._items.move_to_end(key)
            return found

    def put(self, key: str, result: Result) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._items[key] = result
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)


def _log_line(loaded: Loaded, result: Result, cached: bool) -> dict[str, Any]:
    """What is logged for monitoring: no requests, only outcomes."""
    return {
        "event": "identify",
        "status": result.status,
        "reason": result.reason,
        **loaded.identifier.log_fields(result),
        "level": result.answer.level if result.answer else None,
        "confident": result.answer.confident if result.answer else None,
        "seconds": result.seconds,
        "cached": cached,
    }


def _problem(status: int, detail: str, **extra: Any) -> JSONResponse:
    """An error as "problem details" (RFC 9457)."""
    return JSONResponse(
        {
            "type": "about:blank",
            "title": HTTPStatus(status).phrase,
            "status": status,
            "detail": detail,
            **extra,
        },
        status_code=status,
        media_type="application/problem+json",
        headers={"Cache-Control": "no-store"},
    )


def _field(error: dict[str, Any]) -> str:
    location = [str(part) for part in error.get("loc", ()) if part != "body"]
    return ".".join(location) or "body"


def create_app(settings: Settings, loaded: Loaded | None = None) -> FastAPI:
    """Make the API for the bundle in ``settings`` (loaded now, so the API
    can describe the app's own request and result), or for ``loaded``."""
    found = loaded or load_bundle(settings.bundle)
    config = found.identifier.config
    cache = _Cache(settings.cache_size)
    request_model = found.identifier.request_model()
    logger.info(json.dumps({"event": "started", "versions": found.versions}))

    app = FastAPI(
        title=settings.title or f"{config.title} API",
        version=__version__,
        description=config.serving.description,
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        return _problem(error.status_code, str(error.detail))

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, error: RequestValidationError) -> JSONResponse:
        errors = [
            {"field": _field(e), "message": str(e.get("msg", ""))}
            for e in error.errors()
        ]
        detail = "; ".join(f"{e['field']}: {e['message']}" for e in errors)
        return _problem(422, f"The request is not valid ({detail}).", errors=errors)

    def check_key(
        x_prokopton_key: Annotated[str | None, Header()] = None,
    ) -> None:
        if not settings.api_key:
            return
        if x_prokopton_key is None or not hmac.compare_digest(
            x_prokopton_key, settings.api_key
        ):
            raise HTTPException(status_code=401, detail="Missing or wrong API key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/version")
    def version() -> Version:
        return Version(
            app=config.name, prokopton_ml=__version__, versions=found.versions
        )

    def answer(body: BaseModel) -> Result:
        text = found.identifier.text_of(body)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result = cache.get(key)
        cached = result is not None
        if result is None:
            result = found.identifier.identify(text)
            cache.put(key, result)
        logger.info(json.dumps(_log_line(found, result, cached)))
        return result

    # The request type is the app's, so the signature is set here
    answer.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [
            inspect.Parameter(
                "body",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=request_model,
            )
        ],
        return_annotation=found.identifier.result_type,
    )
    app.add_api_route(
        f"/{config.serving.endpoint}",
        answer,
        methods=["POST"],
        dependencies=[Depends(check_key)],
        response_model=found.identifier.result_type,
        summary=config.serving.description.split(". ")[0],
        operation_id=config.serving.endpoint.replace("-", "_"),
    )
    return app


def create_app_from_env() -> FastAPI:
    """The app for ``uvicorn --factory``, with settings from the environment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings.from_env())
