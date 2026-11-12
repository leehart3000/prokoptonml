"""The HTTP API: identify one sequence per request, from a serving bundle.

Endpoints:

- ``GET /health``: the service is up and its models are loaded.
- ``GET /version``: the app, data release and model versions.
- ``POST /identify``: body ``{"sequence": "..."}`` (plain or one FASTA
  record); returns the result with its decision trace.

Settings come from environment variables (see ``Settings.from_env``). If an
API key is set, ``/identify`` requires it in the ``X-Prokopton-Key``
header: the public website's Cloudflare Worker adds it (with rate limiting
and a Turnstile check in front), so the API cannot be used directly.

Each request is logged as one line of JSON (lengths, outcomes, scores and
timings, never the sequence), for monitoring. Identical sequences are
answered from a small in-memory cache.
"""

import hashlib
import hmac
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from prokopton.ml import __version__
from prokopton.ml.inference.bundle import load_bundle
from prokopton.ml.inference.identify import Result
from prokopton.ml.inference.load import Loaded

__all__ = ["Settings", "create_app", "create_app_from_env"]

logger = logging.getLogger("prokopton.ml.serving")

MAX_SEQUENCE_CHARACTERS = 30_000
"""Longer bodies are refused before any work (the identifier's own limit is
in bases, after cleaning)."""


@dataclass(frozen=True, slots=True)
class Settings:
    """How the service runs."""

    bundle: Path
    api_key: str = ""
    cache_size: int = 256

    @classmethod
    def from_env(cls) -> "Settings":
        """Read ``PROKOPTON_BUNDLE`` (required), ``PROKOPTON_API_KEY`` and
        ``PROKOPTON_CACHE_SIZE``."""
        return cls(
            bundle=Path(os.environ["PROKOPTON_BUNDLE"]),
            api_key=os.environ.get("PROKOPTON_API_KEY", ""),
            cache_size=int(os.environ.get("PROKOPTON_CACHE_SIZE", "256")),
        )


EXAMPLE = (
    ">HQ945416.1 Aedes aegypti COI barcode\n"
    "AACTTTATATTTCATTTTTGGAGTATGATCCGGAATAGTCGGAACTTCTCTAAGAATTTTAATTCGTGCT"
    "GAACTTAGCCACCCTGGTATATTTATTGGGAATGACCAAATTTATAATGTAATTGTAACAGCTCATGCAT"
    "TTATTATAATTTTCTTTATAGTAATACCAATTATAATTGGAGGATTTGGAAATTGATTAGTTCCTTTAAT"
    "ATTAGGAGCCCCTGATATAGCCTTTCCTCGAATAAATAATATAAGTTTTTGAATACTACCTCCTTCATTG"
    "ACTCTTCTATTATCAAGCTCAATAGTAGAAAATGGGGCAGGAACTGGGTGAACAGTTTATCCTCCTCTCT"
    "CTTCAGGAACAGCTCATGCTGGAGCTTCTGTTGATTTAGCTATTTTTTCTCTTCATTTAGCTGGAATTTC"
    "CTCAATTTTAGGGGCAGTAAATTTTATTACAACTGTAATTAATATACGATCGTCAGGAATTACTTTAGAC"
    "CGACTACCTTTATTTGTTTGATCTGTAGTTATTACAGCTATCTTATTACTTCTTTCTCTTCCTGTTTTAG"
    "CTGGAGCTATTACTATGTTATTAACAGACCGAAACTTAAATACATCTTTCTTTGATCCAATCGGAGGAGG"
    "AGATCCTATTTTATACCAACACTTATTC\n"
)
"""A real Aedes aegypti COI barcode (GenBank HQ945416.1), shown in the
API's interactive docs."""


class IdentifyRequest(BaseModel):
    """One sequence to identify."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"examples": [{"sequence": EXAMPLE}]}
    )

    sequence: str = Field(min_length=1, max_length=MAX_SEQUENCE_CHARACTERS)


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


def _log_line(result: Result, cached: bool) -> dict[str, Any]:
    """What is logged for monitoring: no sequences, only outcomes."""
    steps = {step.name: step for step in result.trace}
    router = steps.get("router")
    top = router.details["top"][0] if router else {}
    specialist = next(
        (s for n, s in steps.items() if n.startswith("specialist:")), None
    )
    return {
        "event": "identify",
        "status": result.status,
        "reason": result.reason,
        "length": result.length,
        "strand": result.strand,
        "long_input": result.region is not None,
        "level": result.answer.level if result.answer else None,
        "confident": result.answer.confident if result.answer else None,
        "router_label": top.get("label"),
        "router_probability": top.get("probability"),
        "specialist_outcome": specialist.details["outcome"] if specialist else None,
        "identity": specialist.details["identity"] if specialist else None,
        "seconds": result.seconds,
        "cached": cached,
    }


def create_app(settings: Settings, loaded: Loaded | None = None) -> FastAPI:
    """Make the API. The bundle is loaded at start-up unless ``loaded`` is
    given (for tests)."""
    state: dict[str, Loaded] = {}
    cache = _Cache(settings.cache_size)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["loaded"] = loaded or load_bundle(settings.bundle)
        logger.info(
            json.dumps({"event": "started", "versions": state["loaded"].versions})
        )
        yield

    app = FastAPI(
        title="dnabarcode API",
        version=__version__,
        description=(
            "Identify a COI DNA barcode. Answers describe our reference data "
            "and this marker, with a decision trace."
        ),
        lifespan=lifespan,
    )

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
        found = state["loaded"]
        return Version(
            app=found.identifier.config.name,
            prokopton_ml=__version__,
            versions=found.versions,
        )

    @app.post("/identify", dependencies=[Depends(check_key)])
    def identify(body: IdentifyRequest) -> Result:
        key = hashlib.sha256(body.sequence.encode("utf-8")).hexdigest()
        result = cache.get(key)
        cached = result is not None
        if result is None:
            result = state["loaded"].identifier.identify(body.sequence)
            cache.put(key, result)
        logger.info(json.dumps(_log_line(result, cached)))
        return result

    return app


def create_app_from_env() -> FastAPI:
    """The app for ``uvicorn --factory``, with settings from the environment."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return create_app(Settings.from_env())
