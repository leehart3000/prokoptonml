"""Schema 4 configs for sequence apps, for the tests.

``sequence_app`` gives a small app's settings, with the ``[bio]`` section
every sequence stage reads and one model, the router. Tests add models and
change settings in the dict, then ``app_config`` adds the stages and serving
steps a sequence app has (unless the dict gives its own) and checks it.

``write_config`` writes a completed app as a TOML file, for the command
line. ``dnabarcode`` reads the DNAbarcode fixture, and ``minimal`` keeps
only its core settings and ``[bio]`` section.
"""

import json
import re
import tomllib
from pathlib import Path
from typing import Any

from prokopton.ml.config import AppConfig

__all__ = [
    "CLASSIFIER",
    "SPECIALIST",
    "app_config",
    "complete",
    "dnabarcode",
    "minimal",
    "mlflow",
    "sequence_app",
    "source",
    "write_config",
]

CLASSIFIER = "bio.kmer-classifier"
SPECIALIST = "bio.specialist"
FOLDERS = ("raw", "interim", "processed", "splits", "leakage", "samples")
FIXTURES = Path(__file__).resolve().parent / "fixtures"
DESCRIPTION = (
    "Identify a DNA barcode. Answers describe the reference data and marker "
    "behind them, with a decision trace."
)


def dnabarcode() -> dict[str, Any]:
    """DNAbarcode's config (a frozen copy), as read from TOML."""
    with (FIXTURES / "dnabarcode.toml").open("rb") as handle:
        return tomllib.load(handle)


def minimal(**extra: Any) -> dict[str, Any]:
    """DNAbarcode's name, title, data release, paths and ``[bio]`` section,
    with no models, stages or serving steps unless given."""
    data = dnabarcode()
    kept = ("schema", "name", "title", "data", "paths", "bio")
    return {**{key: data[key] for key in kept}, **extra}


def source(**fields: Any) -> dict[str, Any]:
    """A MIDORI2 source that is never downloaded (``[bio.source]``)."""
    return {
        "doi": "10.0000/example",
        "licence": "CC-BY-4.0",
        "archive_url": "https://example.org/fake.zip",
        "checksums_url": "https://example.org/sums.txt",
        "archive_name": "fake.zip",
        "member_template": "x/{gene}.fasta.gz",
        "genes": ["CO1"],
        **fields,
    }


def mlflow(tmp_path: Path) -> dict[str, str]:
    """MLflow settings (``[mlflow]``) that keep everything under ``tmp_path``."""
    return {
        "tracking_uri": f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}",
        "artifact_dir": str(tmp_path / "mlflow" / "artifacts"),
    }


def sequence_app(
    tmp_path: Path,
    *,
    title: str = "Test app",
    release: str = "GB999",
    genes: list[str] | None = None,
    models: dict[str, Any] | None = None,
    **sections: Any,
) -> dict[str, Any]:
    """A sequence app's settings, with its folders under ``tmp_path``.

    ``models`` maps each model's name to its settings (with ``kind``); the
    default is a router with default settings. Other keyword arguments are
    added as top-level tables (such as ``mlflow``).
    """
    return {
        "schema": 4,
        "name": "testapp",
        "title": title,
        "data": {"release": release},
        "paths": {"models_dir": str(tmp_path / "models")},
        "bio": {
            "source": source(**({"genes": genes} if genes else {})),
            "paths": {f"{name}_dir": str(tmp_path / name) for name in FOLDERS},
            "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
        },
        "models": models if models is not None else {"router": {"kind": CLASSIFIER}},
        **sections,
    }


_CLASSIFIER_STAGES = ("sample", "train", "calibrate", "evaluate", "card")
_SPECIALIST_STAGES = ("reference", "evaluate", "card")


def _stages(models: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The stages of a sequence app with these models, in order."""
    stages = {
        name: {"kind": f"bio.{name}"}
        for name in ("ingest", "validate", "split", "leakage")
    }
    for name, settings in models.items():
        if settings.get("kind") == CLASSIFIER:
            for step in _CLASSIFIER_STAGES:
                kind = f"bio.classifier-{step}"
                stages[f"{name}-{step}"] = {"kind": kind, "model": name}
    for name, settings in models.items():
        if settings.get("kind") == SPECIALIST:
            anchors = ("anchors",) if settings.get("anchors") else ()
            for step in (*anchors, *_SPECIALIST_STAGES):
                kind = f"bio.specialist-{step}"
                stages[f"{name}-{step}"] = {"kind": kind, "model": name}
    stages["bundle"] = {"kind": "core.bundle"}
    return stages


def _serving(models: dict[str, Any]) -> dict[str, Any]:
    """The serving steps of a sequence app with these models: the input, the
    marker (if any), the router, then each specialist with a router label."""
    steps: dict[str, dict[str, Any]] = {}
    if "router" in models:
        steps["input"] = {"kind": "bio.sequence-input", "model": "router"}
        if "marker" in models:
            steps["marker"] = {"kind": "bio.marker", "model": "marker"}
        steps["router"] = {"kind": "bio.router", "model": "router"}
    for name, settings in models.items():
        if settings.get("kind") == SPECIALIST and settings.get("router_label"):
            steps[name] = {"kind": "bio.specialist", "model": name}
    return {
        "endpoint": "identify",
        "description": DESCRIPTION,
        "chain": list(steps),
        "steps": steps,
    }


def complete(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` with the stages and serving steps of its models, where it
    gives none of its own."""
    models = data.get("models") or {}
    completed = dict(data)
    if "stages" not in data:
        stages = _stages(models)
        completed["stages"] = stages
        completed["pipeline"] = {"stages": list(stages)}
    if "serving" not in data:
        completed["serving"] = _serving(models)
    return completed


def app_config(data: dict[str, Any]) -> AppConfig:
    """Check a sequence app's settings (see ``complete``)."""
    return AppConfig.model_validate(complete(data))


def _toml_key(key: str) -> str:
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str | Path):
        return json.dumps(str(value))  # A JSON string is a TOML string
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = (f"{_toml_key(k)} = {_toml_value(v)}" for k, v in value.items())
        return "{ " + ", ".join(pairs) + " }"
    msg = f"Can't write {value!r} as TOML"
    raise TypeError(msg)


def _toml_lines(data: dict[str, Any], table: str = "") -> list[str]:
    # TOML has no null: a setting that is None is left out (its default)
    values = {
        k: v for k, v in data.items() if v is not None and not isinstance(v, dict)
    }
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    lines = [f"[{table}]"] if table and (values or not tables) else []
    lines += [f"{_toml_key(k)} = {_toml_value(v)}" for k, v in values.items()]
    for key, value in tables.items():
        name = f"{table}.{_toml_key(key)}" if table else _toml_key(key)
        lines += ["", *_toml_lines(value, name)]
    return lines


def write_config(path: Path, data: dict[str, Any]) -> Path:
    """Write an app's settings (see ``complete``) as a TOML file."""
    path.write_text("\n".join(_toml_lines(complete(data))) + "\n", "utf-8")
    return path
