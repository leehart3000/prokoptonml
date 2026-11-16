"""Pipeline stages: the named steps of an app's pipeline, run by their kind.

An app's config lists its stages (``[pipeline]`` and ``[stages.<name>]``).
Each stage has a *kind*, such as ``bio.ingest`` or ``core.bundle``, which says
what it does. Kinds are provided by the platform (``core_kinds`` here, and
``prokopton.ml.bio.stages`` for sequence data) and by any installed package
that declares an entry point in the ``prokopton.ml.stages`` group: a function
that returns a list of ``StageKind``.

Each run writes a small record (when it finished, with which platform version
and settings), so ``prokopton-ml stages`` can show when a stage last ran and
whether its settings have changed since. Nothing is skipped automatically.
"""

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import entry_points
from pathlib import Path

from pydantic import BaseModel, ValidationError

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.bundle import BundleError, build_bundle, bundle_path

__all__ = [
    "ENTRY_POINT_GROUP",
    "Registry",
    "StageContext",
    "StageError",
    "StageKind",
    "StageRecord",
    "core_kinds",
    "entry_point_kinds",
    "plan",
    "read_record",
    "record_path",
    "run_stage",
    "settings_hash",
]

ENTRY_POINT_GROUP = "prokopton.ml.stages"


class StageError(ValueError):
    """Raised for a stage that is not configured, or not configured correctly."""


@dataclass(frozen=True)
class StageContext:
    """What a stage is given when it runs."""

    config: AppConfig
    name: str
    """The stage's name in the config, such as ``router-train``."""
    model: str | None
    """The model it acts on, if any."""


@dataclass(frozen=True)
class StageKind:
    """A kind of stage: what it does, and what it needs."""

    kind: str
    description: str
    run: Callable[[StageContext], str]
    """Runs the stage and returns a one-line summary."""
    model_kind: str | None = None
    """The kind of model it acts on (such as ``bio.kmer-classifier``), if any."""
    uses: tuple[str, ...] = ("data",)
    """The config sections its result depends on (besides its model)."""
    errors: tuple[type[Exception], ...] = field(default=())
    """Expected errors, reported without a traceback."""


class StageRecord(BaseModel):
    """What a stage last ran with."""

    stage: str
    kind: str
    model: str | None
    finished_at: datetime
    platform_version: str
    settings_sha256: str
    summary: str


class Registry:
    """The stage kinds available, by kind."""

    def __init__(self, kinds: Iterable[StageKind]) -> None:
        self._kinds: dict[str, StageKind] = {}
        for kind in kinds:
            if kind.kind in self._kinds:
                msg = f"Stage kind {kind.kind!r} is provided twice"
                raise StageError(msg)
            self._kinds[kind.kind] = kind

    def __getitem__(self, kind: str) -> StageKind:
        if kind not in self._kinds:
            known = ", ".join(sorted(self._kinds))
            msg = f"Unknown stage kind {kind!r} (known kinds: {known})"
            raise StageError(msg)
        return self._kinds[kind]

    @property
    def errors(self) -> tuple[type[Exception], ...]:
        """Every kind's expected errors."""
        return tuple({e for kind in self._kinds.values() for e in kind.errors})

    def check(self, config: AppConfig) -> None:
        """Check that every configured stage has a known kind, and acts on a
        model of the right kind (or on none, if its kind takes none)."""
        for name, stage in config.stages.items():
            kind = self[stage.kind]
            if kind.model_kind is None and stage.model is not None:
                msg = f"Stage {name!r} ({stage.kind}) does not act on a model"
                raise StageError(msg)
            if kind.model_kind is not None:
                if stage.model is None:
                    msg = f"Stage {name!r} ({stage.kind}) needs a model"
                    raise StageError(msg)
                found = config.models[stage.model].kind
                if found != kind.model_kind:
                    msg = (
                        f"Stage {name!r} ({stage.kind}) acts on a {kind.model_kind} "
                        f"model, but {stage.model!r} is a {found}"
                    )
                    raise StageError(msg)


def entry_point_kinds() -> list[StageKind]:
    """The stage kinds provided by installed packages' entry points."""
    kinds: list[StageKind] = []
    for point in entry_points(group=ENTRY_POINT_GROUP):
        kinds.extend(point.load()())
    return kinds


def _bundle(context: StageContext) -> str:
    manifest = build_bundle(context.config)
    return (
        f"serving bundle {manifest.versions} written to {bundle_path(context.config)}"
    )


def core_kinds() -> list[StageKind]:
    """The generic stage kinds."""
    return [
        StageKind(
            kind="core.bundle",
            description=(
                "Build the serving bundle: the registered models and each serving "
                "specialist's reference set, with checksums."
            ),
            run=_bundle,
            uses=("inference", "models"),
            errors=(BundleError,),
        )
    ]


def plan(
    config: AppConfig,
    names: Iterable[str] = (),
    *,
    run_all: bool = False,
    start: str | None = None,
) -> list[str]:
    """The stages to run: the ones named, every stage, or every stage from
    ``start`` on (in pipeline order).

    Raises:
        StageError: If a stage is not configured, or the choice is unclear.
    """
    names = list(names)
    if sum((bool(names), run_all, start is not None)) != 1:
        msg = "Name the stages to run, or use --all or --from (only one of them)"
        raise StageError(msg)
    order = list(config.pipeline.stages)
    if start is not None:
        if start not in order:
            msg = f"{start!r} is not in the pipeline (prokopton-ml stages lists it)"
            raise StageError(msg)
        return order[order.index(start) :]
    if run_all:
        return order
    for name in names:
        if name not in config.stages:
            msg = f"This app has no stage named {name!r} (prokopton-ml stages)"
            raise StageError(msg)
    return names


def settings_hash(config: AppConfig, name: str, kind: StageKind) -> str:
    """A hash of the settings a stage's result depends on."""
    stage = config.stages[name]
    payload = {
        "stage": stage.model_dump(mode="json"),
        "model": (
            config.models[stage.model].model_dump(mode="json") if stage.model else None
        ),
        "sections": config.model_dump(mode="json", include=set(kind.uses)),
    }
    text = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def record_path(config: AppConfig, name: str) -> Path:
    """Where a stage's last run is recorded."""
    return config.data.paths.models_dir / config.name / "stages" / f"{name}.json"


def read_record(config: AppConfig, name: str) -> StageRecord | None:
    """A stage's last run, if it has run (and its record is readable)."""
    path = record_path(config, name)
    try:
        return StageRecord.model_validate_json(path.read_text("utf-8"))
    except (OSError, ValidationError):
        return None


def run_stage(registry: Registry, config: AppConfig, name: str) -> str:
    """Run one configured stage, record the run, and return its summary."""
    stage = config.stages[name]
    kind = registry[stage.kind]
    summary = kind.run(StageContext(config=config, name=name, model=stage.model))
    record = StageRecord(
        stage=name,
        kind=stage.kind,
        model=stage.model,
        finished_at=datetime.now(UTC),
        platform_version=__version__,
        settings_sha256=settings_hash(config, name, kind),
        summary=summary,
    )
    path = record_path(config, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", "utf-8")
    return summary
