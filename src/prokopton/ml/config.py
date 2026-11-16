"""Configuration models for ProkoptonML applications.

Each application (for example dnabarcode) is described by one TOML file that is
validated against ``AppConfig``. Relative paths in the file are resolved from
the current working directory, normally the repository root.

Schema 4 (``schema = 4``) holds only what every app has: its name and title,
its data release (``[data]``), where its models go (``[paths]``), its models
(``[models.<name>]``, each with a ``kind``), its pipeline (``[pipeline]`` and
``[stages.<name>]``), its serving steps (``[serving]``) and MLflow. Everything
else belongs to the package that reads it (see ``prokopton.ml.settings``): a
model's or serving step's own settings, checked against those declared for
its kind, and app-wide sections such as ``[bio]``, checked against the
section of that name.

Older files are converted when they are read: schema 1 (fixed ``[router]``,
``[marker]`` and ``[specialists.<name>]``), schema 2 (``[inference]``) and
schema 3 (sequence settings in ``[data.source]``, ``[validation]`` and so on);
``prokopton-ml migrate-config`` rewrites them.
"""

from pathlib import Path
from typing import Any, Literal

from prokopton.core.config import ConfigModel
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    model_validator,
)

from prokopton.ml.settings import Declared, declared

__all__ = [
    "AppConfig",
    "DataConfig",
    "MlflowConfig",
    "ModelEntry",
    "PathsConfig",
    "PipelineConfig",
    "ServingConfig",
    "StageConfig",
    "StepEntry",
    "UnknownClassifierError",
    "convert_schema_1",
    "convert_schema_2",
    "convert_schema_3",
    "default_stages",
    "upgrade",
]


class DataConfig(ConfigModel):
    """The app's data."""

    release: str = Field(min_length=1)
    """Which release of its data the app is built from (such as ``GB273``),
    shown in model cards and by ``GET /version``."""


class PathsConfig(ConfigModel):
    """Where the app's models, model cards and bundle are written."""

    models_dir: Path


class MlflowConfig(ConfigModel):
    """Where MLflow keeps experiment runs and model files.

    The default keeps everything in the git-ignored ``data/`` folder: run
    metadata and the model registry in a SQLite database, and model files in
    an artifact folder.
    """

    tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    artifact_dir: Path = Path("data/mlflow/artifacts")


class UnknownClassifierError(ValueError):
    """Raised when an app has no model (or settings) of the name asked for."""


class ModelEntry(ConfigModel):
    """One of the app's models: its ``kind``, and the kind's own settings."""

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: str = Field(min_length=1)


class StepEntry(ConfigModel):
    """One serving step: its ``kind``, the model it uses (if any), and the
    kind's own settings."""

    model_config = ConfigDict(extra="allow", frozen=True)

    kind: str = Field(min_length=1)
    model: str | None = None


class PipelineConfig(ConfigModel):
    """The app's pipeline: its stages, in the order they run."""

    stages: tuple[str, ...] = ()


class StageConfig(ConfigModel):
    """One stage of the pipeline: what it does (``kind``) and, for stages
    that act on a model, which one (``model``)."""

    kind: str = Field(min_length=1)
    model: str | None = None


class ServingConfig(ConfigModel):
    """How the app answers a request: a chain of serving steps, in order
    (``chain``), each configured under ``[serving.steps.<name>]``, served at
    ``POST /<endpoint>``."""

    endpoint: str = Field(default="predict", pattern=r"^[a-z][a-z0-9-]*$")
    description: str = "Make a prediction, with a decision trace."
    chain: tuple[str, ...] = ()
    steps: dict[str, StepEntry] = Field(default_factory=dict)


_CLASSIFIER_STAGES = ("sample", "train", "calibrate", "evaluate", "card")
_SPECIALIST_STAGES = ("reference", "evaluate", "card")


def default_stages(models: dict[str, Any]) -> dict[str, dict[str, str]]:
    """The stages a schema 1 app ran, in order, as schema 2 ``[stages]``.

    ``models`` maps each model's name to its settings (with ``kind``).
    """
    stages: dict[str, dict[str, str]] = {
        name: {"kind": f"bio.{name}"}
        for name in ("ingest", "validate", "split", "leakage")
    }
    for name, settings in models.items():
        if settings.get("kind") == "bio.kmer-classifier":
            for step in _CLASSIFIER_STAGES:
                kind = f"bio.classifier-{step}"
                stages[f"{name}-{step}"] = {"kind": kind, "model": name}
    for name, settings in models.items():
        if settings.get("kind") == "bio.specialist":
            steps = (
                ("anchors", *_SPECIALIST_STAGES)
                if settings.get("anchors")
                else (_SPECIALIST_STAGES)
            )
            for step in steps:
                kind = f"bio.specialist-{step}"
                stages[f"{name}-{step}"] = {"kind": kind, "model": name}
    stages["bundle"] = {"kind": "core.bundle"}
    return stages


def convert_schema_1(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a schema 1 config (as read from TOML or JSON) to schema 2.

    ``[router]`` and ``[marker]`` become ``[models.router]`` and
    ``[models.marker]``, each ``[specialists.<name>]`` becomes
    ``[models.<name>]``, and the pipeline is the stages a schema 1 app ran.
    """
    converted = {
        key: value
        for key, value in data.items()
        if key not in ("router", "marker", "specialists", "schema", "schema_version")
    }
    models: dict[str, Any] = {
        "router": {"kind": "bio.kmer-classifier", **data.get("router", {})}
    }
    if data.get("marker") is not None:
        models["marker"] = {"kind": "bio.kmer-classifier", **data["marker"]}
    for name, settings in (data.get("specialists") or {}).items():
        if not isinstance(settings, dict):
            msg = f"[specialists.{name}] must be a table of settings"
            raise ValueError(msg)
        models[name] = {"kind": "bio.specialist", **settings}
    stages = default_stages(models)
    converted["schema"] = 2
    converted["models"] = models
    converted["pipeline"] = {"stages": list(stages)}
    converted["stages"] = stages
    return converted


_SPECIALIST_KIND = "bio.specialist"


_SCHEMA_2_DESCRIPTION = (
    "Identify a DNA barcode. Answers describe the reference data and marker "
    "behind them, with a decision trace."
)
_INPUT_KEYS = ("max_length", "max_ambiguous_fraction")
_MARKER_KEYS = (
    "barcode_gene",
    "long_input",
    "window",
    "step",
    "region_probability",
    "unsupported_note",
)
_SPECIALIST_KEYS = ("route_threshold",)


def convert_schema_2(data: dict[str, Any]) -> dict[str, Any]:
    """Convert a schema 2 config to schema 3.

    ``[inference]`` becomes the serving steps a schema 2 app ran: ``input``
    (with the router's validated range), ``marker`` (if the app has one),
    ``router``, then each specialist with a ``router_label``.
    """
    # A dumped config says schema_version, a written one schema: keep neither
    converted = {
        key: value
        for key, value in data.items()
        if key not in ("inference", "schema", "schema_version")
    }
    inference = data.get("inference") or {}

    def take(keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: inference[key] for key in keys if key in inference}

    models = data.get("models") or {}
    steps: dict[str, dict[str, Any]] = {}
    if "router" in models:
        steps["input"] = {
            "kind": "bio.sequence-input",
            "model": "router",
            **take(_INPUT_KEYS),
        }
        if "marker" in models:
            steps["marker"] = {
                "kind": "bio.marker",
                "model": "marker",
                **take(_MARKER_KEYS),
            }
        steps["router"] = {"kind": "bio.router", "model": "router"}
    for name, settings in models.items():
        if settings.get("kind") == _SPECIALIST_KIND and settings.get("router_label"):
            steps[name] = {
                "kind": "bio.specialist",
                "model": name,
                **take(_SPECIALIST_KEYS),
            }
    converted["schema"] = 3
    converted["serving"] = {
        # What schema 1 and 2 apps served
        "endpoint": "identify",
        "description": _SCHEMA_2_DESCRIPTION,
        "chain": list(steps),
        "steps": steps,
    }
    return converted


def _pop_path(data: dict[str, Any], path: str) -> Any:
    """Remove and return the table at a dotted path (``None`` if absent)."""
    *parents, last = path.split(".")
    for key in parents:
        child = data.get(key)
        if not isinstance(child, dict):
            return None
        data = child
    return data.pop(last, None)


def _set_path(data: dict[str, Any], path: str, value: Any) -> None:
    """Set the table at a dotted path, making the tables above it."""
    *parents, last = path.split(".")
    for key in parents:
        data = data.setdefault(key, {})
    data[last] = value


def convert_schema_3(
    data: dict[str, Any], found: Declared | None = None
) -> dict[str, Any]:
    """Convert a schema 3 config to schema 4.

    The data release (``[data.source] release``) becomes ``[data] release``,
    and ``models_dir`` moves from ``[data.paths]`` to ``[paths]``. Each
    section's other schema 3 tables move into it, as its settings declare
    (for sequence apps, ``[data.source]`` becomes ``[bio.source]``,
    ``[validation]`` becomes ``[bio.validation]``, and so on).
    """
    found = found or declared()
    converted = {
        key: value
        for key, value in data.items()
        if key not in ("schema", "schema_version")
    }
    old = dict(converted.pop("data", None) or {})
    source = dict(old.pop("source", None) or {})
    folders = dict(old.pop("paths", None) or {})
    release = source.pop("release", None)
    models_dir = folders.pop("models_dir", None)
    converted["data"] = {**old, **({"release": release} if release else {})}
    if models_dir is not None:
        converted["paths"] = {"models_dir": models_dir}
    rest: dict[str, Any] = {"data": {}}
    if source:
        rest["data"]["source"] = source
    if folders:
        rest["data"]["paths"] = folders
    for section in found.sections.values():
        for table, target in section.schema_3_tables.items():
            moving = _pop_path(rest, table)
            if moving is None:
                moving = _pop_path(converted, table)
            if moving is not None:
                _set_path(converted, target, moving)
    for key, value in (rest.get("data") or {}).items():
        converted["data"][key] = value  # Not claimed: reported as unknown
    converted["schema"] = 4
    return converted


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """Convert config data of any older schema to the current one (4)."""
    schema = data.get("schema", data.get("schema_version", 1))
    if schema == 1:
        data = convert_schema_1(data)
        schema = 2
    if schema == 2:
        data = convert_schema_2(data)
        schema = 3
    if schema == 3:
        data = convert_schema_3(data)
    return data


def _settings_error(where: str, error: ValidationError) -> ValueError:
    problems = "; ".join(
        f"{'.'.join(str(part) for part in problem['loc']) or '(table)'}: "
        f"{problem['msg']}"
        for problem in error.errors()
    )
    return ValueError(f"{where}: {problems}")


class AppConfig(ConfigModel):
    """Top-level configuration for a ProkoptonML application.

    Top-level tables other than the ones below are sections that installed
    packages declare (such as ``[bio]``); each is checked against its
    declared settings.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    schema_version: Literal[4] = Field(
        default=4,
        validation_alias=AliasChoices("schema", "schema_version"),
        serialization_alias="schema",
    )
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    data: DataConfig
    paths: PathsConfig
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

    _models: dict[str, BaseModel] = PrivateAttr(default_factory=dict)
    _steps: dict[str, BaseModel] = PrivateAttr(default_factory=dict)
    _sections: dict[str, BaseModel] = PrivateAttr(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _read_older_schemas(cls, data: Any) -> Any:
        return upgrade(data) if isinstance(data, dict) else data

    @model_validator(mode="after")
    def _check_pipeline(self) -> "AppConfig":
        for name in self.pipeline.stages:
            if name not in self.stages:
                msg = f"pipeline stage {name!r} has no [stages.{name}] section"
                raise ValueError(msg)
        for name, stage in self.stages.items():
            if stage.model is not None and stage.model not in self.models:
                msg = f"stage {name!r} acts on {stage.model!r}, which is not a model"
                raise ValueError(msg)
        for name in self.serving.chain:
            if name not in self.serving.steps:
                msg = f"serving step {name!r} has no [serving.steps.{name}] section"
                raise ValueError(msg)
        for name, step in self.serving.steps.items():
            if step.model is not None and step.model not in self.models:
                msg = f"serving step {name!r} uses {step.model!r}, which is not a model"
                raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_settings(self) -> "AppConfig":
        found = declared()
        for name, entry in self.models.items():
            self._models[name] = self._checked(
                f"[models.{name}]", entry, found.models.get(entry.kind)
            )
        for name, step in self.serving.steps.items():
            settings = found.steps.get(step.kind)
            self._steps[name] = self._checked(f"[serving.steps.{name}]", step, settings)
            needed = settings.needs_model if settings else None
            if needed and step.model and self.models[step.model].kind != needed:
                msg = (
                    f"serving step {name!r} ({step.kind}) needs a {needed} model, "
                    f"but {step.model!r} is a {self.models[step.model].kind}"
                )
                raise ValueError(msg)
        for key, value in (self.__pydantic_extra__ or {}).items():
            section = found.sections.get(key)
            if section is None:
                msg = f"unknown section [{key}]: no installed package declares it"
                raise ValueError(msg)
            try:
                self._sections[key] = section.model.model_validate(value)
            except ValidationError as error:
                raise _settings_error(f"[{key}]", error) from None
        return self

    @staticmethod
    def _checked(where: str, entry: BaseModel, settings: Any) -> BaseModel:
        if settings is None:
            if entry.__pydantic_extra__:
                kind = getattr(entry, "kind", "")
                msg = (
                    f"{where}: no settings are known for kind {kind!r}; is the "
                    "package that provides it installed?"
                )
                raise ValueError(msg)
            return entry
        try:
            checked: BaseModel = settings.model.model_validate(entry.model_dump())
        except ValidationError as error:
            raise _settings_error(where, error) from None
        return checked

    def model_settings(self, name: str) -> BaseModel:
        """A model's own settings, as checked against its kind's.

        Raises:
            UnknownClassifierError: If the app has no model of that name.
        """
        if name not in self._models:
            msg = f"This app has no {name!r} model configured"
            raise UnknownClassifierError(msg)
        return self._models[name]

    def step_settings(self, name: str) -> BaseModel:
        """A serving step's own settings, as checked against its kind's.

        Raises:
            KeyError: If the app has no serving step of that name.
        """
        return self._steps[name]

    def section(self, name: str) -> BaseModel | None:
        """An app-wide section's settings (such as ``bio``), if it has one."""
        return self._sections.get(name)
