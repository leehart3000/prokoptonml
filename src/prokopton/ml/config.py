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

Schema 3 files (sequence settings in ``[data.source]``, ``[validation]`` and
so on) are converted when they are read, and ``prokopton-ml migrate-config``
rewrites them. Schemas 1 and 2 are no longer read: prokopton-ml 0.2.0a7 was
the last version to rewrite them.
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
    "CURRENT_SCHEMA",
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
    "convert_schema_3",
    "old_schema_message",
    "schema_of",
    "upgrade",
]

CURRENT_SCHEMA = 4
"""The config schema written; schema 3 is also read, by converting it."""


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


def schema_of(data: dict[str, Any]) -> int:
    """The schema of config data (as read from TOML or JSON): 1 if it doesn't
    say, as schema 1 files didn't."""
    found = data.get("schema", data.get("schema_version", 1))
    return found if isinstance(found, int) else 1


def old_schema_message(schema: int) -> str:
    """Why a schema 1 or 2 config is refused, and what to do."""
    return (
        f"This config is schema {schema}, which is no longer read: rewrite it as "
        f"schema {CURRENT_SCHEMA} with prokopton-ml 0.2.0a7 (prokopton-ml "
        "migrate-config --write), the last version to read it"
    )


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """Convert schema 3 config data to the current schema (4).

    Raises:
        ValueError: If the data is schema 1 or 2, which are no longer read.
    """
    schema = schema_of(data)
    if schema < 3:
        raise ValueError(old_schema_message(schema))
    return convert_schema_3(data) if schema == 3 else data


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
