"""Configuration models for ProkoptonML applications.

Each application (for example dnabarcode) is described by one TOML file that is
validated against ``AppConfig``. Relative paths in the file are resolved from
the current working directory, normally the repository root.

Schema 2 (``schema = 2``) describes an app's models under ``[models.<name>]``,
each with a ``kind``, and its pipeline as named stages (``[pipeline]`` and
``[stages.<name>]``). Schema 1 files, with fixed ``[router]``, ``[marker]``
and ``[specialists.<name>]`` sections, are converted when they are read (see
``convert_schema_1``); ``prokopton-ml migrate-config`` rewrites them.

The settings of sequence apps (the MIDORI2 source, the k-mer classifiers,
the specialists and the sequence serving steps) are defined in
``prokopton.ml.bio.config``; they are named here too, for now.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from prokopton.core.config import ConfigModel
from pydantic import (
    AliasChoices,
    Discriminator,
    Field,
    Tag,
    model_validator,
)

from prokopton.ml.bio.config import (
    ALLOWED_URL_SCHEMES,
    ClassifierCalibrationConfig,
    ClassifierCardConfig,
    ClassifierConfig,
    ClassifierEvaluationConfig,
    ClassifierSampleConfig,
    ClassifierServingConfig,
    ClassifierTrainingConfig,
    ComplexConfig,
    LeakageConfig,
    MarkerStepConfig,
    Midori2SourceConfig,
    RouterStepConfig,
    SequenceInputStepConfig,
    SpecialistConfig,
    SpecialistEvaluationConfig,
    SpecialistStepConfig,
    SplitConfig,
    ValidationConfig,
)

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "AppConfig",
    "ClassifierCalibrationConfig",
    "ClassifierCardConfig",
    "ClassifierConfig",
    "ClassifierEvaluationConfig",
    "ClassifierSampleConfig",
    "ClassifierServingConfig",
    "ClassifierTrainingConfig",
    "ComplexConfig",
    "DataConfig",
    "DataPathsConfig",
    "LeakageConfig",
    "MarkerStepConfig",
    "Midori2SourceConfig",
    "MlflowConfig",
    "ModelConfig",
    "OtherStepConfig",
    "PipelineConfig",
    "RouterStepConfig",
    "SequenceInputStepConfig",
    "ServingConfig",
    "ServingStepConfig",
    "SpecialistConfig",
    "SpecialistEvaluationConfig",
    "SpecialistStepConfig",
    "SplitConfig",
    "StageConfig",
    "UnknownClassifierError",
    "ValidationConfig",
    "convert_schema_1",
    "convert_schema_2",
    "default_stages",
    "upgrade",
]


class DataPathsConfig(ConfigModel):
    """Local folders for downloaded (raw), converted (interim) and validated
    (processed) data."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    splits_dir: Path
    leakage_dir: Path
    samples_dir: Path
    models_dir: Path


class DataConfig(ConfigModel):
    """Data settings for an application."""

    source: Midori2SourceConfig
    paths: DataPathsConfig


class MlflowConfig(ConfigModel):
    """Where MLflow keeps experiment runs and model files.

    The default keeps everything in the git-ignored ``data/`` folder: run
    metadata and the model registry in a SQLite database, and model files in
    an artifact folder.
    """

    tracking_uri: str = "sqlite:///data/mlflow/mlflow.db"
    artifact_dir: Path = Path("data/mlflow/artifacts")


class UnknownClassifierError(ValueError):
    """Raised when an app has no model of the name (and kind) asked for."""


ModelConfig = Annotated[
    ClassifierConfig | SpecialistConfig, Field(discriminator="kind")
]
"""One of an app's models, chosen by its ``kind``."""


class PipelineConfig(ConfigModel):
    """The app's pipeline: its stages, in the order they run."""

    stages: tuple[str, ...] = ()


class StageConfig(ConfigModel):
    """One stage of the pipeline: what it does (``kind``) and, for stages
    that act on a model, which one (``model``)."""

    kind: str = Field(min_length=1)
    model: str | None = None


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


_CLASSIFIER_KIND = "bio.kmer-classifier"
_SPECIALIST_KIND = "bio.specialist"


class OtherStepConfig(ConfigModel):
    """A serving step of a kind provided by another package."""

    kind: str = Field(min_length=1)
    model: str | None = None


_STEP_TYPES: dict[str, type[ConfigModel]] = {
    "bio.sequence-input": SequenceInputStepConfig,
    "bio.marker": MarkerStepConfig,
    "bio.router": RouterStepConfig,
    "bio.specialist": SpecialistStepConfig,
}
_STEP_MODEL_KINDS = {
    "bio.sequence-input": _CLASSIFIER_KIND,
    "bio.marker": _CLASSIFIER_KIND,
    "bio.router": _CLASSIFIER_KIND,
    "bio.specialist": _SPECIALIST_KIND,
}


def _step_tag(value: Any) -> str:
    kind = value.get("kind") if isinstance(value, dict) else getattr(value, "kind", "")
    return kind if kind in _STEP_TYPES else "other"


ServingStepConfig = Annotated[
    Annotated[SequenceInputStepConfig, Tag("bio.sequence-input")]
    | Annotated[MarkerStepConfig, Tag("bio.marker")]
    | Annotated[RouterStepConfig, Tag("bio.router")]
    | Annotated[SpecialistStepConfig, Tag("bio.specialist")]
    | Annotated[OtherStepConfig, Tag("other")],
    Discriminator(_step_tag),
]
"""One serving step's settings, chosen by its ``kind``."""


class ServingConfig(ConfigModel):
    """How the app answers a request: a chain of serving steps, in order
    (``chain``), each configured under ``[serving.steps.<name>]``, served at
    ``POST /<endpoint>``."""

    endpoint: str = Field(default="predict", pattern=r"^[a-z][a-z0-9-]*$")
    description: str = "Make a prediction, with a decision trace."
    chain: tuple[str, ...] = ()
    steps: dict[str, ServingStepConfig] = Field(default_factory=dict)


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


def upgrade(data: dict[str, Any]) -> dict[str, Any]:
    """Convert config data of any older schema to the current one (3)."""
    schema = data.get("schema", data.get("schema_version", 1))
    if schema == 1:
        data = convert_schema_1(data)
        schema = 2
    if schema == 2:
        data = convert_schema_2(data)
    return data


class AppConfig(ConfigModel):
    """Top-level configuration for a ProkoptonML application."""

    schema_version: Literal[3] = Field(
        default=3,
        validation_alias=AliasChoices("schema", "schema_version"),
        serialization_alias="schema",
    )
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    stages: dict[str, StageConfig] = Field(default_factory=dict)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)

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
            if step.model is None:
                continue
            if step.model not in self.models:
                msg = f"serving step {name!r} uses {step.model!r}, which is not a model"
                raise ValueError(msg)
            needed = _STEP_MODEL_KINDS.get(step.kind)
            found = self.models[step.model].kind
            if needed is not None and found != needed:
                msg = (
                    f"serving step {name!r} ({step.kind}) needs a {needed} model, "
                    f"but {step.model!r} is a {found}"
                )
                raise ValueError(msg)
        return self

    def model(self, name: str) -> ClassifierConfig | SpecialistConfig:
        """Return a model's settings by name.

        Raises:
            UnknownClassifierError: If the app has no model of that name.
        """
        if name not in self.models:
            msg = f"This app has no {name!r} model configured"
            raise UnknownClassifierError(msg)
        return self.models[name]

    def classifier(self, name: str) -> ClassifierConfig:
        """Return a classifier's settings by name (such as ``router``).

        Raises:
            UnknownClassifierError: If the app has no classifier of that name.
        """
        found = self.models.get(name)
        if not isinstance(found, ClassifierConfig):
            msg = f"This app has no {name!r} classifier configured"
            raise UnknownClassifierError(msg)
        return found

    def specialist(self, name: str) -> SpecialistConfig:
        """Return a specialist's settings by name.

        Raises:
            UnknownClassifierError: If the app has no specialist of that name.
        """
        found = self.models.get(name)
        if not isinstance(found, SpecialistConfig):
            msg = f"This app has no {name!r} specialist configured"
            raise UnknownClassifierError(msg)
        return found

    @property
    def router(self) -> ClassifierConfig:
        """The router (serving still expects one, named ``router``)."""
        return self.classifier("router")

    @property
    def marker(self) -> ClassifierConfig | None:
        """The marker classifier, if the app has one (named ``marker``)."""
        found = self.models.get("marker")
        return found if isinstance(found, ClassifierConfig) else None

    @property
    def specialists(self) -> dict[str, SpecialistConfig]:
        """The app's specialists, by name."""
        return {
            name: model
            for name, model in self.models.items()
            if isinstance(model, SpecialistConfig)
        }
