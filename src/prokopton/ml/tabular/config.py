"""Settings of a tabular classifier (``[models.<name>]`` with ``kind =
"core.tabular-classifier"``) and of its serving steps.

They are declared to the platform through the ``prokopton.ml.settings`` entry
point (``tabular_settings``), so an app's config is checked against them.
"""

from pathlib import Path
from typing import Any, Literal

from prokopton.core.config import ConfigModel
from pydantic import Field, model_validator

from prokopton.ml.config import AppConfig, UnknownClassifierError
from prokopton.ml.settings import Settings

__all__ = [
    "KIND",
    "ClassifierStepConfig",
    "RecordInputStepConfig",
    "TableConfig",
    "TableSplitConfig",
    "TableTrainingConfig",
    "TabularClassifierConfig",
    "table_classifier",
    "tabular_settings",
]

KIND = "core.tabular-classifier"


class TableConfig(ConfigModel):
    """The table a classifier learns from: a CSV file with a header row."""

    path: Path
    target: str = Field(min_length=1)
    """The column to predict."""
    numeric: tuple[str, ...] = ()
    """Columns of numbers; missing values are filled with the column's median."""
    categorical: tuple[str, ...] = ()
    """Columns of categories; missing values are filled with the most common."""
    missing: tuple[str, ...] = ("", "NA")
    """How the file writes a missing value."""

    @model_validator(mode="after")
    def _check_columns(self) -> "TableConfig":
        columns = (*self.numeric, *self.categorical)
        if not columns:
            msg = "A table needs at least one numeric or categorical column"
            raise ValueError(msg)
        if len(set(columns)) != len(columns) or self.target in columns:
            msg = "Each column may be used once, and the target not as an input"
            raise ValueError(msg)
        return self

    @property
    def columns(self) -> tuple[str, ...]:
        """The input columns, numeric first."""
        return (*self.numeric, *self.categorical)


class TableSplitConfig(ConfigModel):
    """How records are divided: stratified by the target, with a seed."""

    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    seed: int = 42

    @model_validator(mode="after")
    def _check_fractions(self) -> "TableSplitConfig":
        if self.validation_fraction + self.test_fraction >= 1.0:
            msg = "validation_fraction + test_fraction must be less than 1"
            raise ValueError(msg)
        return self


class TableTrainingConfig(ConfigModel):
    """Logistic regression, with its regularisation strength chosen on the
    validation split; the best is registered in MLflow."""

    c_values: tuple[float, ...] = Field(default=(0.1, 1.0, 10.0), min_length=1)
    max_iter: int = Field(default=1000, ge=1)
    experiment: str | None = None
    """The MLflow experiment (default ``<app>-<model>``)."""
    registered_model: str | None = None
    """The registered model's name (default ``<app>-<model>``)."""
    alias: str = Field(default="candidate", min_length=1)


class TabularClassifierConfig(ConfigModel):
    """A classifier of a table's records."""

    kind: Literal["core.tabular-classifier"] = "core.tabular-classifier"
    table: TableConfig
    split: TableSplitConfig = Field(default_factory=TableSplitConfig)
    training: TableTrainingConfig = Field(default_factory=TableTrainingConfig)
    card: Path | None = None
    """Where the model card is written (default ``<models_dir>/<model>/card.md``)."""


class RecordInputStepConfig(ConfigModel):
    """Accepts one record: a JSON object with the model's input columns, any
    of them missing (``null`` or left out)."""

    kind: Literal["core.record-input"]
    model: str
    example: dict[str, Any] = Field(default_factory=dict)
    """An example record, shown in the API docs."""


class ClassifierStepConfig(ConfigModel):
    """Predicts the record's class: "Most likely …" when the probability is
    at least ``threshold``, otherwise "Possibly …"."""

    kind: Literal["core.classifier"]
    model: str
    threshold: float = Field(default=0.9, gt=0.0, le=1.0)


def tabular_settings() -> list[Settings]:
    """The tabular settings (the ``prokopton.ml.settings`` entry point)."""
    return [
        Settings("model", KIND, TabularClassifierConfig),
        Settings("step", "core.record-input", RecordInputStepConfig, KIND),
        Settings("step", "core.classifier", ClassifierStepConfig, KIND),
    ]


def table_classifier(config: AppConfig, name: str) -> TabularClassifierConfig:
    """A tabular classifier's settings, by name.

    Raises:
        UnknownClassifierError: If the app has no tabular classifier of that
            name.
    """
    found = config.model_settings(name) if name in config.models else None
    if not isinstance(found, TabularClassifierConfig):
        msg = f"This app has no {name!r} tabular classifier configured"
        raise UnknownClassifierError(msg)
    return found
