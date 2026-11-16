"""Stages for a tabular classifier: split its table, train it, evaluate it
and write its model card.

- ``core.tabular-split`` reads the table (``[models.<name>.table]``), checks
  its columns, leaves out records without a target, and divides the rest into
  training, validation and test sets, stratified by the target.
- ``core.tabular-train`` trains logistic regression for each regularisation
  strength (``c_values``), keeps the one with the best validation accuracy
  (then macro-F1, then calibration), and registers it in MLflow under the
  training alias.
- ``core.tabular-evaluate`` measures the registered model on the test set.
- ``core.tabular-card`` writes a model card from the three reports.

The splits and reports are kept in ``<models_dir>/<model>/``.
"""

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from prokopton.ml.config import AppConfig
from prokopton.ml.stages import StageContext, StageKind
from prokopton.ml.tabular.config import KIND, TableConfig, table_classifier
from prokopton.ml.training.metrics import classification_metrics
from prokopton.ml.training.tracking import Tracker, mlflow_tracker

__all__ = [
    "SPLITS",
    "TableError",
    "build_pipeline",
    "evaluate",
    "read_split",
    "read_table",
    "split",
    "tabular_stage_kinds",
    "train",
    "write_card",
]

SPLITS = ("train", "validation", "test")

TrackerFactory = Callable[[AppConfig], Tracker]


class TableError(ValueError):
    """Raised when a table, its splits or its reports are missing or wrong."""


class SplitReport(BaseModel):
    model: str
    table: str
    table_sha256: str
    release: str
    seed: int
    left_out: int
    """Records without a target."""
    records: dict[str, int]
    classes: dict[str, dict[str, int]]
    created_at: datetime


class Candidate(BaseModel):
    c: float
    validation: dict[str, float]


class TrainReport(BaseModel):
    model: str
    registered_model: str
    model_version: str
    run_id: str
    classes: list[str]
    candidates: list[Candidate]
    chosen_c: float
    created_at: datetime


class EvaluationReport(BaseModel):
    model: str
    model_version: str
    records: int
    metrics: dict[str, float]
    confusion: dict[str, dict[str, int]]
    """For each true class, how many records were predicted as each class."""
    created_at: datetime


def _folder(config: AppConfig, model: str) -> Path:
    return config.paths.models_dir / model


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _typed(frame: pd.DataFrame, table: TableConfig, where: str) -> pd.DataFrame:
    missing = [c for c in (*table.columns, table.target) if c not in frame.columns]
    if missing:
        msg = f"{where} has no column {', '.join(missing)}"
        raise TableError(msg)
    frame = frame[[*table.columns, table.target]].copy()
    for column in table.numeric:
        try:
            frame[column] = pd.to_numeric(frame[column]).astype(float)
        except (ValueError, TypeError) as error:
            msg = f"{where}: column {column!r} is not numeric ({error})"
            raise TableError(msg) from error
    return frame


def read_table(table: TableConfig) -> pd.DataFrame:
    """The table's input columns and target, with numbers as numbers.

    Raises:
        TableError: If the file or a column is missing, or a numeric column
            holds something else.
    """
    if not table.path.is_file():
        msg = f"{table.path} not found"
        raise TableError(msg)
    frame = pd.read_csv(
        table.path, dtype=str, keep_default_na=False, na_values=list(table.missing)
    )
    return _typed(frame, table, str(table.path))


def read_split(config: AppConfig, model: str, name: str) -> pd.DataFrame:
    """One of a classifier's splits (``train``, ``validation`` or ``test``).

    Raises:
        TableError: If the split stage has not run.
    """
    path = _folder(config, model) / "table" / f"{name}.csv"
    if not path.is_file():
        msg = f"{path} not found; run the split stage for {model!r} first"
        raise TableError(msg)
    table = table_classifier(config, model).table
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    return _typed(frame, table, str(path))


def _counts(labels: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(labels.value_counts().items())}


def split(config: AppConfig, model: str) -> SplitReport:
    """Divide a classifier's table into its splits (see the module docstring).

    Raises:
        TableError: If the table cannot be read, or a class has too few
            records to be split.
    """
    settings = table_classifier(config, model)
    table, rules = settings.table, settings.split
    frame = read_table(table)
    labelled = frame[frame[table.target].notna()]
    try:
        rest, test = train_test_split(
            labelled,
            test_size=rules.test_fraction,
            random_state=rules.seed,
            stratify=labelled[table.target],
        )
        train_part, validation = train_test_split(
            rest,
            test_size=rules.validation_fraction / (1 - rules.test_fraction),
            random_state=rules.seed,
            stratify=rest[table.target],
        )
    except ValueError as error:
        msg = f"The table cannot be split by {table.target!r}: {error}"
        raise TableError(msg) from error
    folder = _folder(config, model) / "table"
    folder.mkdir(parents=True, exist_ok=True)
    parts = {"train": train_part, "validation": validation, "test": test}
    for name, part in parts.items():
        part.to_csv(folder / f"{name}.csv", index=False)
    report = SplitReport(
        model=model,
        table=str(table.path),
        table_sha256=_sha256(table.path),
        release=config.data.release,
        seed=rules.seed,
        left_out=len(frame) - len(labelled),
        records={name: len(part) for name, part in parts.items()},
        classes={name: _counts(part[table.target]) for name, part in parts.items()},
        created_at=datetime.now(UTC),
    )
    (folder / "split.json").write_text(report.model_dump_json(indent=2), "utf-8")
    return report


def build_pipeline(table: TableConfig, c: float, max_iter: int) -> Pipeline:
    """The model: fill missing values, scale numbers, one-hot categories, then
    logistic regression."""
    steps: list[tuple[str, Any, list[str]]] = []
    if table.numeric:
        numbers = Pipeline(
            [("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        steps.append(("numeric", numbers, list(table.numeric)))
    if table.categorical:
        categories = Pipeline(
            [
                ("fill", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        steps.append(("categorical", categories, list(table.categorical)))
    return Pipeline(
        [
            ("columns", ColumnTransformer(steps)),
            ("classify", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def _registered(config: AppConfig, model: str) -> str:
    return table_classifier(config, model).training.registered_model or (
        f"{config.name}-{model}"
    )


def _scores(
    pipeline: Pipeline, frame: pd.DataFrame, table: TableConfig
) -> dict[str, float]:
    classes = [str(c) for c in pipeline.classes_]
    probabilities = pipeline.predict_proba(frame[list(table.columns)])
    labels = [str(label) for label in frame[table.target]]
    return classification_metrics(labels, np.asarray(probabilities), classes)


def _rank(candidate: Candidate) -> tuple[float, float, float]:
    """Better validation accuracy first, then macro-F1, then calibration (a
    smaller expected calibration error); the first of equals."""
    scores = candidate.validation
    return scores["accuracy"], scores["macro_f1"], -scores["ece"]


def train(
    config: AppConfig, model: str, tracker_factory: TrackerFactory | None = None
) -> TrainReport:
    """Train, choose and register a classifier (see the module docstring).

    Raises:
        TableError: If the split stage has not run.
    """
    settings = table_classifier(config, model)
    table, rules = settings.table, settings.training
    training = read_split(config, model, "train")
    validation = read_split(config, model, "validation")
    make_tracker = tracker_factory or partial(mlflow_tracker, model=model)
    tracker = make_tracker(config)
    inputs = training[list(table.columns)]
    candidates: list[tuple[Candidate, Pipeline]] = []
    with tracker.run(f"{model}-train") as run_id:
        tracker.log_params({"columns": ",".join(table.columns), "target": table.target})
        tracker.set_tags({"release": config.data.release})
        for c in rules.c_values:
            with tracker.run(f"{model}-c-{c}", nested=True):
                pipeline = build_pipeline(table, c, rules.max_iter)
                pipeline.fit(inputs, training[table.target].astype(str))
                scores = _scores(pipeline, validation, table)
                tracker.log_params({"c": c})
                tracker.log_metrics({f"validation_{k}": v for k, v in scores.items()})
            candidates.append((Candidate(c=c, validation=scores), pipeline))
        best, chosen = max(candidates, key=lambda pair: _rank(pair[0]))
        tracker.log_metrics({f"chosen_{k}": v for k, v in best.validation.items()})
        version = tracker.log_model(
            chosen, inputs.head(3), _registered(config, model), rules.alias
        )
    report = TrainReport(
        model=model,
        registered_model=_registered(config, model),
        model_version=version,
        run_id=run_id,
        classes=[str(c) for c in chosen.classes_],
        candidates=[candidate for candidate, _ in candidates],
        chosen_c=best.c,
        created_at=datetime.now(UTC),
    )
    path = _folder(config, model) / "train.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    return report


def evaluate(
    config: AppConfig, model: str, tracker_factory: TrackerFactory | None = None
) -> EvaluationReport:
    """Measure the registered classifier on the test set.

    Raises:
        TableError: If the split stage has not run.
    """
    settings = table_classifier(config, model)
    table = settings.table
    test = read_split(config, model, "test")
    make_tracker = tracker_factory or partial(mlflow_tracker, model=model)
    pipeline, version = make_tracker(config).load_model(
        _registered(config, model), settings.training.alias
    )
    predicted = [str(p) for p in pipeline.predict(test[list(table.columns)])]
    truth = [str(t) for t in test[table.target]]
    classes = sorted({*truth, *(str(c) for c in pipeline.classes_)})
    confusion = {t: dict.fromkeys(classes, 0) for t in classes}
    for t, p in zip(truth, predicted, strict=True):
        confusion[t][p] += 1
    report = EvaluationReport(
        model=model,
        model_version=version,
        records=len(test),
        metrics=_scores(pipeline, test, table),
        confusion=confusion,
        created_at=datetime.now(UTC),
    )
    path = _folder(config, model) / "evaluation.json"
    path.write_text(report.model_dump_json(indent=2), "utf-8")
    return report


def _read[R: BaseModel](path: Path, report: type[R], stage: str) -> R:
    if not path.is_file():
        msg = f"{path} not found; run the {stage} stage first"
        raise TableError(msg)
    return report.model_validate(json.loads(path.read_text("utf-8")))


def write_card(config: AppConfig, model: str) -> Path:
    """Write the classifier's model card (Markdown) from its reports.

    Raises:
        TableError: If a report is missing.
    """
    settings = table_classifier(config, model)
    folder = _folder(config, model)
    splits = _read(folder / "table" / "split.json", SplitReport, "split")
    trained = _read(folder / "train.json", TrainReport, "train")
    tested = _read(folder / "evaluation.json", EvaluationReport, "evaluate")
    table = settings.table
    lines = [
        f"# Model card: {model} ({config.title})",
        "",
        f"Predicts `{table.target}` from one record of a table, with a "
        "probability for each class.",
        "",
        "## Model",
        "",
        f"- Logistic regression (C = {trained.chosen_c:g}, chosen on the "
        "validation set from "
        + ", ".join(f"{c.c:g}" for c in trained.candidates)
        + "), after filling missing values, scaling numbers and one-hot "
        "encoding categories.",
        f"- Registered as `{trained.registered_model}`, version "
        f"{trained.model_version}.",
        f"- Classes: {', '.join(trained.classes)}.",
        "",
        "## Data",
        "",
        f"- `{splits.table}` (release {splits.release}, SHA-256 "
        f"`{splits.table_sha256[:12]}…`).",
        f"- Inputs: {', '.join(f'`{c}`' for c in table.columns)}.",
        f"- {splits.left_out} records without a target were left out; the rest "
        f"were split by class (seed {splits.seed}): "
        + ", ".join(f"{n} {splits.records[n]}" for n in SPLITS)
        + ".",
        "",
        "## Results on the test set",
        "",
        f"- {tested.records} records: accuracy {tested.metrics['accuracy']:.3f}, "
        f"macro-F1 {tested.metrics['macro_f1']:.3f}, expected calibration error "
        f"{tested.metrics['ece']:.3f}.",
        "",
        "| True class | " + " | ".join(tested.confusion) + " |",
        "|---|" + "---|" * len(tested.confusion),
        *(
            f"| {t} | " + " | ".join(str(n) for n in row.values()) + " |"
            for t, row in tested.confusion.items()
        ),
        "",
        "## Limits",
        "",
        "- Answers are only as good as the records the model learnt from: "
        "records unlike them (other places, years or kinds) may be wrong with "
        "confidence.",
        "",
    ]
    path = settings.card or folder / "card.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), "utf-8")
    return path


def _split(context: StageContext) -> str:
    assert context.model is not None  # The registry checks this
    report = split(context.config, context.model)
    return ", ".join(f"{n} {report.records[n]}" for n in SPLITS)


def _train(context: StageContext) -> str:
    assert context.model is not None
    report = train(context.config, context.model)
    return f"registered version {report.model_version} (C = {report.chosen_c:g})"


def _evaluate(context: StageContext) -> str:
    assert context.model is not None
    report = evaluate(context.config, context.model)
    return f"test accuracy {report.metrics['accuracy']:.3f} on {report.records}"


def _card(context: StageContext) -> str:
    assert context.model is not None
    return f"model card written to {write_card(context.config, context.model)}"


def tabular_stage_kinds() -> list[StageKind]:
    """The tabular stage kinds (the ``prokopton.ml.stages`` entry point)."""
    uses = ("data", "paths", "mlflow")
    return [
        StageKind(
            "core.tabular-split",
            "Split a table's records into training, validation and test sets.",
            _split,
            model_kind=KIND,
            uses=("data", "paths"),
            errors=(TableError,),
        ),
        StageKind(
            "core.tabular-train",
            "Train a tabular classifier, choose the best and register it.",
            _train,
            model_kind=KIND,
            uses=uses,
            errors=(TableError,),
        ),
        StageKind(
            "core.tabular-evaluate",
            "Measure a registered tabular classifier on the test set.",
            _evaluate,
            model_kind=KIND,
            uses=uses,
            errors=(TableError,),
        ),
        StageKind(
            "core.tabular-card",
            "Write a tabular classifier's model card from its reports.",
            _card,
            model_kind=KIND,
            uses=uses,
            errors=(TableError,),
        ),
    ]
