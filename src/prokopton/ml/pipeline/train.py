"""Train stage: choose and register the router from the sampled data.

Reads the router samples written by the sample stage, tries every candidate
in ``[router.training]`` (see ``prokopton.ml.training.router``), and records
each in MLflow. The best is registered under ``registered_model`` with the
alias ``alias``. Every run is tagged with the data's lineage: the release,
the gene and the SHA-256 of each sample file, so a model can be traced back
to exactly the data it was trained on.

Outputs ``<models_dir>/router/_train_report.json``.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, RouterTrainingConfig
from prokopton.ml.data.checksums import sha256_of_file
from prokopton.ml.pipeline.sample import REPORT_NAME as SAMPLE_REPORT_NAME
from prokopton.ml.training.router import RouterSample, train_router
from prokopton.ml.training.tracking import MlflowTracker, Tracker

__all__ = ["REPORT_NAME", "SAMPLE_SPLITS", "TrainReport", "TrainStageError", "train"]

logger = logging.getLogger(__name__)

REPORT_NAME = "_train_report.json"
SAMPLE_SPLITS = ("train", "val_known", "val_novel")


class TrainStageError(RuntimeError):
    """Raised when the training stage cannot run."""


class TrainReport(BaseModel):
    """Provenance and results of a training run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app: str
    release: str
    gene: str
    prokopton_ml_version: str
    rules: RouterTrainingConfig
    started_at: datetime
    finished_at: datetime
    sample_sha256: dict[str, str]
    candidates_tried: int
    best_model: str
    best_k: int
    best_max_ambiguous_fraction: float
    best_alpha: float
    metrics: dict[str, float]
    registered_model: str
    model_version: str
    alias: str
    parent_run_id: str


def _read_sample(path: Path) -> RouterSample:
    table = pq.read_table(path, columns=["sequence", "label", "ambiguous_fraction"])
    return RouterSample(
        sequences=table.column("sequence").to_pylist(),
        labels=table.column("label").to_pylist(),
        ambiguous_fraction=np.asarray(
            table.column("ambiguous_fraction").to_pylist(), dtype=np.float64
        ),
    )


def _mlflow_tracker(config: AppConfig) -> Tracker:
    return MlflowTracker(
        config.mlflow.tracking_uri,
        config.mlflow.artifact_dir,
        config.router.training.experiment,
    )


def train(
    config: AppConfig,
    tracker_factory: Callable[[AppConfig], Tracker] = _mlflow_tracker,
) -> TrainReport:
    """Train the router candidates, register the best and write a report.

    Args:
        config: The app's configuration.
        tracker_factory: Makes the experiment tracker (MLflow by default).

    Raises:
        TrainStageError: If the samples are missing, or an ambiguity threshold
            is above the one the samples were drawn with (so it would have no
            effect).
    """
    started_at = datetime.now(UTC)
    rules = config.router.training
    gene = config.router.gene
    folder = f"midori2_{config.data.source.release.lower()}"
    sample_root = config.data.paths.samples_dir / folder / "router"
    paths = {name: sample_root / f"{name}.parquet" for name in SAMPLE_SPLITS}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"{', '.join(missing)} not found; run the sample stage first"
        raise TrainStageError(msg)

    sampled_at = config.router.sample.max_ambiguous_fraction
    thresholds = {*rules.ambiguity_thresholds, rules.logistic_ambiguity_threshold}
    too_high = sorted(t for t in thresholds if t > sampled_at)
    if too_high:
        msg = (
            f"Ambiguity thresholds {too_high} are above the sample's "
            f"max_ambiguous_fraction ({sampled_at}), so would have no effect"
        )
        raise TrainStageError(msg)

    checksums = {name: sha256_of_file(path) for name, path in paths.items()}
    report_path = sample_root / SAMPLE_REPORT_NAME
    if report_path.exists():
        checksums["sample_report"] = sha256_of_file(report_path)
    tags = {
        "app": config.name,
        "release": config.data.source.release,
        "gene": gene,
        "prokopton_ml_version": __version__,
        **{f"sample_sha256.{name}": value for name, value in checksums.items()},
    }

    samples = {name: _read_sample(path) for name, path in paths.items()}
    for name, loaded in samples.items():
        logger.info("Loaded %s: %d records", name, len(loaded.labels))

    result = train_router(samples, rules, tracker_factory(config), tags)
    logger.info(
        "Best: %s (selection score %.4f); registered %s version %s as @%s",
        result.best.name,
        result.metrics["selection_score"],
        rules.registered_model,
        result.model_version,
        rules.alias,
    )

    report = TrainReport(
        app=config.name,
        release=config.data.source.release,
        gene=gene,
        prokopton_ml_version=__version__,
        rules=rules,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        sample_sha256=checksums,
        candidates_tried=result.candidates_tried,
        best_model=result.best.model,
        best_k=result.best.k,
        best_max_ambiguous_fraction=result.best.max_ambiguous_fraction,
        best_alpha=result.best.alpha,
        metrics=result.metrics,
        registered_model=rules.registered_model,
        model_version=result.model_version,
        alias=rules.alias,
        parent_run_id=result.parent_run_id,
    )
    output = config.data.paths.models_dir / "router" / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)
    return report
