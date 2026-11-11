"""Calibrate stage: correct the trained router's confidence.

Loads the model the training alias points at, fits a length-aware
temperature on the validation samples (whole sequences and random
fragments), and registers the calibrated model as a new version with the
same alias. Predictions are unchanged; only confidence changes. If the
loaded model is already calibrated, the model inside is recalibrated, so
running the stage twice does not stack corrections.

Outputs ``<models_dir>/router/_calibration_report.json``.
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, ClassifierCalibrationConfig
from prokopton.ml.features.fragments import random_fragments
from prokopton.ml.pipeline.train import read_sample
from prokopton.ml.training.calibration import (
    LengthCalibratedClassifier,
    fit_length_temperature,
    scaled_log_probabilities,
)
from prokopton.ml.training.metrics import metrics_from_ranks, top_ranks
from prokopton.ml.training.router import predict_proba
from prokopton.ml.training.tracking import Tracker, mlflow_tracker

__all__ = ["REPORT_NAME", "CalibrationReport", "CalibrationStageError", "calibrate"]

logger = logging.getLogger(__name__)

REPORT_NAME = "_calibration_report.json"
_SPLITS = ("val_known", "val_novel")


class CalibrationStageError(RuntimeError):
    """Raised when the calibration stage cannot run."""


class CalibrationReport(BaseModel):
    """Provenance and results of a calibration run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app: str
    model: str = "router"
    release: str
    gene: str
    prokopton_ml_version: str
    rules: ClassifierCalibrationConfig
    started_at: datetime
    finished_at: datetime
    registered_model: str
    base_version: str
    model_version: str
    alias: str
    a: float
    b: float
    nll_before: float
    nll_after: float
    before: dict[str, dict[str, float]]
    after: dict[str, dict[str, float]]
    run_id: str


def _scores(
    log_probabilities: np.ndarray, truth: np.ndarray, sets: np.ndarray
) -> dict[str, dict[str, float]]:
    ranked = top_ranks(log_probabilities)
    confidence = np.exp(log_probabilities.max(axis=1))
    return {
        name: metrics_from_ranks(
            truth[sets == name], ranked[sets == name], confidence[sets == name]
        )
        for name in sorted(set(sets.tolist()))
    }


def calibrate(
    config: AppConfig,
    tracker_factory: Callable[[AppConfig], Tracker] | None = None,
    *,
    model: str = "router",
) -> CalibrationReport:
    """Fit the length-aware temperature, register the result, write a report.

    ``model`` says which classifier to calibrate (``router`` or ``marker``).

    Raises:
        CalibrationStageError: If the validation samples are missing.
    """
    started_at = datetime.now(UTC)
    classifier = config.classifier(model)
    rules = classifier.calibration
    training = classifier.training
    make_tracker = tracker_factory or partial(mlflow_tracker, model=model)
    folder = f"midori2_{config.data.source.release.lower()}"
    sample_root = config.data.paths.samples_dir / folder / model
    splits = [s for s in _SPLITS if s in classifier.sample.eval_splits]
    paths = {name: sample_root / f"{name}.parquet" for name in splits}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"{', '.join(missing)} not found; run the sample stage first"
        raise CalibrationStageError(msg)

    tracker = make_tracker(config)
    loaded, base_version = tracker.load_model(training.registered_model, training.alias)
    if isinstance(loaded, LengthCalibratedClassifier):
        loaded = loaded.model
    logger.info("Calibrating %s version %s", training.registered_model, base_version)

    rng = np.random.default_rng(rules.random_state)
    sequences: list[str] = []
    labels: list[str] = []
    sets: list[str] = []
    for split in splits:
        sample = read_sample(paths[split])
        pieces = random_fragments(
            sample.sequences, rules.fragment_min_length, rules.fragment_max_length, rng
        )
        for name, chosen in ((split, sample.sequences), (f"{split}_fragments", pieces)):
            sequences += chosen
            labels += sample.labels
            sets += [name] * len(chosen)
    index = {str(label): i for i, label in enumerate(loaded.classes_)}
    known = np.array([label in index for label in labels])
    truth = np.array(
        [index[label] for label in labels if label in index], dtype=np.int64
    )
    sequences = [s for s, keep in zip(sequences, known, strict=True) if keep]
    set_names = np.array(sets, dtype=object)[known]

    raw = predict_proba(loaded, sequences, rules.threads, "predict_log_proba")
    lengths = [len(sequence) for sequence in sequences]
    a, b, nll_before, nll_after = fit_length_temperature(raw, truth, lengths)
    scaled = scaled_log_probabilities(raw, np.log(np.maximum(lengths, 1)), a, b)
    before = _scores(raw, truth, set_names)
    after = _scores(scaled, truth, set_names)
    for name in after:
        logger.info(
            "%s: ECE %.4f -> %.4f, mean confidence %.4f -> %.4f (accuracy %.4f)",
            name,
            before[name]["ece"],
            after[name]["ece"],
            before[name]["mean_confidence"],
            after[name]["mean_confidence"],
            after[name]["accuracy"],
        )

    calibrated = LengthCalibratedClassifier(loaded, a, b)
    with tracker.run("router-calibration") as run_id:
        tracker.set_tags(
            {
                "app": config.name,
                "release": config.data.source.release,
                "gene": ",".join(classifier.genes),
                "prokopton_ml_version": __version__,
                "stage": "calibrate",
                "model_role": model,
                "base_model_version": base_version,
            }
        )
        tracker.log_params({"a": a, "b": b, "random_state": rules.random_state})
        tracker.log_metrics(
            {
                "nll_before": nll_before,
                "nll_after": nll_after,
                **{
                    f"{n}_{k}_before": v
                    for n, m in before.items()
                    for k, v in m.items()
                },
                **{f"{n}_{k}_after": v for n, m in after.items() for k, v in m.items()},
            }
        )
        example = pd.DataFrame({"sequence": sequences[:3]})
        version = tracker.log_model(
            calibrated, example, training.registered_model, training.alias
        )
    logger.info(
        "a=%.4f, b=%.4f; NLL %.4f -> %.4f; registered %s version %s as @%s",
        a,
        b,
        nll_before,
        nll_after,
        training.registered_model,
        version,
        training.alias,
    )

    report = CalibrationReport(
        app=config.name,
        model=model,
        release=config.data.source.release,
        gene=",".join(classifier.genes),
        prokopton_ml_version=__version__,
        rules=rules,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        registered_model=training.registered_model,
        base_version=base_version,
        model_version=version,
        alias=training.alias,
        a=a,
        b=b,
        nll_before=nll_before,
        nll_after=nll_after,
        before=before,
        after=after,
        run_id=run_id,
    )
    output = config.data.paths.models_dir / model / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)
    return report
