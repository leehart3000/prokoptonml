"""Evaluate stage: measure a classifier chosen by training on the test sets.

Loads the model that ``[router.training]``'s alias points at, predicts every
record in the test splits, and reports:

- headline scores for each split, and for novel-species splits, separately
  for records without (``clean``) and with (``leaky``) a near-copy in
  training;
- scores by ambiguous fraction, calibration and confidence coverage;
- per-class scores, and scores by group of organisms (to show whether the
  model works across groups, rather than learning a shortcut from them);
- the "how short can you go?" table: scores on random fragments of each
  length.

The run is recorded in MLflow, tagged with the model version and the SHA-256
of the data it was measured on. With ``orientation`` on, the fragment
records are also predicted as their reverse complement, to measure how often
the right strand is the more confident one. Outputs
``<models_dir>/<model>/_evaluation_report.json``.
"""

import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from functools import partial
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, ClassifierEvaluationConfig
from prokopton.ml.data.checksums import sha256_of_file
from prokopton.ml.evaluation.scores import (
    AmbiguityBin,
    CalibrationBin,
    ClassScore,
    CoveragePoint,
    GroupScore,
    Predictions,
    by_ambiguity,
    by_group,
    calibration,
    coverage,
    per_class,
)
from prokopton.ml.features.fragments import fragment, reverse_complement
from prokopton.ml.pipeline.sample import label_for
from prokopton.ml.pipeline.split import hash_fraction
from prokopton.ml.training.router import predict_proba
from prokopton.ml.training.tracking import Tracker, mlflow_tracker

__all__ = [
    "REPORT_NAME",
    "EvaluationReport",
    "EvaluationStageError",
    "FragmentScore",
    "SplitScores",
    "evaluate",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_evaluation_report.json"
_COLUMNS = ["record_id", "split", "group", "order", "ambiguous_fraction", "sequence"]
_HEADLINE = ("accuracy", "top5_accuracy", "macro_f1", "ece")


class EvaluationStageError(RuntimeError):
    """Raised when the evaluation stage cannot run."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitScores(_Report):
    """Scores for one set of test records."""

    records: int
    metrics: dict[str, float]
    by_ambiguity: list[AmbiguityBin]
    calibration: list[CalibrationBin]
    coverage: list[CoveragePoint]
    by_group: list[GroupScore] = []


class FragmentScore(_Report):
    """Scores on fragments of one length (``None``: whole sequences)."""

    length: int | None
    records: int
    too_short: int
    metrics: dict[str, float]
    # Share where the forward strand was the more confident (if measured)
    forward_strand_share: float | None = None


class EvaluationReport(_Report):
    """Provenance and results of an evaluation run."""

    app: str
    model: str = "router"
    release: str
    gene: str
    prokopton_ml_version: str
    rules: ClassifierEvaluationConfig
    started_at: datetime
    finished_at: datetime
    registered_model: str
    alias: str
    model_version: str
    data_sha256: dict[str, str]
    unlabelled_records: dict[str, int]
    unseen_class_records: dict[str, int]
    splits: dict[str, SplitScores]
    fragments: dict[str, list[FragmentScore]]
    per_class: dict[str, dict[str, ClassScore]]
    run_id: str


class _Collector:
    """Predicts buffered test records in batches and keeps compact results."""

    def __init__(self, model: Any, rules: ClassifierEvaluationConfig) -> None:
        self.model = model
        self.rules = rules
        self.index = {str(label): i for i, label in enumerate(model.classes_)}
        self.rng = np.random.default_rng(rules.random_state)
        self.buffer: list[dict[str, Any]] = []
        self.sets: dict[str, Predictions] = {}
        self.fragments: dict[tuple[str, int | None], Predictions] = {}
        self.too_short: Counter[tuple[str, int]] = Counter()
        self.forward_wins: Counter[tuple[str, int | None]] = Counter()
        self.strand_checks: Counter[tuple[str, int | None]] = Counter()

    def _predictions(self, name: str) -> Predictions:
        return self.sets.setdefault(name, Predictions())

    def add(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.rules.batch_size:
            self.flush()

    def _predict(self, sequences: list[str]) -> np.ndarray:
        return predict_proba(self.model, sequences, self.rules.threads)

    def flush(self) -> None:
        rows, self.buffer = self.buffer, []
        if not rows:
            return
        probabilities = self._predict([row["sequence"] for row in rows])
        truth = np.array([self.index[row["label"]] for row in rows], dtype=np.int32)
        ambiguity = np.array([row["ambiguous_fraction"] for row in rows])
        groups = np.array([row["group"] or "" for row in rows], dtype=object)
        names = np.array([row["set"] for row in rows], dtype=object)
        for name in {row["set"] for row in rows}:
            mask = names == name
            self._predictions(name).add(
                truth[mask], probabilities[mask], ambiguity[mask], groups[mask]
            )
        self._flush_fragments(rows, truth, probabilities, ambiguity)

    def _check_strand(
        self, key: tuple[str, int | None], sequences: list[str], forward: np.ndarray
    ) -> None:
        if not self.rules.orientation:
            return
        reverse = self._predict([reverse_complement(s) for s in sequences])
        self.forward_wins[key] += int(
            (forward.max(axis=1) >= reverse.max(axis=1)).sum()
        )
        self.strand_checks[key] += len(sequences)

    def _flush_fragments(
        self,
        rows: list[dict[str, Any]],
        truth: np.ndarray,
        probabilities: np.ndarray,
        ambiguity: np.ndarray,
    ) -> None:
        chosen = [i for i, row in enumerate(rows) if row["fragment_set"] is not None]
        for fragment_set in {rows[i]["fragment_set"] for i in chosen}:
            members = [i for i in chosen if rows[i]["fragment_set"] == fragment_set]
            self.fragments.setdefault((fragment_set, None), Predictions()).add(
                truth[members], probabilities[members], ambiguity[members]
            )
            self._check_strand(
                (fragment_set, None),
                [rows[i]["sequence"] for i in members],
                probabilities[members],
            )
            for length in self.rules.fragment_lengths:
                cut: list[str] = []
                kept: list[int] = []
                for i in members:
                    piece = fragment(rows[i]["sequence"], length, self.rng)
                    if piece is None:
                        self.too_short[fragment_set, length] += 1
                    else:
                        cut.append(piece)
                        kept.append(i)
                if kept:
                    predicted = self._predict(cut)
                    self.fragments.setdefault(
                        (fragment_set, length), Predictions()
                    ).add(truth[kept], predicted, ambiguity[kept])
                    self._check_strand((fragment_set, length), cut, predicted)


def _split_scores(
    predictions: Predictions, rules: ClassifierEvaluationConfig
) -> SplitScores:
    return SplitScores(
        records=len(predictions),
        metrics=predictions.metrics(),
        by_ambiguity=by_ambiguity(predictions, rules.ambiguity_edges),
        calibration=calibration(predictions, rules.calibration_bins),
        coverage=coverage(predictions, rules.confidence_thresholds),
        by_group=by_group(predictions),
    )


def evaluate(
    config: AppConfig,
    tracker_factory: Callable[[AppConfig], Tracker] | None = None,
    *,
    model: str = "router",
) -> EvaluationReport:
    """Measure a classifier on the test splits, record it and write a report.

    ``model`` says which classifier to evaluate (``router`` or ``marker``).

    Raises:
        EvaluationStageError: If the split or leakage data is missing.
    """
    started_at = datetime.now(UTC)
    classifier = config.classifier(model)
    rules = classifier.evaluation
    training = classifier.training
    make_tracker = tracker_factory or partial(mlflow_tracker, model=model)
    folder = f"midori2_{config.data.source.release.lower()}"
    split_paths = {
        gene: config.data.paths.splits_dir / folder / f"gene={gene}" / "records.parquet"
        for gene in classifier.genes
    }
    # Near-copies are only listed for the genes the leakage stage covers
    leakage_paths = {
        gene: config.data.paths.leakage_dir
        / folder
        / f"gene={gene}"
        / "near_duplicates.parquet"
        for gene in classifier.genes
        if rules.novel_splits and gene in config.leakage.genes
    }
    needed = [*split_paths.values(), *leakage_paths.values()]
    missing = [str(path) for path in needed if not path.exists()]
    if missing:
        msg = f"{', '.join(missing)} not found; run the split and leakage stages first"
        raise EvaluationStageError(msg)

    single = len(split_paths) == 1
    data_sha256 = {
        "records" if single else f"records.{gene}": sha256_of_file(path)
        for gene, path in split_paths.items()
    }
    leaky: set[str] = set()
    for gene, path in leakage_paths.items():
        data_sha256["near_duplicates" if single else f"near_duplicates.{gene}"] = (
            sha256_of_file(path)
        )
        leaky |= set(
            pq.read_table(path, columns=["record_id"]).column("record_id").to_pylist()
        )

    tracker = make_tracker(config)
    loaded, version = tracker.load_model(training.registered_model, training.alias)
    logger.info(
        "Evaluating %s version %s (@%s)",
        training.registered_model,
        version,
        training.alias,
    )
    novel = set(rules.novel_splits)
    collector = _Collector(loaded, rules)
    wanted = set(rules.splits)
    unlabelled: Counter[str] = Counter()
    unseen: Counter[str] = Counter()
    for gene, split_path in split_paths.items():
        for batch in pq.ParquetFile(split_path).iter_batches(
            batch_size=rules.batch_size, columns=_COLUMNS
        ):
            for row in batch.to_pylist():
                split = row["split"]
                if split not in wanted:
                    continue
                label = (
                    gene
                    if classifier.label == "gene"
                    else label_for(row["group"], row["order"])
                )
                if label is None:
                    unlabelled[split] += 1
                    continue
                if label not in collector.index:
                    unseen[split] += 1
                    continue
                name = split
                if split in novel:
                    leaked = row["record_id"] in leaky
                    name = f"{split}_leaky" if leaked else f"{split}_clean"
                chosen = (
                    hash_fraction(row["record_id"], rules.salt)
                    < rules.fragment_fraction
                )
                # Fragments of novel splits use only clean records
                fragment_set = name if chosen and not name.endswith("_leaky") else None
                collector.add(
                    {**row, "label": label, "set": name, "fragment_set": fragment_set}
                )
    collector.flush()

    # Whole novel splits combine the clean and leaky records
    for split in novel & wanted:
        combined = Predictions()
        for part in (f"{split}_clean", f"{split}_leaky"):
            found = collector.sets.get(part, Predictions())
            for field_name in ("truth", "ranked", "confidence", "ambiguity", "group"):
                getattr(combined, field_name).extend(getattr(found, field_name))
        collector.sets[split] = combined

    splits = {
        name: _split_scores(predictions, rules)
        for name, predictions in sorted(collector.sets.items())
    }
    fragment_sets = sorted({name for name, _ in collector.fragments})
    fragments = {
        name: [
            FragmentScore(
                length=length,
                records=len(collector.fragments.get((name, length), Predictions())),
                too_short=collector.too_short[name, length] if length else 0,
                metrics=collector.fragments.get(
                    (name, length), Predictions()
                ).metrics(),
                forward_strand_share=(
                    collector.forward_wins[name, length]
                    / collector.strand_checks[name, length]
                    if collector.strand_checks[name, length]
                    else None
                ),
            )
            for length in [None, *rules.fragment_lengths]
        ]
        for name in fragment_sets
    }
    classes = [str(label) for label in loaded.classes_]
    class_scores = {
        name: per_class(predictions, classes)
        for name, predictions in collector.sets.items()
        if not name.endswith("_leaky") and name not in novel
    }

    metrics: dict[str, float] = {}
    for name, scores in splits.items():
        metrics |= {f"{name}_{key}": scores.metrics[key] for key in _HEADLINE}
        metrics[f"{name}_records"] = float(scores.records)
    for name, points in fragments.items():
        for point in points:
            suffix = f"len{point.length}" if point.length else "full"
            metrics[f"{name}_{suffix}_accuracy"] = point.metrics["accuracy"]
            metrics[f"{name}_{suffix}_macro_f1"] = point.metrics["macro_f1"]

    with tracker.run("router-evaluation") as run_id:
        tracker.set_tags(
            {
                "app": config.name,
                "release": config.data.source.release,
                "gene": ",".join(classifier.genes),
                "prokopton_ml_version": __version__,
                "stage": "evaluate",
                "model_role": model,
                "registered_model": training.registered_model,
                "model_version": version,
                **{f"data_sha256.{key}": value for key, value in data_sha256.items()},
            }
        )
        tracker.log_params(
            {
                "fragment_fraction": rules.fragment_fraction,
                "fragment_lengths": ",".join(map(str, rules.fragment_lengths)),
                "random_state": rules.random_state,
            }
        )
        tracker.log_metrics(metrics)
        report = EvaluationReport(
            app=config.name,
            model=model,
            release=config.data.source.release,
            gene=",".join(classifier.genes),
            prokopton_ml_version=__version__,
            rules=rules,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            registered_model=training.registered_model,
            alias=training.alias,
            model_version=version,
            data_sha256=data_sha256,
            unlabelled_records=dict(sorted(unlabelled.items())),
            unseen_class_records=dict(sorted(unseen.items())),
            splits=splits,
            fragments=fragments,
            per_class=class_scores,
            run_id=run_id,
        )
        tracker.log_dict(report.model_dump(mode="json"), "evaluation_report.json")

    for name, scores in splits.items():
        logger.info(
            "%s: %d records, accuracy %.4f, macro-F1 %.4f, ECE %.4f",
            name,
            scores.records,
            scores.metrics["accuracy"],
            scores.metrics["macro_f1"],
            scores.metrics["ece"],
        )
    output = config.data.paths.models_dir / model / REPORT_NAME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output)
    return report
