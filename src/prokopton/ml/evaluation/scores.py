"""Summaries of a model's test predictions, kept small enough to report.

Predictions are collected as a few numbers per record (true class, top
classes, confidence, ambiguity), so millions of records fit in memory. From
those, this module builds the tables shown in the model card:

- **by ambiguity**: scores for records with different shares of ambiguous
  bases (such as ``N``).
- **calibration**: in each confidence bin, how often the model was right.
- **coverage**: if only predictions above a confidence threshold are
  accepted, what share is accepted and how accurate those are.
- **per class**: precision, recall and F1 for each class.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.metrics import precision_recall_fscore_support

from prokopton.ml.features.fragments import fragment
from prokopton.ml.training.metrics import TOP_K, metrics_from_ranks, top_ranks

__all__ = [
    "AmbiguityBin",
    "CalibrationBin",
    "ClassScore",
    "CoveragePoint",
    "GroupScore",
    "Predictions",
    "by_ambiguity",
    "by_group",
    "calibration",
    "coverage",
    "fragment",
    "per_class",
]


class _Row(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AmbiguityBin(_Row):
    """Scores for records whose ambiguous fraction is in one range."""

    bin: str
    records: int
    accuracy: float
    macro_f1: float


class GroupScore(_Row):
    """Scores for records from one group of organisms."""

    group: str
    records: int
    accuracy: float
    macro_f1: float


class CalibrationBin(_Row):
    """How often predictions in one confidence range were right."""

    lower: float
    upper: float
    records: int
    mean_confidence: float
    accuracy: float


class CoveragePoint(_Row):
    """The share accepted, and its accuracy, at one confidence threshold."""

    threshold: float
    coverage: float
    accuracy: float


class ClassScore(_Row):
    """Precision, recall and F1 for one class."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class Predictions:
    """A growing collection of compact predictions for one set of records."""

    truth: list[np.ndarray] = field(default_factory=list)
    ranked: list[np.ndarray] = field(default_factory=list)
    confidence: list[np.ndarray] = field(default_factory=list)
    ambiguity: list[np.ndarray] = field(default_factory=list)
    group: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        truth: np.ndarray,
        probabilities: np.ndarray,
        ambiguity: np.ndarray,
        group: np.ndarray | None = None,
    ) -> None:
        """Add a batch of true class numbers and predicted probabilities.

        ``group`` optionally names each record's group of organisms.
        """
        if truth.size == 0:
            return
        self.truth.append(truth.astype(np.int32))
        self.ranked.append(top_ranks(probabilities, TOP_K).astype(np.int32))
        self.confidence.append(probabilities.max(axis=1).astype(np.float64))
        self.ambiguity.append(ambiguity.astype(np.float64))
        self.group.append(
            np.full(truth.size, "", dtype=object)
            if group is None
            else np.asarray(group, dtype=object)
        )

    def groups(self) -> np.ndarray:
        """Each record's group of organisms ("" where not given)."""
        if not self.group:
            return np.zeros(0, dtype=object)
        groups: np.ndarray = np.concatenate(self.group)
        return groups

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return truth, ranked, confidence and ambiguity as single arrays."""
        if not self.truth:
            empty = np.zeros(0)
            return empty.astype(np.int32), np.zeros((0, TOP_K), np.int32), empty, empty
        return (
            np.concatenate(self.truth),
            np.concatenate(self.ranked),
            np.concatenate(self.confidence),
            np.concatenate(self.ambiguity),
        )

    def __len__(self) -> int:
        return sum(part.size for part in self.truth)

    def metrics(self) -> dict[str, float]:
        """Accuracy, top-5 accuracy, macro-F1 and ECE."""
        truth, ranked, confidence, _ = self.arrays()
        return metrics_from_ranks(truth, ranked, confidence)


def by_ambiguity(
    predictions: Predictions, edges: Sequence[float]
) -> list[AmbiguityBin]:
    """Score records grouped by ambiguous fraction.

    With edges ``(0, 0.01, 0.03)`` the groups are exactly 0, above 0 up to
    0.01, above 0.01 up to 0.03, and above 0.03.
    """
    truth, ranked, confidence, ambiguity = predictions.arrays()
    bounds = sorted(edges)
    groups: list[tuple[str, np.ndarray]] = []
    if bounds and bounds[0] == 0.0:
        groups.append(("0", ambiguity == 0.0))
    elif bounds:
        groups.append((f"<= {bounds[0]:g}", ambiguity <= bounds[0]))
    for lower, upper in pairwise(bounds):
        groups.append(
            (f"({lower:g}, {upper:g}]", (ambiguity > lower) & (ambiguity <= upper))
        )
    if bounds:
        groups.append((f"> {bounds[-1]:g}", ambiguity > bounds[-1]))
    rows = []
    for name, mask in groups:
        found = metrics_from_ranks(truth[mask], ranked[mask], confidence[mask])
        rows.append(
            AmbiguityBin(
                bin=name,
                records=int(mask.sum()),
                accuracy=found["accuracy"],
                macro_f1=found["macro_f1"],
            )
        )
    return rows


def by_group(predictions: Predictions) -> list[GroupScore]:
    """Score records grouped by their group of organisms, largest first."""
    truth, ranked, confidence, _ = predictions.arrays()
    groups = predictions.groups()
    rows = []
    for name in sorted({str(g) for g in groups if g}):
        mask = groups == name
        found = metrics_from_ranks(truth[mask], ranked[mask], confidence[mask])
        rows.append(
            GroupScore(
                group=name,
                records=int(mask.sum()),
                accuracy=found["accuracy"],
                macro_f1=found["macro_f1"],
            )
        )
    return sorted(rows, key=lambda row: -row.records)


def calibration(predictions: Predictions, bins: int) -> list[CalibrationBin]:
    """Split predictions into equal-width confidence bins (empty bins left out)."""
    truth, ranked, confidence, _ = predictions.arrays()
    correct = ranked[:, 0] == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for index in range(bins):
        mask = which == index
        if mask.any():
            rows.append(
                CalibrationBin(
                    lower=float(edges[index]),
                    upper=float(edges[index + 1]),
                    records=int(mask.sum()),
                    mean_confidence=float(confidence[mask].mean()),
                    accuracy=float(correct[mask].mean()),
                )
            )
    return rows


def coverage(
    predictions: Predictions, thresholds: Sequence[float]
) -> list[CoveragePoint]:
    """The share and accuracy of predictions at or above each threshold."""
    truth, ranked, confidence, _ = predictions.arrays()
    correct = ranked[:, 0] == truth
    rows = []
    for threshold in thresholds:
        accepted = confidence >= threshold
        rows.append(
            CoveragePoint(
                threshold=threshold,
                coverage=float(accepted.mean()) if accepted.size else 0.0,
                accuracy=float(correct[accepted].mean()) if accepted.any() else 0.0,
            )
        )
    return rows


def per_class(
    predictions: Predictions, classes: Sequence[str]
) -> dict[str, ClassScore]:
    """Precision, recall and F1 for each class that has test records."""
    truth, ranked, _, _ = predictions.arrays()
    present = np.unique(truth).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, ranked[:, 0], labels=present, zero_division=0
    )
    return {
        classes[label]: ClassScore(
            precision=float(p), recall=float(r), f1=float(f), support=int(s)
        )
        for label, p, r, f, s in zip(
            present, precision, recall, f1, support, strict=True
        )
    }
