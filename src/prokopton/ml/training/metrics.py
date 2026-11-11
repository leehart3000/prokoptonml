"""Classification metrics used to compare and choose models.

- **accuracy**: share of records whose top prediction is correct.
- **top-5 accuracy**: share whose true label is among the five most probable.
- **macro-F1**: the F1 score averaged over classes with equal weight, so rare
  classes count as much as common ones.
- **expected calibration error (ECE)**: how far confidence is from accuracy.
  Predictions are grouped into bins by confidence; in each bin, the gap
  between average confidence and accuracy is weighted by the bin's size. 0 is
  perfect: "80% confident" means right 80% of the time.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import f1_score

__all__ = [
    "TOP_K",
    "classification_metrics",
    "expected_calibration_error",
    "metrics_from_ranks",
    "top_ranks",
]

TOP_K = 5


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, bins: int = 15
) -> float:
    """Return the expected calibration error of top-1 predictions."""
    if confidence.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        in_bin = which == index
        if in_bin.any():
            gap = abs(float(confidence[in_bin].mean()) - float(correct[in_bin].mean()))
            error += gap * in_bin.mean()
    return float(error)


def classification_metrics(
    labels: Sequence[str], probabilities: np.ndarray, classes: Sequence[str]
) -> dict[str, float]:
    """Return accuracy, top-5 accuracy, macro-F1 and ECE.

    Args:
        labels: The true label of each record.
        probabilities: Predicted probabilities, one column per class.
        classes: The class of each probability column.

    Raises:
        ValueError: If shapes do not match.
    """
    if probabilities.shape != (len(labels), len(classes)):
        msg = (
            f"probabilities has shape {probabilities.shape}, expected "
            f"({len(labels)}, {len(classes)})"
        )
        raise ValueError(msg)
    if not labels:
        return {
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "mean_confidence": 0.0,
        }
    index = {label: position for position, label in enumerate(classes)}
    # A label the model lacks gets its own negative number, so it still
    # counts (with an F1 of 0) in macro-F1
    for label in labels:
        index.setdefault(label, -1 - (len(index) - len(classes)))
    truth = np.array([index[label] for label in labels], dtype=np.int64)
    return metrics_from_ranks(
        truth, top_ranks(probabilities), probabilities.max(axis=1)
    )


def top_ranks(probabilities: np.ndarray, top: int = TOP_K) -> np.ndarray:
    """Return the column numbers of each row's ``top`` most probable classes.

    The first column is the top prediction.
    """
    top = min(top, probabilities.shape[1])
    best = np.argpartition(-probabilities, top - 1, axis=1)[:, :top]
    order = np.argsort(-np.take_along_axis(probabilities, best, axis=1), axis=1)
    ranked: np.ndarray = np.take_along_axis(best, order, axis=1)
    return ranked


def metrics_from_ranks(
    truth: np.ndarray, ranked: np.ndarray, confidence: np.ndarray
) -> dict[str, float]:
    """Return the metrics from class numbers rather than probabilities.

    Args:
        truth: Each record's true class number; negative numbers stand for
            classes the model lacks.
        ranked: Each record's most probable class numbers, best first.
        confidence: Each record's top probability.

    This lets large test sets be predicted in batches, keeping only a few
    numbers per record.
    """
    if truth.size == 0:
        return {
            "accuracy": 0.0,
            "top5_accuracy": 0.0,
            "macro_f1": 0.0,
            "ece": 0.0,
            "mean_confidence": 0.0,
        }
    correct = ranked[:, 0] == truth
    present = np.unique(truth).tolist()
    return {
        "accuracy": float(correct.mean()),
        "top5_accuracy": float(
            (ranked[:, :TOP_K] == truth[:, None]).any(axis=1).mean()
        ),
        "macro_f1": float(
            f1_score(
                truth,
                ranked[:, 0],
                labels=present,
                average="macro",
                zero_division=0,
            )
        ),
        "ece": expected_calibration_error(confidence, correct),
        # Above accuracy: overconfident; below: underconfident
        "mean_confidence": float(confidence.mean()),
    }
