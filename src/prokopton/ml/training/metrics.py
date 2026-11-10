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

__all__ = ["classification_metrics", "expected_calibration_error"]


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
        return {"accuracy": 0.0, "top5_accuracy": 0.0, "macro_f1": 0.0, "ece": 0.0}
    class_array = np.asarray(classes, dtype=object)
    truth = np.asarray(labels, dtype=object)
    ranked = np.argsort(-probabilities, axis=1)
    predicted = class_array[ranked[:, 0]]
    correct = predicted == truth
    top5 = (class_array[ranked[:, :5]] == truth[:, None]).any(axis=1)
    present = sorted(set(truth.tolist()))
    return {
        "accuracy": float(correct.mean()),
        "top5_accuracy": float(top5.mean()),
        "macro_f1": float(
            f1_score(truth, predicted, labels=present, average="macro", zero_division=0)
        ),
        "ece": expected_calibration_error(probabilities.max(axis=1), correct),
    }
