"""Tests for prokopton.ml.training.metrics."""

import numpy as np
import pytest

from prokopton.ml.training.metrics import (
    classification_metrics,
    expected_calibration_error,
    metrics_from_ranks,
    top_ranks,
)

CLASSES = ["a", "b", "c", "d", "e", "f"]


def one_hot(rows: list[int], confidence: float = 1.0) -> np.ndarray:
    probabilities = np.full(
        (len(rows), len(CLASSES)), (1 - confidence) / (len(CLASSES) - 1)
    )
    for row, column in enumerate(rows):
        probabilities[row, column] = confidence
    return probabilities


def test_perfect_predictions() -> None:
    metrics = classification_metrics(["a", "b", "c"], one_hot([0, 1, 2]), CLASSES)

    assert metrics == {
        "accuracy": 1.0,
        "top5_accuracy": 1.0,
        "macro_f1": 1.0,
        "ece": 0.0,
    }


def test_wrong_top_prediction_can_still_be_in_top_five() -> None:
    probabilities = np.array([[0.05, 0.4, 0.3, 0.1, 0.1, 0.05]])

    metrics = classification_metrics(["a"], probabilities, CLASSES)

    assert metrics["accuracy"] == 0.0
    assert metrics["top5_accuracy"] == 1.0


def test_true_label_outside_top_five() -> None:
    probabilities = np.array([[0.01, 0.3, 0.3, 0.2, 0.1, 0.09]])

    metrics = classification_metrics(["a"], probabilities, CLASSES)

    assert metrics["top5_accuracy"] == 0.0


def test_macro_f1_weights_classes_equally() -> None:
    # Class "a" is always right (8 records); class "b" is always wrong (2 records)
    labels = ["a"] * 8 + ["b"] * 2
    metrics = classification_metrics(labels, one_hot([0] * 10), CLASSES)

    assert metrics["accuracy"] == 0.8
    assert metrics["macro_f1"] == pytest.approx((8 / 9 + 0) / 2)


def test_overconfident_predictions_have_high_calibration_error() -> None:
    confidence = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([True, False, False, False])

    assert expected_calibration_error(confidence, correct) == pytest.approx(0.65)


def test_empty_inputs() -> None:
    assert expected_calibration_error(np.array([]), np.array([])) == 0.0
    assert classification_metrics([], np.empty((0, 6)), CLASSES)["accuracy"] == 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="expected"):
        classification_metrics(["a"], np.zeros((1, 3)), CLASSES)


def test_top_ranks_orders_best_first() -> None:
    probabilities = np.array([[0.1, 0.5, 0.2, 0.15, 0.05, 0.0]])

    assert top_ranks(probabilities).tolist() == [[1, 2, 3, 0, 4]]
    assert top_ranks(probabilities[:, :2]).tolist() == [[1, 0]]


def test_label_the_model_lacks_counts_in_macro_f1() -> None:
    # "z" is not a model class: wrong, and an extra class with F1 0
    metrics = classification_metrics(["a", "z"], one_hot([0, 0]), CLASSES)

    assert metrics["accuracy"] == 0.5
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 0) / 2)


def test_metrics_from_ranks_with_no_records() -> None:
    empty = np.zeros(0, dtype=np.int32)

    assert metrics_from_ranks(empty, np.zeros((0, 5), np.int32), empty)["accuracy"] == 0
