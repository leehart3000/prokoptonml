"""Tests for prokopton.ml.evaluation.scores."""

import numpy as np
import pytest

from prokopton.ml.evaluation.scores import (
    Predictions,
    by_ambiguity,
    calibration,
    coverage,
    fragment,
    per_class,
)


def make_predictions() -> Predictions:
    # Four records over three classes; the first three are right
    probabilities = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.2, 0.7, 0.1],
            [0.1, 0.3, 0.6],
            [0.55, 0.4, 0.05],
        ]
    )
    predictions = Predictions()
    predictions.add(
        np.array([0, 1, 2, 1]), probabilities, np.array([0.0, 0.005, 0.02, 0.5])
    )
    return predictions


def test_predictions_collect_batches() -> None:
    predictions = make_predictions()
    predictions.add(np.array([], dtype=np.int32), np.zeros((0, 3)), np.zeros(0))

    assert len(predictions) == 4
    assert predictions.metrics()["accuracy"] == 0.75


def test_empty_predictions() -> None:
    assert len(Predictions()) == 0
    assert Predictions().metrics()["accuracy"] == 0.0
    assert calibration(Predictions(), 5) == []


def test_fragment_cuts_a_piece_of_the_right_length() -> None:
    rng = np.random.default_rng(1)

    piece = fragment("ACGTACGTAC", 4, rng)

    assert piece is not None
    assert len(piece) == 4
    assert piece in "ACGTACGTAC"
    assert fragment("ACG", 4, rng) is None
    assert fragment("ACGT", 4, rng) == "ACGT"


def test_by_ambiguity_groups_records() -> None:
    rows = by_ambiguity(make_predictions(), (0.0, 0.01, 0.03))

    assert [(row.bin, row.records) for row in rows] == [
        ("0", 1),
        ("(0, 0.01]", 1),
        ("(0.01, 0.03]", 1),
        ("> 0.03", 1),
    ]
    assert rows[-1].accuracy == 0.0
    assert by_ambiguity(make_predictions(), (0.01,))[0].bin == "<= 0.01"


def test_calibration_bins() -> None:
    rows = calibration(make_predictions(), bins=2)

    assert [(row.lower, row.upper, row.records) for row in rows] == [(0.5, 1.0, 4)]
    assert rows[0].accuracy == 0.75
    assert rows[0].mean_confidence == pytest.approx((0.9 + 0.7 + 0.6 + 0.55) / 4)


def test_coverage_at_thresholds() -> None:
    rows = coverage(make_predictions(), (0.5, 0.65, 0.95))

    assert [(row.coverage, row.accuracy) for row in rows] == [
        (1.0, 0.75),
        (0.5, 1.0),
        (0.0, 0.0),
    ]


def test_per_class_scores() -> None:
    scores = per_class(make_predictions(), ["a", "b", "c"])

    assert scores["a"].precision == 0.5
    assert scores["b"].recall == 0.5
    assert scores["c"].f1 == 1.0
    assert scores["b"].support == 2
