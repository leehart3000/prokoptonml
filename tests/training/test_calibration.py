"""Tests for prokopton.ml.training.calibration."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import log_softmax
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.training.calibration import (
    LengthCalibratedClassifier,
    fit_length_temperature,
    scaled_log_probabilities,
    sequences_of,
)


def test_sequences_of_a_list_or_a_table() -> None:
    assert sequences_of(["ACGT", "GG"]) == ["ACGT", "GG"]
    assert sequences_of(pd.DataFrame({"sequence": ["AC"], "x": [1]})) == ["AC"]


def test_temperature_one_keeps_probabilities() -> None:
    log_probabilities = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))

    scaled = scaled_log_probabilities(log_probabilities, np.zeros(2), 0.0, 0.0)

    np.testing.assert_allclose(scaled, log_probabilities)


def test_low_temperature_sharpens_and_keeps_the_top_class() -> None:
    log_probabilities = np.log(np.array([[0.5, 0.3, 0.2]]))

    scaled = np.exp(
        scaled_log_probabilities(log_probabilities, np.zeros(1), np.log(0.5), 0.0)
    )

    assert scaled.argmax() == 0
    assert scaled[0, 0] > 0.5
    assert scaled.sum() == pytest.approx(1.0)


def simulate(
    rng: np.random.Generator, records: int, true_temperature: float
) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose real class probabilities are softmax(logits / T)."""
    logits = rng.normal(0.0, 2.0, size=(records, 4))
    real = np.exp(log_softmax(logits / true_temperature, axis=1))
    truth = np.array([rng.choice(4, p=row) for row in real])
    return log_softmax(logits, axis=1), truth


def test_fit_finds_the_temperature_for_each_length() -> None:
    rng = np.random.default_rng(7)
    # Short inputs need softening (T=2), long ones sharpening (T=0.5)
    short, short_truth = simulate(rng, 4000, 2.0)
    long, long_truth = simulate(rng, 4000, 0.5)
    log_probabilities = np.vstack([short, long])
    truth = np.concatenate([short_truth, long_truth])
    lengths = [100] * 4000 + [600] * 4000

    a, b, before, after = fit_length_temperature(log_probabilities, truth, lengths)

    assert after < before
    assert np.exp(a + b * np.log(100)) == pytest.approx(2.0, rel=0.15)
    assert np.exp(a + b * np.log(600)) == pytest.approx(0.5, rel=0.15)


def make_model() -> Pipeline:
    sequences = ["ACGTTGCA" * 10, "GGGCCCAT" * 10] * 3
    labels = ["Diptera", "Hymenoptera"] * 3
    model = Pipeline([("kmers", KmerVectorizer(k=3)), ("classifier", MultinomialNB())])
    return model.fit(sequences, labels)


def test_wrapper_keeps_predictions_and_changes_only_confidence() -> None:
    model = make_model()
    queries = ["ACGTTGCA" * 3, "GGGCCCAT" * 10, "ACGTGGGC" * 5]
    unchanged = LengthCalibratedClassifier(model)
    sharper = LengthCalibratedClassifier(model, a=np.log(0.5), b=0.0)

    assert unchanged.fit() is unchanged
    assert list(unchanged.classes_) == ["Diptera", "Hymenoptera"]
    np.testing.assert_allclose(
        unchanged.predict_proba(queries), model.predict_proba(queries)
    )
    assert sharper.predict(queries).tolist() == model.predict(queries).tolist()
    assert (
        sharper.predict_proba(queries).max(axis=1)
        >= model.predict_proba(queries).max(axis=1)
    ).all()
    table = pd.DataFrame({"sequence": queries})
    assert sharper.predict(table).tolist() == sharper.predict(queries).tolist()
