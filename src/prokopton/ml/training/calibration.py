"""Length-aware temperature scaling of a classifier's confidence.

A model's probabilities can be made sharper or softer with a *temperature*
``T``: each class's log-probability is divided by ``T`` and the result
rescaled to add up to 1. ``T`` below 1 sharpens (for an underconfident
model), above 1 softens (for an overconfident one). The top class never
changes, so accuracy is unaffected.

Short inputs need a different correction from long ones, so here the
temperature depends on the sequence's length::

    T = exp(a + b * log(length))

``a`` and ``b`` are fitted on validation data by minimising the negative
log-likelihood (the average of ``-log(probability of the true class)``).
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = [
    "LengthCalibratedClassifier",
    "fit_length_temperature",
    "scaled_log_probabilities",
    "sequences_of",
]

_FLOOR = -1e4
"""Lowest log-probability used, so a zero probability stays finite."""


def sequences_of(inputs: Any) -> list[str]:
    """Return the sequences from a list, or a table's first column."""
    columns = getattr(inputs, "columns", None)
    if columns is not None:
        inputs = inputs[columns[0]]
    return [str(sequence) for sequence in inputs]


def _log_lengths(sequences: Sequence[str]) -> np.ndarray:
    return np.log(np.maximum([len(sequence) for sequence in sequences], 1.0))


def scaled_log_probabilities(
    log_probabilities: np.ndarray, log_lengths: np.ndarray, a: float, b: float
) -> np.ndarray:
    """Apply the length-aware temperature, returning log-probabilities."""
    temperature = np.exp(a + b * log_lengths)[:, None]
    scaled: np.ndarray = log_softmax(
        np.maximum(log_probabilities, _FLOOR) / temperature, axis=1
    )
    return scaled


def fit_length_temperature(
    log_probabilities: np.ndarray, truth: np.ndarray, lengths: Sequence[int]
) -> tuple[float, float, float, float]:
    """Fit ``a`` and ``b`` to minimise negative log-likelihood.

    Args:
        log_probabilities: The model's log-probabilities, one row per record.
        truth: Each record's true class number (records the model lacks a
            class for should be left out).
        lengths: Each record's sequence length.

    Returns:
        ``a``, ``b``, and the negative log-likelihood before and after.
    """
    log_lengths = np.log(np.maximum(np.asarray(lengths, dtype=np.float64), 1.0))
    rows = np.arange(truth.size)

    def loss(params: np.ndarray) -> float:
        scaled = scaled_log_probabilities(
            log_probabilities, log_lengths, float(params[0]), float(params[1])
        )
        return float(-scaled[rows, truth].mean())

    before = loss(np.zeros(2))
    found = minimize(
        loss,
        x0=np.zeros(2),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-7},
    )
    a, b = (float(value) for value in found.x)
    return a, b, before, float(found.fun)


class LengthCalibratedClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """A fitted model whose confidence is corrected by sequence length.

    Wraps a model that takes sequences (such as a pipeline starting with
    ``KmerVectorizer``) and has ``predict_log_proba``. Predictions are the
    wrapped model's; only the probabilities change.
    """

    def __init__(self, model: Any = None, a: float = 0.0, b: float = 0.0) -> None:
        self.model = model
        self.a = a
        self.b = b

    @property
    def classes_(self) -> np.ndarray:
        classes: np.ndarray = self.model.classes_
        return classes

    def fit(
        self, inputs: Any = None, labels: Any = None
    ) -> "LengthCalibratedClassifier":
        """Do nothing: the wrapped model and ``a`` and ``b`` are already fitted."""
        return self

    def predict_log_proba(self, inputs: Any) -> np.ndarray:
        """Return length-calibrated log-probabilities, one row per sequence."""
        sequences = sequences_of(inputs)
        raw = self.model.predict_log_proba(sequences)
        return scaled_log_probabilities(raw, _log_lengths(sequences), self.a, self.b)

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return length-calibrated probabilities, one row per sequence."""
        probabilities: np.ndarray = np.exp(self.predict_log_proba(inputs))
        return probabilities

    def predict(self, inputs: Any) -> np.ndarray:
        """Return the most probable class for each sequence."""
        predicted: np.ndarray = self.model.predict(sequences_of(inputs))
        return predicted
