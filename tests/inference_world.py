"""Shared fakes for the inference tests: COI-like sequences, a marker
classifier that recognises them by their 10-mers, and a fixed router."""

from typing import Any

import numpy as np

from prokopton.ml.features.fragments import reverse_complement

__all__ = [
    "AEGYPTI",
    "ALBOPICTUS",
    "CYTB",
    "MUSCA",
    "FakeMarker",
    "FakeRouter",
    "random_sequence",
]

rng = np.random.default_rng(17)


def random_sequence(length: int) -> str:
    return "".join(rng.choice(list("ACGT"), length))


AEGYPTI = random_sequence(600)
ALBOPICTUS = random_sequence(600)
CYTB = random_sequence(600)
MUSCA = random_sequence(600)


def kmers(sequence: str, k: int = 10) -> set[str]:
    return {sequence[i : i + k] for i in range(len(sequence) - k + 1)}


class FakeMarker:
    """Calls a window CO1 if most of its 10-mers come from known COI."""

    classes_ = np.array(["CO1", "Cytb", "reverse"])

    def __init__(self) -> None:
        self.co1 = kmers(AEGYPTI) | kmers(ALBOPICTUS) | kmers(MUSCA)
        self.reverse = {reverse_complement(k) for k in self.co1}
        self.cytb = kmers(CYTB)

    def predict_proba(self, inputs: Any) -> np.ndarray:
        rows = []
        for sequence in inputs:
            found = kmers(sequence)
            shares = [
                len(found & known) / len(found)
                for known in (self.co1, self.cytb, self.reverse)
            ]
            best = int(np.argmax(shares))
            row = np.full(3, 0.05 / 2)
            if shares[best] >= 0.5:
                row[best] = 0.95
            else:
                row = np.full(3, 1 / 3)
            rows.append(row)
        return np.array(rows)


class FakeRouter:
    def __init__(self, **probabilities: float) -> None:
        self.classes_ = np.array(list(probabilities))
        self.row = np.array(list(probabilities.values()))

    def predict_proba(self, inputs: Any) -> np.ndarray:
        return np.array([self.row for _ in inputs])
