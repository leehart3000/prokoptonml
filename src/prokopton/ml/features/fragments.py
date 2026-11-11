"""Random fragments of sequences, for training and testing on short inputs.

Real queries are often shorter than the full barcodes in the reference data
(for example, "mini-barcodes" from environmental DNA). A model trained only
on full-length sequences is both less accurate and overconfident on these,
so training adds random fragments, and evaluation measures each length.
"""

from collections.abc import Sequence

import numpy as np

__all__ = ["fragment", "random_fragments"]


def fragment(sequence: str, length: int, rng: np.random.Generator) -> str | None:
    """Cut ``length`` bases from a random position, or None if too short."""
    if len(sequence) < length:
        return None
    start = int(rng.integers(0, len(sequence) - length + 1))
    return sequence[start : start + length]


def random_fragments(
    sequences: Sequence[str],
    min_length: int,
    max_length: int,
    rng: np.random.Generator,
) -> list[str]:
    """Cut one fragment from each sequence, of a random length.

    Each length is drawn evenly from ``min_length`` to ``max_length``
    (inclusive). A sequence no longer than its drawn length is kept whole.
    """
    lengths = rng.integers(min_length, max_length + 1, size=len(sequences))
    pieces = []
    for sequence, length in zip(sequences, lengths, strict=True):
        piece = fragment(sequence, int(length), rng)
        pieces.append(sequence if piece is None else piece)
    return pieces
