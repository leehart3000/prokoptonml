"""Percent identity between two DNA sequences, by alignment.

Barcoding results are usually given as percent identity: the share of
aligned positions where two sequences have the same base. K-mer similarity
is fast for finding candidates, but is not identity, so each candidate is
aligned to the query.

The alignment is global with free end gaps (an *overlap* alignment): either
sequence may start or end inside the other, so a short query can sit
anywhere within a longer reference, and overhanging ends cost nothing. A
match scores +1, a mismatch -1 and each gap position -2. Ambiguous bases
(anything but A, C, G and T) score 0, and are left out of the identity.

Each row of the dynamic-programming table is computed with NumPy: the
best move from the left is a running maximum, so no inner Python loop is
needed.

Free end gaps mean unrelated sequences can "align" over a few bases at
their ends, with high identity, so identity must always be read together
with ``coverage`` and ``aligned``. Coverage is measured on whichever
sequence lies more completely inside the alignment: a 658-base barcode
inside a 1,500-base CO1 gene is fully covered, whichever is the query.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ["Alignment", "align"]

MATCH = 1
MISMATCH = -1
GAP = 2
_BASES = np.zeros(256, dtype=bool)
_BASES[[ord(b) for b in "ACGT"]] = True
_DIAGONAL, _UP, _LEFT = 0, 1, 2


@dataclass(frozen=True, slots=True)
class Alignment:
    """How a query lines up with a reference."""

    identity: float
    """Matching share of the aligned, unambiguous positions (0-1)."""
    aligned: int
    """Aligned positions counted: matches, mismatches and internal gaps."""
    coverage: float
    """Share of the query's or the reference's bases inside the aligned
    region, whichever is larger (0-1): 1.0 means one lies wholly inside it."""


def _codes(sequence: str) -> np.ndarray:
    return np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)


def align(query: str, reference: str) -> Alignment:
    """Align ``query`` to ``reference`` and measure their identity."""
    q, r = _codes(query), _codes(reference)
    n, m = q.size, r.size
    if n == 0 or m == 0:
        return Alignment(identity=0.0, aligned=0, coverage=0.0)
    known_r = _BASES[r]
    steps = GAP * np.arange(m + 1)
    previous = np.zeros(m + 1, dtype=np.int64)
    moves = np.zeros((n + 1, m + 1), dtype=np.uint8)
    last_column = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, n + 1):
        same = r == q[i - 1]
        known = known_r & _BASES[q[i - 1]]
        score = np.where(known, np.where(same, MATCH, MISMATCH), 0)
        diagonal = previous[:-1] + score
        up = previous[1:] - GAP
        best = np.maximum(diagonal, up)
        # From the left: H[j] = max(best[j], H[j-1] - GAP), which is a
        # running maximum of (best[k] + GAP*k), shifted back by GAP*j
        current = np.empty(m + 1, dtype=np.int64)
        current[0] = 0  # free leading overhang of the query
        shifted = np.concatenate(([0], best)) + steps
        current[1:] = (np.maximum.accumulate(shifted) - steps)[1:]
        moves[i, 1:] = np.where(
            current[1:] == diagonal, _DIAGONAL, np.where(current[1:] == up, _UP, _LEFT)
        )
        last_column[i] = current[m]
        previous = current
    # Free trailing overhangs: end anywhere on the last row or column
    end_row, end_column = int(previous.argmax()), int(last_column.argmax())
    if previous[end_row] >= last_column[end_column]:
        i, j = n, end_row
    else:
        i, j = end_column, m
    query_end, reference_end = i, j
    matches = mismatches = gaps = 0
    while i > 0 and j > 0:
        move = moves[i, j]
        if move == _DIAGONAL:
            if _BASES[q[i - 1]] and _BASES[r[j - 1]]:
                if q[i - 1] == r[j - 1]:
                    matches += 1
                else:
                    mismatches += 1
            i, j = i - 1, j - 1
        elif move == _UP:
            gaps += 1
            i -= 1
        else:
            gaps += 1
            j -= 1
    aligned = matches + mismatches + gaps
    return Alignment(
        identity=matches / aligned if aligned else 0.0,
        aligned=aligned,
        coverage=max((query_end - i) / n, (reference_end - j) / m),
    )
