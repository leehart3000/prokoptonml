"""Tests for prokopton.ml.specialists.identity."""

import time

import numpy as np
import pytest

from prokopton.ml.specialists.identity import align

rng = np.random.default_rng(1)
SEQUENCE = "".join(rng.choice(list("ACGT"), 600))


def test_identical_sequences() -> None:
    found = align(SEQUENCE, SEQUENCE)

    assert found.identity == 1.0
    assert found.aligned == 600
    assert found.coverage == 1.0


def test_mismatches_lower_identity() -> None:
    changed = list(SEQUENCE)
    for position in (10, 200, 400):
        changed[position] = "A" if changed[position] != "A" else "C"

    found = align("".join(changed), SEQUENCE)

    assert found.identity == pytest.approx(597 / 600)


def test_an_insertion_is_a_gap_not_many_mismatches() -> None:
    inserted = SEQUENCE[:300] + "TTT" + SEQUENCE[300:]

    found = align(inserted, SEQUENCE)

    assert found.aligned == 603
    assert found.identity == pytest.approx(600 / 603)


def test_a_fragment_aligns_inside_a_longer_reference() -> None:
    found = align(SEQUENCE[150:350], SEQUENCE)

    assert found.identity == 1.0
    assert found.aligned == 200
    assert found.coverage == 1.0


def test_overhanging_ends_cost_nothing() -> None:
    # The query starts before and ends after the reference
    found = align("GGGGG" + SEQUENCE[:300] + "CCCCC", SEQUENCE[:300])

    assert found.identity == 1.0
    assert found.aligned == 300


def test_a_long_query_covers_a_shorter_reference() -> None:
    # A whole CO1 gene against a barcode-length reference: the reference
    # lies wholly inside the alignment, so coverage is 1.0
    longer = "".join(rng.choice(list("ACGT"), 400)) + SEQUENCE + "ACGT" * 100

    found = align(longer, SEQUENCE)

    assert (found.identity, found.aligned, found.coverage) == (1.0, 600, 1.0)


def test_partial_overlap_is_not_full_coverage() -> None:
    # Only the last 200 bases of the query overlap the first 200 of the
    # reference
    overlap = align(SEQUENCE[:400], SEQUENCE[200:])

    assert overlap.aligned == 200
    assert overlap.coverage == pytest.approx(200 / 400)


def test_ambiguous_bases_are_left_out() -> None:
    masked = SEQUENCE[:100] + "N" * 10 + SEQUENCE[110:]

    found = align(masked, SEQUENCE)

    assert found.identity == 1.0
    assert found.aligned == 590


def test_lower_case() -> None:
    assert align(SEQUENCE.lower(), SEQUENCE).identity == 1.0


def test_unrelated_sequences_barely_overlap() -> None:
    # Free end gaps let unrelated sequences "align" over a few bases, so
    # identity must always be read together with coverage
    unrelated = "".join(rng.choice(list("ACGT"), 600))

    found = align(unrelated, SEQUENCE)

    assert found.coverage < 0.1


def test_empty_sequences() -> None:
    assert align("", SEQUENCE).identity == 0.0
    assert align(SEQUENCE, "").coverage == 0.0


def test_fast_enough_for_twenty_candidates_per_query() -> None:
    started = time.perf_counter()
    for _ in range(20):
        align(SEQUENCE, SEQUENCE[::-1])
    assert time.perf_counter() - started < 5.0
