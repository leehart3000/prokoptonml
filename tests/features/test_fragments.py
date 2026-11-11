"""Tests for prokopton.ml.features.fragments."""

import numpy as np

from prokopton.ml.features.fragments import fragment, random_fragments


def test_fragment_cuts_a_piece_of_the_right_length() -> None:
    rng = np.random.default_rng(1)

    piece = fragment("ACGTACGTAC", 4, rng)

    assert piece is not None
    assert len(piece) == 4
    assert piece in "ACGTACGTAC"
    assert fragment("ACG", 4, rng) is None
    assert fragment("ACGT", 4, rng) == "ACGT"


def test_random_fragments_have_lengths_in_range() -> None:
    rng = np.random.default_rng(2)
    sequences = ["ACGT" * 50] * 200

    pieces = random_fragments(sequences, 20, 40, rng)

    lengths = {len(piece) for piece in pieces}
    assert min(lengths) == 20
    assert max(lengths) == 40
    assert all(piece in sequences[0] for piece in pieces)


def test_random_fragments_keep_short_sequences_whole() -> None:
    rng = np.random.default_rng(3)

    assert random_fragments(["ACGTA", "AC"], 10, 20, rng) == ["ACGTA", "AC"]


def test_random_fragments_are_repeatable_with_a_seed() -> None:
    sequences = ["ACGTTGCA" * 20] * 10

    first = random_fragments(sequences, 5, 50, np.random.default_rng(4))
    second = random_fragments(sequences, 5, 50, np.random.default_rng(4))

    assert first == second
