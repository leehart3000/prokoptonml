"""Tests for prokopton.ml.features.kmers."""

import random
import time

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline

from prokopton.ml.features.kmers import (
    AMINO_ACIDS,
    KmerVectorizer,
    reverse_complement,
)


def frequencies(vectorizer: KmerVectorizer, sequence: str) -> dict[str, float]:
    """Return the non-zero k-mer frequencies of one sequence, by k-mer."""
    names = vectorizer.get_feature_names_out()
    row = vectorizer.transform([sequence]).toarray()[0]
    return {str(names[i]): float(row[i]) for i in np.flatnonzero(row)}


def test_counts_kmers_as_frequencies() -> None:
    vectorizer = KmerVectorizer(k=3).fit()

    assert frequencies(vectorizer, "ACGTAC") == {
        "ACG": 0.25,
        "CGT": 0.25,
        "GTA": 0.25,
        "TAC": 0.25,
    }


def test_repeated_kmers_are_counted() -> None:
    vectorizer = KmerVectorizer(k=2).fit()

    assert frequencies(vectorizer, "AAAA") == {"AA": 1.0}


def test_feature_order_follows_the_alphabet() -> None:
    names = KmerVectorizer(k=2).fit().get_feature_names_out()

    assert list(names[:5]) == ["AA", "AC", "AG", "AT", "CA"]
    assert len(names) == 16


def test_kmers_with_ambiguous_letters_are_skipped() -> None:
    vectorizer = KmerVectorizer(k=3).fit()

    assert frequencies(vectorizer, "ACGNTAC") == {"ACG": 0.5, "TAC": 0.5}


def test_lower_case_and_non_ascii_input() -> None:
    vectorizer = KmerVectorizer(k=2).fit()

    assert frequencies(vectorizer, "acgt") == frequencies(vectorizer, "ACGT")
    assert frequencies(vectorizer, "AéCG") == {"CG": 1.0}


@pytest.mark.parametrize("sequence", ["", "AC", "NNNNNN"])
def test_sequences_without_valid_kmers_give_zero_rows(sequence: str) -> None:
    matrix = KmerVectorizer(k=3).fit().transform(["ACGT", sequence])

    assert matrix.shape == (2, 64)
    assert matrix[1].nnz == 0
    assert matrix[0].sum() == pytest.approx(1.0)


def test_empty_input_gives_empty_matrix() -> None:
    matrix = KmerVectorizer(k=3).fit().transform([])

    assert matrix.shape == (0, 64)


def test_rows_sum_to_one_and_are_float32() -> None:
    matrix = KmerVectorizer(k=6).fit().transform(["ACGT" * 50, "TTGACC" * 30])

    assert matrix.dtype == np.float32
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_peptide_alphabet() -> None:
    vectorizer = KmerVectorizer(k=2, alphabet=AMINO_ACIDS).fit()

    assert vectorizer.n_features_out_ == 400
    assert frequencies(vectorizer, "MKVL") == pytest.approx(
        {"MK": 1 / 3, "KV": 1 / 3, "VL": 1 / 3}
    )


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"k": 0}, "k must be at least 1"),
        ({"alphabet": ""}, "non-empty"),
        ({"alphabet": "AACG"}, "no repeated letters"),
        ({"alphabet": "ACGé"}, "ASCII"),
    ],
)
def test_invalid_settings_are_rejected(
    settings: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        KmerVectorizer(**settings).fit()  # type: ignore[arg-type]


def test_works_in_a_scikit_learn_pipeline() -> None:
    vectorizer = KmerVectorizer(k=4)
    pipeline = make_pipeline(vectorizer)

    matrix = pipeline.fit_transform(["ACGTACGT", "TTTTGGGG"])

    assert matrix.shape == (2, 256)
    assert clone(vectorizer).get_params() == {"k": 4, "alphabet": "ACGT"}


def test_accepts_a_dataframe_and_uses_its_first_column() -> None:
    import pandas as pd

    vectorizer = KmerVectorizer(k=2).fit()
    frame = pd.DataFrame({"sequence": ["ACGT", "AAAA"]})

    matrix = vectorizer.transform(frame)

    assert matrix.shape == (2, 16)
    assert (matrix != vectorizer.transform(["ACGT", "AAAA"])).nnz == 0


def test_uses_32_bit_indices() -> None:
    matrix = KmerVectorizer(k=6).fit().transform(["ACGT" * 30])

    assert matrix.indices.dtype == np.int32
    assert matrix.indptr.dtype == np.int32


def test_reverse_complement() -> None:
    assert reverse_complement("AACGTN") == "NACGTT"
    assert reverse_complement("acgR") == "YCGT"


def test_is_fast_enough_for_training() -> None:
    rng = random.Random(0)
    sequences = ["".join(rng.choice("ACGT") for _ in range(650)) for _ in range(10_000)]
    vectorizer = KmerVectorizer(k=6).fit()

    started = time.perf_counter()
    matrix = vectorizer.transform(sequences)
    elapsed = time.perf_counter() - started

    assert matrix.shape == (10_000, 4096)
    assert elapsed < 10.0
