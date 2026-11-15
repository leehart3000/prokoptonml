"""k-mer frequency features for biological sequences.

A k-mer is a "word" of ``k`` consecutive letters. Sliding a window along a
sequence and counting each word turns any sequence into a fixed-length vector
of ``len(alphabet) ** k`` numbers. Counts are divided by the number of k-mers
counted, so short fragments and full-length sequences are comparable, and
k-mers containing letters outside the alphabet (such as the ambiguity code
``N``) are skipped.

``KmerVectorizer`` is a scikit-learn transformer, so it can be the first step
of a model pipeline. The same code then computes features during training and
in the deployed service, which avoids training/serving skew.
"""

from collections.abc import Iterable
from itertools import product

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["AMINO_ACIDS", "DNA", "KmerVectorizer", "reverse_complement"]

DNA = "ACGT"
"""The four DNA bases."""

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
"""The 20 standard amino acids, for peptide sequences."""

_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVN", "TGCAYRSWMKVHDBN")


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence (IUPAC codes included)."""
    return sequence.upper().translate(_COMPLEMENT)[::-1]


class KmerVectorizer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Convert sequences into sparse k-mer frequency vectors.

    Args:
        k: Length of each k-mer.
        alphabet: The letters to count. Each k-mer is numbered in base
            ``len(alphabet)``, so feature ``i`` is the ``i``-th k-mer in
            dictionary order of the alphabet as given.
    """

    def __init__(self, k: int = 6, alphabet: str = DNA) -> None:
        self.k = k
        self.alphabet = alphabet

    def fit(
        self, sequences: Iterable[str] | None = None, y: object = None
    ) -> "KmerVectorizer":
        """Validate the settings. No data is needed: the features are fixed."""
        if self.k < 1:
            msg = f"k must be at least 1, got {self.k}"
            raise ValueError(msg)
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            msg = (
                "alphabet must be non-empty with no repeated letters: "
                f"{self.alphabet!r}"
            )
            raise ValueError(msg)
        if not self.alphabet.isascii():
            msg = "alphabet must contain ASCII letters only"
            raise ValueError(msg)
        lookup = np.full(256, -1, dtype=np.int64)
        for code, letter in enumerate(self.alphabet.upper()):
            lookup[ord(letter)] = code
            lookup[ord(letter.lower())] = code
        self.lookup_ = lookup
        self.n_features_out_ = len(self.alphabet) ** self.k
        return self

    def _kmer_indices(self, sequence: str) -> np.ndarray:
        """Return the index of every valid k-mer in ``sequence``."""
        raw = np.frombuffer(sequence.encode("ascii", errors="replace"), dtype=np.uint8)
        codes = self.lookup_[raw]
        windows = len(codes) - self.k + 1
        if windows < 1:
            return np.empty(0, dtype=np.int64)
        invalid = np.concatenate(([0], np.cumsum(codes < 0)))
        valid_window = (invalid[self.k :] - invalid[:windows]) == 0
        safe = np.where(codes < 0, 0, codes)
        base = len(self.alphabet)
        index = np.zeros(windows, dtype=np.int64)
        for offset in range(self.k):
            index = index * base + safe[offset : offset + windows]
        valid: np.ndarray = index[valid_window]
        return valid

    def transform(self, sequences: Iterable[str]) -> sparse.csr_matrix:
        """Return a sparse matrix of k-mer frequencies, one row per sequence.

        ``sequences`` may be any iterable of strings, or a table (such as a
        pandas DataFrame, which MLflow passes when serving), in which case its
        first column is used. A sequence with no valid k-mers (too short, or
        all ambiguous) gets a row of zeros.
        """
        check_is_fitted(self, "lookup_")
        columns = getattr(sequences, "columns", None)
        if columns is not None:
            sequences = sequences[columns[0]]  # type: ignore[index]
        indptr = [0]
        indices: list[np.ndarray] = []
        data: list[np.ndarray] = []
        for sequence in sequences:
            kmers = self._kmer_indices(sequence)
            if kmers.size:
                unique, counts = np.unique(kmers, return_counts=True)
                indices.append(unique)
                data.append((counts / kmers.size).astype(np.float32))
                indptr.append(indptr[-1] + unique.size)
            else:
                indptr.append(indptr[-1])
        # 32-bit indices halve memory; fine below 2**31 features and values
        index_type = (
            np.int32
            if self.n_features_out_ < 2**31 and indptr[-1] < 2**31
            else np.int64
        )
        return sparse.csr_matrix(
            (
                np.concatenate(data) if data else np.empty(0, dtype=np.float32),
                (np.concatenate(indices) if indices else np.empty(0)).astype(
                    index_type
                ),
                np.array(indptr, dtype=index_type),
            ),
            shape=(len(indptr) - 1, self.n_features_out_),
        )

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        """Return the k-mer each feature column counts, e.g. ``"ACGTAC"``."""
        check_is_fitted(self, "lookup_")
        return np.array(
            ["".join(letters) for letters in product(self.alphabet, repeat=self.k)],
            dtype=object,
        )
