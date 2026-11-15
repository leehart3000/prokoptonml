"""Models saved before 0.2.0 refer to ``prokopton.ml.features.kmers``."""

import pickle

from prokopton.ml.bio.features.kmers import KmerVectorizer
from prokopton.ml.features import kmers as old


def test_the_old_path_is_the_same_class() -> None:
    assert old.KmerVectorizer is KmerVectorizer


def test_a_vectoriser_saved_under_the_old_path_still_loads() -> None:
    saved = pickle.dumps(KmerVectorizer(k=4), protocol=0)
    new_path = b"prokopton.ml.bio.features.kmers"
    assert new_path in saved
    as_before = saved.replace(new_path, b"prokopton.ml.features.kmers")

    loaded = pickle.loads(as_before)  # made by this test

    assert isinstance(loaded, KmerVectorizer)
    assert loaded.k == 4
