"""Tests for prokopton.ml.training.tracking (MLflow itself is tested in
tests/pipeline/test_train.py)."""

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.training.calibration import LengthCalibratedClassifier
from prokopton.ml.training.tracking import own_types


def test_own_types_lists_only_prokopton_classes() -> None:
    pipeline = Pipeline([("kmers", KmerVectorizer(k=3)), ("nb", MultinomialNB())])

    assert own_types(pipeline) == ["prokopton.ml.features.kmers.KmerVectorizer"]


def test_own_types_of_a_single_model() -> None:
    assert own_types(MultinomialNB()) == []


def test_own_types_finds_classes_inside_a_wrapper() -> None:
    pipeline = Pipeline([("kmers", KmerVectorizer(k=3)), ("nb", MultinomialNB())])

    assert own_types(LengthCalibratedClassifier(pipeline)) == [
        "prokopton.ml.features.kmers.KmerVectorizer",
        "prokopton.ml.training.calibration.LengthCalibratedClassifier",
    ]


def test_own_types_of_an_object_without_parameters() -> None:
    assert own_types(object()) == []
