"""Classifiers saved before 0.2.0 refer to ``prokopton.ml.features.kmers``,
which is now only a path that leads to the new module while they load."""

import pickle
import sys
from pathlib import Path
from typing import Any

import pytest

from bio_app import app_config, sequence_app
from fakes import FakeTracker
from prokopton.ml.bio import models
from prokopton.ml.bio.features.kmers import KmerVectorizer
from prokopton.ml.inference.identify import model_kinds

OLD = "prokopton.ml.features.kmers"


def saved_before_0_2_0() -> bytes:
    """A vectoriser pickled as models saved before 0.2.0 were."""
    saved = pickle.dumps(KmerVectorizer(k=4), protocol=0)
    new_path = b"prokopton.ml.bio.features.kmers"
    assert new_path in saved
    return saved.replace(new_path, OLD.encode())


def test_the_old_path_is_gone() -> None:
    assert OLD not in sys.modules
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(saved_before_0_2_0())  # made by this test


def test_the_old_path_leads_to_the_new_module_while_models_load() -> None:
    with models.old_paths():
        loaded = pickle.loads(saved_before_0_2_0())  # made by this test

    assert isinstance(loaded, KmerVectorizer)
    assert loaded.k == 4
    assert OLD not in sys.modules


class OldRegistry(FakeTracker):
    """A registry holding a classifier saved before 0.2.0."""

    def load_model(self, registered_name: str, alias: str) -> tuple[Any, str]:
        return pickle.loads(saved_before_0_2_0()), "3"  # made by this test


def test_classifiers_saved_before_0_2_0_load_from_the_registry(
    tmp_path: Path,
) -> None:
    config = app_config(sequence_app(tmp_path))
    kind = model_kinds()["bio.kmer-classifier"]

    model, version = kind.load_current(config, "router", lambda _c, _n: OldRegistry())

    assert isinstance(model, KmerVectorizer)
    assert version == "3"


def test_bundles_refer_to_the_new_path(tmp_path: Path) -> None:
    config = app_config(sequence_app(tmp_path))
    kind = model_kinds()["bio.kmer-classifier"]

    kind.bundle(config, "router", tmp_path, lambda _c, _n: OldRegistry())

    saved = (tmp_path / "router.joblib").read_bytes()
    assert b"prokopton.ml.bio.features.kmers" in saved
    assert OLD.encode() not in saved
