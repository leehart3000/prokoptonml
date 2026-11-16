"""Tests that the platform's own kinds are found through its entry points,
like any other package's, and that the classes inside saved models stay
where the saved models expect them."""

import importlib
from importlib.metadata import entry_points

import pytest

from prokopton.ml.bio.models import bio_model_kinds
from prokopton.ml.bio.serving import bio_serving_kinds
from prokopton.ml.bio.stages import bio_kinds
from prokopton.ml.cli import registry
from prokopton.ml.inference.identify import model_kinds, serving_kinds

DECLARED = {
    "prokopton.ml.stages": ("prokopton.ml.bio.stages:bio_kinds", bio_kinds),
    "prokopton.ml.serving_steps": (
        "prokopton.ml.bio.serving:bio_serving_kinds",
        bio_serving_kinds,
    ),
    "prokopton.ml.model_kinds": (
        "prokopton.ml.bio.models:bio_model_kinds",
        bio_model_kinds,
    ),
}


@pytest.mark.parametrize("group", sorted(DECLARED))
def test_the_bio_kinds_are_declared_as_entry_points(group: str) -> None:
    value, function = DECLARED[group]
    found = {point.name: point for point in entry_points(group=group)}

    assert found["bio"].value == value
    assert found["bio"].load() is function


def test_every_kind_is_found_through_the_entry_points() -> None:
    stages = registry()
    steps = serving_kinds()
    models = model_kinds()

    assert stages["core.bundle"].kind == "core.bundle"
    assert stages["bio.classifier-train"].kind == "bio.classifier-train"
    assert steps["bio.sequence-input"].kind == "bio.sequence-input"
    assert models["bio.specialist"].kind == "bio.specialist"


SAVED_TYPES = (
    "prokopton.ml.bio.features.kmers.KmerVectorizer",
    "prokopton.ml.training.calibration.LengthCalibratedClassifier",
    # The old path, used by models saved before 0.2.0 (see design 0004, D3)
    "prokopton.ml.features.kmers.KmerVectorizer",
)
"""Classes inside saved models (in the MLflow registry and in bundles), by the
import path the saved files refer to. Moving one of them stops saved models
from loading, unless the old path still leads to it."""


@pytest.mark.parametrize("path", SAVED_TYPES)
def test_classes_inside_saved_models_stay_where_they_are(path: str) -> None:
    module, _, name = path.rpartition(".")

    found = getattr(importlib.import_module(module), name)

    assert isinstance(found, type)
    assert path.endswith(f".{found.__qualname__}")
