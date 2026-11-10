"""Tests for prokopton.ml.training.router."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB

from fakes import FakeTracker
from prokopton.ml.config import RouterTrainingConfig
from prokopton.ml.training import router
from prokopton.ml.training.router import (
    Candidate,
    RouterSample,
    candidates,
    class_feature_counts,
    naive_bayes_from_counts,
    predict_proba,
    train_router,
)

# Each class is built from its own repeated motif, so k-mers separate them
MOTIFS = {"Diptera": "ACGTTGCA", "Fungi": "GGGCCCAT", "Lepidoptera": "TTAACCGG"}


def make_sample(per_class: int, seed: int, ambiguous: float = 0.0) -> RouterSample:
    rng = np.random.default_rng(seed)
    sequences: list[str] = []
    labels: list[str] = []
    for label, motif in MOTIFS.items():
        for _ in range(per_class):
            bases = list(motif * 12)
            for position in rng.choice(len(bases), size=6, replace=False):
                bases[position] = str(rng.choice(list("ACGT")))
            sequences.append("".join(bases))
            labels.append(label)
    fractions = np.full(len(labels), ambiguous)
    return RouterSample(sequences, labels, fractions)


def small_rules(**fields: object) -> RouterTrainingConfig:
    settings: dict[str, object] = {
        "k_values": (3, 4),
        "ambiguity_thresholds": (0.01, 0.05),
        "naive_bayes_alphas": (0.1,),
        "logistic_k_values": (4,),
        "logistic_alphas": (1e-4,),
        "logistic_max_iter": 20,
        "threads": 1,
        "registered_model": "test-router",
    }
    settings.update(fields)
    return RouterTrainingConfig.model_validate(settings)


def test_candidates_cover_the_grid_in_k_order() -> None:
    found = candidates(small_rules())

    assert len(found) == 2 * 2 * 1 + 1
    assert [c.k for c in found] == sorted(c.k for c in found)
    assert Candidate("logistic", 4, 0.03, 1e-4) in found


def test_candidate_name() -> None:
    assert (
        Candidate("naive_bayes", 6, 0.03, 0.1).name == "naive_bayes-k6-amb0.03-alpha0.1"
    )


def test_train_router_tracks_every_candidate_and_registers_the_best() -> None:
    samples = {
        "train": make_sample(30, seed=1),
        "val_known": make_sample(10, seed=2),
        "val_novel": make_sample(10, seed=3),
    }
    tracker = FakeTracker()

    result = train_router(samples, small_rules(), tracker, {"release": "GB999"})

    parent, *children = tracker.runs
    assert parent.name == "router-search" and not parent.nested
    assert len(children) == result.candidates_tried == 5
    assert all(child.nested for child in children)
    assert all(child.tags["release"] == "GB999" for child in children)
    best_child = max(children, key=lambda child: child.metrics["selection_score"])
    assert result.metrics == best_child.metrics
    assert result.metrics["val_known_accuracy"] > 0.9
    assert parent.params["best_k"] == result.best.k
    assert parent.documents["classes.json"]["classes"] == sorted(MOTIFS)

    model, example, name, alias = tracker.models[0]
    assert (name, alias, result.model_version) == ("test-router", "candidate", "1")
    assert list(example.columns) == ["sequence"]
    # The saved pipeline turns raw sequences (in a DataFrame) into labels
    query = pd.DataFrame({"sequence": [MOTIFS["Fungi"] * 12]})
    assert model.predict(query).tolist() == ["Fungi"]


def test_ambiguity_threshold_filters_training_records() -> None:
    train = make_sample(30, seed=1, ambiguous=0.02)
    samples = {
        "train": train,
        "val_known": make_sample(5, seed=2),
        "val_novel": make_sample(5, seed=3),
    }
    tracker = FakeTracker()
    rules = small_rules(
        k_values=(3,), ambiguity_thresholds=(0.01, 0.05), logistic_k_values=()
    )
    # At 0.01 every record is too ambiguous; give a few a lower fraction
    train.ambiguous_fraction[::3] = 0.0

    train_router(samples, rules, tracker, {})

    records = {
        run.params["max_ambiguous_fraction"]: run.metrics["train_records"]
        for run in tracker.runs[1:]
    }
    assert records == {0.01: 30.0, 0.05: 90.0}


def test_train_router_requires_every_sample() -> None:
    with pytest.raises(ValueError, match="val_novel"):
        train_router(
            {"train": make_sample(2, 1), "val_known": make_sample(2, 2)},
            small_rules(),
            FakeTracker(),
            {},
        )


def test_naive_bayes_from_counts_matches_fitting() -> None:
    rng = np.random.default_rng(5)
    features = sparse.csr_matrix(rng.poisson(1.0, size=(60, 16)).astype(np.float32))
    labels = np.array(["a", "b", "c"] * 20, dtype=object)
    keep = rng.random(60) < 0.8

    made = naive_bayes_from_counts(*class_feature_counts(features, labels, keep), 0.5)
    fitted = MultinomialNB(alpha=0.5, fit_prior=False).fit(features[keep], labels[keep])

    assert made.classes_.tolist() == fitted.classes_.tolist()
    np.testing.assert_allclose(made.class_count_, fitted.class_count_)
    np.testing.assert_allclose(made.feature_log_prob_, fitted.feature_log_prob_)
    np.testing.assert_allclose(made.class_log_prior_, fitted.class_log_prior_)
    np.testing.assert_allclose(
        made.predict_proba(features), fitted.predict_proba(features)
    )


def test_predict_proba_in_chunks_matches_all_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(6)
    features = sparse.csr_matrix(rng.poisson(1.0, size=(25, 8)).astype(np.float32))
    model = MultinomialNB().fit(features, np.arange(25) % 3)
    monkeypatch.setattr(router, "PREDICT_CHUNK", 4)

    chunked = predict_proba(model, features, threads=3)

    np.testing.assert_allclose(chunked, model.predict_proba(features))
