"""Train and choose the router: the model that predicts group and order.

Each candidate is a model type (naive Bayes or logistic regression) with its
settings: k-mer size ``k``, the ambiguity threshold applied to the training
sample, and smoothing (naive Bayes ``alpha``) or regularisation (logistic
regression ``alpha``). Candidates are compared on the validation samples
only; the test sets are kept for the final evaluation.

Features for each ``k`` are computed once and shared by every candidate with
that ``k``. Naive Bayes only needs each class's total k-mer counts, so those
are added up once per ``k`` and threshold, and each ``alpha`` is applied to
them directly (much faster than refitting). Predictions are made in chunks
on several threads. Each candidate is tracked as a nested run; the best is saved as a
scikit-learn pipeline (``KmerVectorizer`` then the classifier), registered,
and given the configured alias.
"""

import logging
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby, product
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from prokopton.ml.config import RouterTrainingConfig
from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.training.metrics import classification_metrics
from prokopton.ml.training.tracking import Tracker

__all__ = [
    "Candidate",
    "RouterSample",
    "TrainingResult",
    "candidates",
    "class_feature_counts",
    "naive_bayes_from_counts",
    "predict_proba",
    "train_router",
]

logger = logging.getLogger(__name__)

PREDICT_CHUNK = 2_000
"""Rows per prediction chunk; chunks are spread across ``threads``."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model type with one set of settings."""

    model: str
    k: int
    max_ambiguous_fraction: float
    alpha: float

    @property
    def name(self) -> str:
        return (
            f"{self.model}-k{self.k}-amb{self.max_ambiguous_fraction:g}"
            f"-alpha{self.alpha:g}"
        )


@dataclass(frozen=True, slots=True)
class RouterSample:
    """Sequences, labels and ambiguity of one sample split."""

    sequences: list[str]
    labels: list[str]
    ambiguous_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """The chosen candidate, its validation metrics and its registered version."""

    best: Candidate
    metrics: dict[str, float]
    candidates_tried: int
    parent_run_id: str
    model_version: str


def candidates(rules: RouterTrainingConfig) -> list[Candidate]:
    """List every candidate in the configured search, ordered by ``k``."""
    found = [
        Candidate("naive_bayes", k, threshold, alpha)
        for k, threshold, alpha in product(
            rules.k_values, rules.ambiguity_thresholds, rules.naive_bayes_alphas
        )
    ]
    found += [
        Candidate("logistic", k, rules.logistic_ambiguity_threshold, alpha)
        for k, alpha in product(rules.logistic_k_values, rules.logistic_alphas)
    ]
    # Candidates sharing k, model and threshold are next to each other, so
    # features and per-threshold work are reused
    return sorted(
        found,
        key=lambda c: (c.k, c.model, c.max_ambiguous_fraction, c.alpha),
    )


def class_feature_counts(
    features: sparse.csr_matrix, labels: np.ndarray, keep: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add up the features of each class, using only the kept rows.

    Returns:
        The classes (sorted), each class's number of records, and a
        ``(classes, features)`` array of summed features.
    """
    classes, codes = np.unique(labels[keep], return_inverse=True)
    rows = np.flatnonzero(keep)
    indicator = sparse.csr_matrix(
        (np.ones(rows.size), (codes, rows)), shape=(classes.size, labels.size)
    )
    counts = np.asarray((indicator @ features).todense(), dtype=np.float64)
    return (
        classes,
        np.bincount(codes, minlength=classes.size).astype(np.float64),
        counts,
    )


def naive_bayes_from_counts(
    classes: np.ndarray,
    class_count: np.ndarray,
    feature_count: np.ndarray,
    alpha: float,
) -> MultinomialNB:
    """Make a fitted naive Bayes model from per-class feature totals.

    Gives the same model as ``MultinomialNB(alpha, fit_prior=False).fit``, but
    the totals can be reused for every ``alpha``. Class priors are equal, so
    unbalanced class sizes do not bias predictions.
    """
    model = MultinomialNB(alpha=alpha, fit_prior=False)
    smoothed = feature_count + alpha
    model.classes_ = classes
    model.class_count_ = class_count
    model.feature_count_ = feature_count
    model.feature_log_prob_ = np.log(smoothed) - np.log(
        smoothed.sum(axis=1, keepdims=True)
    )
    model.class_log_prior_ = np.full(classes.size, -np.log(classes.size))
    model.n_features_in_ = feature_count.shape[1]
    return model


def _logistic(candidate: Candidate, rules: RouterTrainingConfig) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        alpha=candidate.alpha,
        max_iter=rules.logistic_max_iter,
        tol=1e-3,
        n_jobs=rules.threads,
        random_state=rules.random_state,
    )


def predict_proba(
    classifier: Any, features: sparse.csr_matrix, threads: int
) -> np.ndarray:
    """Predict class probabilities in chunks spread across threads."""
    chunks = [
        features[start : start + PREDICT_CHUNK]
        for start in range(0, features.shape[0], PREDICT_CHUNK)
    ]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        parts = list(pool.map(classifier.predict_proba, chunks))
    stacked: np.ndarray = np.vstack(parts)
    return stacked


def _evaluate(
    classifier: Any,
    features: dict[str, sparse.csr_matrix],
    samples: dict[str, RouterSample],
    threads: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    classes = list(classifier.classes_)
    for split in ("val_known", "val_novel"):
        probabilities = predict_proba(classifier, features[split], threads)
        found = classification_metrics(samples[split].labels, probabilities, classes)
        metrics.update({f"{split}_{name}": value for name, value in found.items()})
    metrics["selection_score"] = (
        metrics["val_known_macro_f1"] + metrics["val_novel_macro_f1"]
    ) / 2
    return metrics


def _fitted(
    group: list[Candidate],
    features: sparse.csr_matrix,
    labels: np.ndarray,
    keep: np.ndarray,
    rules: RouterTrainingConfig,
) -> Iterator[tuple[Candidate, Any, float]]:
    """Fit each candidate in a group sharing k, model and threshold.

    Yields each candidate, its fitted classifier and the seconds it took.
    """
    if group[0].model == "naive_bayes":
        started = time.monotonic()
        classes, class_count, feature_count = class_feature_counts(
            features, labels, keep
        )
        shared = time.monotonic() - started
        for candidate in group:
            started = time.monotonic()
            model = naive_bayes_from_counts(
                classes, class_count, feature_count, candidate.alpha
            )
            yield candidate, model, shared + time.monotonic() - started
        return
    kept = features if keep.all() else features[keep]
    for candidate in group:
        started = time.monotonic()
        classifier = _logistic(candidate, rules)
        classifier.fit(kept, labels[keep])
        yield candidate, classifier, time.monotonic() - started


def train_router(
    samples: dict[str, RouterSample],
    rules: RouterTrainingConfig,
    tracker: Tracker,
    tags: dict[str, str],
) -> TrainingResult:
    """Try every candidate, then save and register the best.

    Args:
        samples: The ``train``, ``val_known`` and ``val_novel`` samples.
        rules: The search and naming settings.
        tracker: Where runs, metrics and the model are recorded.
        tags: Descriptive labels (e.g. data lineage) for every run.

    Raises:
        ValueError: If a sample is missing.
    """
    missing = {"train", "val_known", "val_novel"} - samples.keys()
    if missing:
        msg = f"Missing samples: {sorted(missing)}"
        raise ValueError(msg)
    search = candidates(rules)
    train = samples["train"]
    labels = np.asarray(train.labels, dtype=object)
    best: tuple[Candidate, Any, KmerVectorizer, dict[str, float]] | None = None

    with tracker.run("router-search") as parent_run_id:
        tracker.set_tags({**tags, "stage": "train", "model_role": "router"})
        tracker.log_params({"candidates": len(search), "train_records": labels.size})
        for k, k_group in groupby(search, key=lambda c: c.k):
            started = time.monotonic()
            vectorizer = KmerVectorizer(k=k).fit()
            features = {
                name: vectorizer.transform(s.sequences) for name, s in samples.items()
            }
            logger.info(
                "Computed k=%d features in %.0f s", k, time.monotonic() - started
            )
            for (_model, threshold), group in groupby(
                k_group, key=lambda c: (c.model, c.max_ambiguous_fraction)
            ):
                keep = train.ambiguous_fraction <= threshold
                for candidate, classifier, fit_seconds in _fitted(
                    list(group), features["train"], labels, keep, rules
                ):
                    with tracker.run(candidate.name, nested=True):
                        metrics = _evaluate(
                            classifier, features, samples, rules.threads
                        )
                        metrics["fit_seconds"] = fit_seconds
                        metrics["train_records"] = float(keep.sum())
                        tracker.set_tags({**tags, "model_type": candidate.model})
                        tracker.log_params(
                            {
                                "model": candidate.model,
                                "k": candidate.k,
                                "max_ambiguous_fraction": threshold,
                                "alpha": candidate.alpha,
                            }
                        )
                        tracker.log_metrics(metrics)
                    logger.info(
                        "%s: selection %.4f (known macro-F1 %.4f, novel macro-F1 %.4f)",
                        candidate.name,
                        metrics["selection_score"],
                        metrics["val_known_macro_f1"],
                        metrics["val_novel_macro_f1"],
                    )
                    if (
                        best is None
                        or metrics["selection_score"] > best[3]["selection_score"]
                    ):
                        best = (candidate, classifier, vectorizer, metrics)
            del features

        if best is None:  # pragma: no cover - candidates() is never empty
            msg = "No candidates to train"
            raise ValueError(msg)
        chosen, classifier, vectorizer, metrics = best
        pipeline = Pipeline([("kmers", vectorizer), ("classifier", classifier)])
        tracker.log_params(
            {
                "best_model": chosen.model,
                "best_k": chosen.k,
                "best_max_ambiguous_fraction": chosen.max_ambiguous_fraction,
                "best_alpha": chosen.alpha,
            }
        )
        tracker.log_metrics({f"best_{name}": value for name, value in metrics.items()})
        tracker.log_dict(
            {"classes": [str(c) for c in classifier.classes_]}, "classes.json"
        )
        example = pd.DataFrame({"sequence": train.sequences[:3]})
        version = tracker.log_model(
            pipeline, example, rules.registered_model, rules.alias
        )

    return TrainingResult(
        best=chosen,
        metrics=metrics,
        candidates_tried=len(search),
        parent_run_id=parent_run_id,
        model_version=version,
    )
