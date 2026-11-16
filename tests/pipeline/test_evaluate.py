"""Tests for prokopton.ml.bio.pipeline.evaluate."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from bio_app import CLASSIFIER, app_config, mlflow, sequence_app, write_config
from fakes import FakeTracker
from prokopton.ml.bio.features.kmers import KmerVectorizer
from prokopton.ml.bio.pipeline.evaluate import (
    REPORT_NAME,
    EvaluationStageError,
    evaluate,
)
from prokopton.ml.cli import main

MOTIFS = {"Diptera": "ACGTTGCA", "Hymenoptera": "GGGCCCAT"}
FOLDER = "midori2_gb999"


def record(
    record_id: str, split: str, order: str | None, motif: str, ambiguous: float = 0.0
) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "split": split,
        "group": "Metazoa",
        "order": order,
        "ambiguous_fraction": ambiguous,
        "sequence": motif * 10,
    }


def write_data(tmp_path: Path) -> None:
    rows = []
    for split, count in {"train": 3, "test_known": 6, "test_novel": 4}.items():
        for order, motif in MOTIFS.items():
            rows += [
                record(f"{split}-{order}-{i}", split, order, motif)
                for i in range(count)
            ]
    rows.append(record("known-no-order", "test_known", None, "ACGT"))
    rows.append(record("novel-rare", "test_novel", "Rare", "ACGT"))
    rows.append(
        record("known-ambiguous", "test_known", "Diptera", MOTIFS["Diptera"], 0.02)
    )
    split_path = tmp_path / "splits" / FOLDER / "gene=CO1" / "records.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split_path)
    leakage_path = (
        tmp_path / "leakage" / FOLDER / "gene=CO1" / "near_duplicates.parquet"
    )
    leakage_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"record_id": "test_novel-Diptera-0"}]), leakage_path
    )


def make_model() -> Pipeline:
    sequences = [motif * 10 for motif in MOTIFS.values()] * 3
    labels = list(MOTIFS) * 3
    model = Pipeline(
        [("kmers", KmerVectorizer(k=3)), ("classifier", MultinomialNB(fit_prior=False))]
    )
    return model.fit(sequences, labels)


def config_dict(tmp_path: Path) -> dict[str, Any]:
    router = {
        "kind": CLASSIFIER,
        "training": {"experiment": "test-router", "registered_model": "test-router"},
        "evaluation": {
            "fragment_lengths": [20, 1000],
            "fragment_fraction": 1.0,
            # 21 records are scored, so the last batch is empty
            "batch_size": 7,
            "threads": 2,
        },
    }
    return sequence_app(tmp_path, models={"router": router}, mlflow=mlflow(tmp_path))


def registered_tracker() -> FakeTracker:
    tracker = FakeTracker()
    tracker.log_model(make_model(), None, "test-router", "candidate")
    return tracker


def test_evaluate_scores_each_test_set(tmp_path: Path) -> None:
    write_data(tmp_path)
    tracker = registered_tracker()
    config = app_config(config_dict(tmp_path))

    report = evaluate(config, lambda _config: tracker)

    assert {name: s.records for name, s in report.splits.items()} == {
        "test_known": 13,
        "test_novel": 8,
        "test_novel_clean": 7,
        "test_novel_leaky": 1,
    }
    assert report.splits["test_known"].metrics["accuracy"] == 1.0
    assert report.unlabelled_records == {"test_known": 1}
    assert report.unseen_class_records == {"test_novel": 1}
    assert [b.records for b in report.splits["test_known"].by_ambiguity] == [
        12,
        0,
        1,
        0,
        0,
        0,
    ]
    known = {point.length: point for point in report.fragments["test_known"]}
    assert known[None].records == known[20].records == 13
    assert (known[1000].records, known[1000].too_short) == (0, 13)
    assert set(report.fragments) == {"test_known", "test_novel_clean"}
    assert set(report.per_class) == {"test_known", "test_novel_clean"}
    assert report.per_class["test_known"]["Diptera"].support == 7
    assert report.model_version == "1"

    run = tracker.runs[-1]
    assert run.name == "router-evaluation"
    assert run.tags["model_version"] == "1"
    assert run.tags["data_sha256.records"] == report.data_sha256["records"]
    assert run.metrics["test_known_len20_accuracy"] == 1.0
    assert run.metrics["test_novel_clean_full_macro_f1"] == 1.0
    assert run.documents["evaluation_report.json"]["model_version"] == "1"
    saved = json.loads((tmp_path / "models" / "router" / REPORT_NAME).read_text())
    assert saved["splits"]["test_novel_leaky"]["records"] == 1


def test_evaluate_requires_split_and_leakage_data(tmp_path: Path) -> None:
    config = app_config(config_dict(tmp_path))

    with pytest.raises(EvaluationStageError, match="run the split and leakage"):
        evaluate(config, lambda _config: registered_tracker())


def test_evaluate_without_novel_splits_needs_no_leakage_data(tmp_path: Path) -> None:
    write_data(tmp_path)
    for path in (tmp_path / "leakage").rglob("*.parquet"):
        path.unlink()
    settings = config_dict(tmp_path)
    settings["models"]["router"]["evaluation"] |= {
        "splits": ["test_known"],
        "novel_splits": [],
    }

    report = evaluate(app_config(settings), lambda _c: registered_tracker())

    assert set(report.splits) == {"test_known"}
    assert "near_duplicates" not in report.data_sha256


def write_app_toml(tmp_path: Path) -> Path:
    return write_config(tmp_path / "app.toml", config_dict(tmp_path))


def test_cli_evaluate_reports_missing_data(tmp_path: Path) -> None:
    assert (
        main(["--config", str(write_app_toml(tmp_path)), "run", "router-evaluate"]) == 1
    )


def test_cli_evaluate_loads_the_registered_model_from_mlflow(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    from prokopton.ml.training.tracking import MlflowTracker

    write_data(tmp_path)
    config = app_config(config_dict(tmp_path))
    tracker = MlflowTracker(
        config.mlflow.tracking_uri, config.mlflow.artifact_dir, "test-router"
    )
    with tracker.run("register"):
        tracker.log_model(make_model(), None, "test-router", "candidate")

    assert (
        main(["--config", str(write_app_toml(tmp_path)), "run", "router-evaluate"]) == 0
    )

    saved = json.loads((tmp_path / "models" / "router" / REPORT_NAME).read_text())
    assert saved["model_version"] == "1"
    assert saved["splits"]["test_known"]["metrics"]["accuracy"] == 1.0


def test_evaluate_marker_by_gene_group_and_strand(tmp_path: Path) -> None:
    write_data(tmp_path)
    cytb = [
        {
            "record_id": f"cytb-{i}",
            "split": "test_known",
            "group": "Fungi",
            "order": None,
            "ambiguous_fraction": 0.0,
            "sequence": "TTTAAACC" * 10,
        }
        for i in range(5)
    ]
    path = tmp_path / "splits" / FOLDER / "gene=Cytb" / "records.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(cytb), path)
    model = Pipeline(
        [("kmers", KmerVectorizer(k=3)), ("classifier", MultinomialNB(fit_prior=False))]
    ).fit(
        [m * 10 for m in MOTIFS.values()] + ["TTTAAACC" * 10] * 2,
        ["CO1", "CO1", "Cytb", "Cytb"],
    )
    tracker = FakeTracker()
    tracker.log_model(model, None, "test-marker", "candidate")
    settings = config_dict(tmp_path)
    settings["models"]["marker"] = {
        "kind": CLASSIFIER,
        "label": "gene",
        "genes": ["CO1", "Cytb"],
        "training": {"registered_model": "test-marker"},
        "evaluation": {
            "splits": ["test_known"],
            "novel_splits": [],
            "fragment_lengths": [20],
            "fragment_fraction": 1.0,
            "orientation": True,
            "threads": 2,
        },
    }

    report = evaluate(app_config(settings), lambda _c: tracker, model="marker")

    assert report.model == "marker"
    assert report.gene == "CO1,Cytb"
    assert set(report.data_sha256) == {"records.CO1", "records.Cytb"}
    known = report.splits["test_known"]
    assert known.records == 13 + 1 + 5  # every CO1 record now has a label
    assert {g.group: g.records for g in known.by_group} == {"Metazoa": 14, "Fungi": 5}
    whole, short = report.fragments["test_known"]
    assert whole.forward_strand_share is not None
    assert 0.0 <= whole.forward_strand_share <= 1.0
    assert short.forward_strand_share is not None
    assert (tmp_path / "models" / "marker" / REPORT_NAME).exists()


def test_strand_check_ignores_the_reverse_class(tmp_path: Path) -> None:
    write_data(tmp_path)
    sequences = [m * 10 for m in MOTIFS.values()]
    complement = {"A": "T", "C": "G", "G": "C", "T": "A"}
    reverse = ["".join(complement[b] for b in s[::-1]) for s in sequences]
    # CO1 and Cytb are trained on the same sequences, so on the right strand
    # the best gene has only about half the probability; on the other strand,
    # "reverse" is nearly certain. Comparing every class would pick the
    # wrong strand; comparing genes only picks the right one.
    model = Pipeline(
        [("kmers", KmerVectorizer(k=3)), ("classifier", MultinomialNB(fit_prior=False))]
    ).fit(
        sequences * 2 + reverse,
        ["CO1", "CO1", "Cytb", "Cytb", "reverse", "reverse"],
    )
    tracker = FakeTracker()
    tracker.log_model(model, None, "test-marker", "candidate")
    settings = config_dict(tmp_path)
    settings["models"]["marker"] = {
        "kind": CLASSIFIER,
        "label": "gene",
        "genes": ["CO1"],
        "training": {"registered_model": "test-marker", "reverse_label": "reverse"},
        "evaluation": {
            "splits": ["test_known"],
            "novel_splits": [],
            "fragment_lengths": [],
            "fragment_fraction": 1.0,
            "orientation": True,
            "threads": 1,
        },
    }

    report = evaluate(app_config(settings), lambda _c: tracker, model="marker")

    (whole,) = report.fragments["test_known"]
    # The motif records all pick the right strand (the 1 in 14 that misses
    # is the "ACGT" record, which is its own reverse complement)
    assert whole.forward_strand_share is not None
    assert whole.forward_strand_share >= 13 / 14
