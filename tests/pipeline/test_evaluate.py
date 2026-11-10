"""Tests for prokopton.ml.pipeline.evaluate."""

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.pipeline.evaluate import (
    REPORT_NAME,
    EvaluationStageError,
    evaluate,
)

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
    paths = ["raw", "interim", "processed", "splits", "leakage", "samples", "models"]
    return {
        "name": "testapp",
        "title": "Test app",
        "data": {
            "source": {
                "release": "GB999",
                "doi": "10.0000/example",
                "licence": "CC-BY-4.0",
                "archive_url": "https://example.org/fake.zip",
                "checksums_url": "https://example.org/sums.txt",
                "archive_name": "fake.zip",
                "member_template": "x/{gene}.fasta.gz",
                "genes": ["CO1"],
            },
            "paths": {f"{name}_dir": str(tmp_path / name) for name in paths},
        },
        "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
        "router": {
            "training": {
                "experiment": "test-router",
                "registered_model": "test-router",
            },
            "evaluation": {
                "fragment_lengths": [20, 1000],
                "fragment_fraction": 1.0,
                # 21 records are scored, so the last batch is empty
                "batch_size": 7,
                "threads": 2,
            },
        },
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}",
            "artifact_dir": str(tmp_path / "mlflow" / "artifacts"),
        },
    }


def registered_tracker() -> FakeTracker:
    tracker = FakeTracker()
    tracker.log_model(make_model(), None, "test-router", "candidate")
    return tracker


def test_evaluate_scores_each_test_set(tmp_path: Path) -> None:
    write_data(tmp_path)
    tracker = registered_tracker()
    config = AppConfig.model_validate(config_dict(tmp_path))

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
    config = AppConfig.model_validate(config_dict(tmp_path))

    with pytest.raises(EvaluationStageError, match="run the split and leakage"):
        evaluate(config, lambda _config: registered_tracker())


def test_evaluate_without_novel_splits_needs_no_leakage_data(tmp_path: Path) -> None:
    write_data(tmp_path)
    for path in (tmp_path / "leakage").rglob("*.parquet"):
        path.unlink()
    settings = config_dict(tmp_path)
    settings["router"]["evaluation"] |= {"splits": ["test_known"], "novel_splits": []}

    report = evaluate(
        AppConfig.model_validate(settings), lambda _c: registered_tracker()
    )

    assert set(report.splits) == {"test_known"}
    assert "near_duplicates" not in report.data_sha256


def write_app_toml(tmp_path: Path) -> Path:
    data = config_dict(tmp_path)
    lines = [f'name = "{data["name"]}"', f'title = "{data["title"]}"']
    for section, values in (
        ("data.source", data["data"]["source"]),
        ("data.paths", data["data"]["paths"]),
        ("validation", data["validation"]),
        ("router.training", data["router"]["training"]),
        ("router.evaluation", data["router"]["evaluation"]),
        ("mlflow", data["mlflow"]),
    ):
        lines.append(f"\n[{section}]")
        # These values are written the same way in TOML and JSON
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    path = tmp_path / "app.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_evaluate_reports_missing_data(tmp_path: Path) -> None:
    assert main(["evaluate", str(write_app_toml(tmp_path))]) == 1


def test_cli_evaluate_loads_the_registered_model_from_mlflow(tmp_path: Path) -> None:
    pytest.importorskip("mlflow")
    from prokopton.ml.training.tracking import MlflowTracker

    write_data(tmp_path)
    config = AppConfig.model_validate(config_dict(tmp_path))
    tracker = MlflowTracker(
        config.mlflow.tracking_uri, config.mlflow.artifact_dir, "test-router"
    )
    with tracker.run("register"):
        tracker.log_model(make_model(), None, "test-router", "candidate")

    assert main(["evaluate", str(write_app_toml(tmp_path))]) == 0

    saved = json.loads((tmp_path / "models" / "router" / REPORT_NAME).read_text())
    assert saved["model_version"] == "1"
    assert saved["splits"]["test_known"]["metrics"]["accuracy"] == 1.0
