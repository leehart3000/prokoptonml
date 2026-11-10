"""Tests for prokopton.ml.pipeline.train."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.train import REPORT_NAME, TrainStageError, train

MOTIFS = {"Diptera": "ACGTTGCA", "Fungi": "GGGCCCAT"}
TRAINING = {
    "k_values": [3],
    "ambiguity_thresholds": [0.05],
    "naive_bayes_alphas": [0.1, 1.0],
    "logistic_k_values": [],
    "threads": 1,
    "experiment": "test-router",
    "registered_model": "test-router",
}


def write_samples(tmp_path: Path) -> Path:
    root = tmp_path / "samples" / "midori2_gb999" / "router"
    root.mkdir(parents=True)
    for split, count in {"train": 20, "val_known": 6, "val_novel": 4}.items():
        rows = [
            {
                "label": label,
                "ambiguous_fraction": 0.0,
                "sequence": motif * 10 + "ACGT"[i % 4] * (i % 5),
            }
            for label, motif in MOTIFS.items()
            for i in range(count)
        ]
        pq.write_table(pa.Table.from_pylist(rows), root / f"{split}.parquet")
    (root / "_sample_report.json").write_text("{}\n", encoding="utf-8")
    return root


def config_dict(tmp_path: Path, **training: Any) -> dict[str, Any]:
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
            "paths": {
                "raw_dir": str(tmp_path / "raw"),
                "interim_dir": str(tmp_path / "interim"),
                "processed_dir": str(tmp_path / "processed"),
                "splits_dir": str(tmp_path / "splits"),
                "leakage_dir": str(tmp_path / "leakage"),
                "samples_dir": str(tmp_path / "samples"),
                "models_dir": str(tmp_path / "models"),
            },
        },
        "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
        "router": {"training": {**TRAINING, **training}},
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}",
            "artifact_dir": str(tmp_path / "mlflow" / "artifacts"),
        },
    }


def make_config(tmp_path: Path, **training: Any) -> AppConfig:
    return AppConfig.model_validate(config_dict(tmp_path, **training))


def test_train_registers_the_best_and_writes_a_report(tmp_path: Path) -> None:
    write_samples(tmp_path)
    tracker = FakeTracker()

    report = train(make_config(tmp_path), lambda _config: tracker)

    assert report.candidates_tried == 2
    assert report.best_model == "naive_bayes"
    assert report.metrics["val_known_accuracy"] > 0.9
    assert set(report.sample_sha256) == {
        "train",
        "val_known",
        "val_novel",
        "sample_report",
    }
    assert tracker.runs[0].tags["sample_sha256.train"] == report.sample_sha256["train"]
    assert tracker.runs[0].tags["release"] == "GB999"
    assert tracker.models[0][2:] == ("test-router", "candidate")
    saved = json.loads(
        (tmp_path / "models" / "router" / REPORT_NAME).read_text("utf-8")
    )
    assert saved["model_version"] == "1"
    assert saved["rules"]["naive_bayes_alphas"] == [0.1, 1.0]


def test_train_requires_samples(tmp_path: Path) -> None:
    with pytest.raises(TrainStageError, match="run the sample stage first"):
        train(make_config(tmp_path), lambda _config: FakeTracker())


def test_train_rejects_thresholds_above_the_samples(tmp_path: Path) -> None:
    write_samples(tmp_path)
    config = make_config(tmp_path, ambiguity_thresholds=[0.05, 0.5])

    with pytest.raises(TrainStageError, match=r"\[0\.5\]"):
        train(config, lambda _config: FakeTracker())


def write_app_toml(tmp_path: Path) -> Path:
    config_path = tmp_path / "app.toml"
    data = config_dict(tmp_path)
    # These values are written the same way in TOML and JSON
    lines = [f'name = "{data["name"]}"', f'title = "{data["title"]}"']
    for section, values in (
        ("data.source", data["data"]["source"]),
        ("data.paths", data["data"]["paths"]),
        ("validation", data["validation"]),
        ("router.training", data["router"]["training"]),
        ("mlflow", data["mlflow"]),
    ):
        lines.append(f"\n[{section}]")
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def test_cli_train_reports_missing_samples(tmp_path: Path) -> None:
    assert main(["train", str(write_app_toml(tmp_path))]) == 1


def test_cli_train_tracks_and_registers_with_mlflow(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    write_samples(tmp_path)

    assert main(["train", str(write_app_toml(tmp_path))]) == 0

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow' / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    version = client.get_model_version_by_alias("test-router", "candidate")
    assert str(version.version) == "1"
    runs = mlflow.search_runs(experiment_names=["test-router"])
    assert len(runs) == 3  # the search run and one per candidate
    model = mlflow.pyfunc.load_model("models:/test-router@candidate")
    predictions = model.predict(pd.DataFrame({"sequence": [MOTIFS["Fungi"] * 10]}))
    assert list(predictions) == ["Fungi"]
