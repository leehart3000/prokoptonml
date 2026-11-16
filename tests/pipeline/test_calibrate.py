"""Tests for prokopton.ml.bio.pipeline.calibrate."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from bio_app import CLASSIFIER, app_config, mlflow, sequence_app, write_config
from fakes import FakeTracker
from prokopton.ml.bio.features.kmers import KmerVectorizer
from prokopton.ml.bio.pipeline.calibrate import (
    REPORT_NAME,
    CalibrationStageError,
    calibrate,
)
from prokopton.ml.cli import main
from prokopton.ml.training.calibration import LengthCalibratedClassifier

MOTIFS = {"Diptera": "ACGTTGCA", "Hymenoptera": "GGGCCCAT"}


def write_samples(tmp_path: Path) -> None:
    root = tmp_path / "samples" / "midori2_gb999" / "router"
    root.mkdir(parents=True)
    for split, count in {"val_known": 8, "val_novel": 6}.items():
        rows = [
            {"label": label, "ambiguous_fraction": 0.0, "sequence": motif * 12}
            for label, motif in MOTIFS.items()
            for _ in range(count)
        ]
        # A label the model lacks is left out of the fit
        rows.append(
            {"label": "Rare", "ambiguous_fraction": 0.0, "sequence": "ACGT" * 20}
        )
        pq.write_table(pa.Table.from_pylist(rows), root / f"{split}.parquet")


def make_model() -> Pipeline:
    sequences = [motif * 12 for motif in MOTIFS.values()] * 3
    model = Pipeline(
        [("kmers", KmerVectorizer(k=3)), ("classifier", MultinomialNB(fit_prior=False))]
    )
    return model.fit(sequences, list(MOTIFS) * 3)


def config_dict(tmp_path: Path) -> dict[str, Any]:
    router = {
        "kind": CLASSIFIER,
        "training": {"experiment": "test-router", "registered_model": "test-router"},
        "calibration": {
            "fragment_min_length": 20,
            "fragment_max_length": 60,
            "threads": 2,
        },
    }
    return sequence_app(tmp_path, models={"router": router}, mlflow=mlflow(tmp_path))


def registered_tracker() -> FakeTracker:
    tracker = FakeTracker()
    tracker.log_model(make_model(), None, "test-router", "candidate")
    return tracker


def test_calibrate_registers_a_calibrated_version(tmp_path: Path) -> None:
    write_samples(tmp_path)
    tracker = registered_tracker()
    config = app_config(config_dict(tmp_path))

    report = calibrate(config, lambda _config: tracker)

    assert (report.base_version, report.model_version) == ("1", "2")
    assert report.nll_after <= report.nll_before
    assert set(report.after) == {
        "val_known",
        "val_known_fragments",
        "val_novel",
        "val_novel_fragments",
    }
    # Calibration never changes which class is predicted
    for name, scores in report.after.items():
        assert scores["accuracy"] == report.before[name]["accuracy"]
    model, version = tracker.load_model("test-router", "candidate")
    assert version == "2"
    assert isinstance(model, LengthCalibratedClassifier)
    assert (model.a, model.b) == (report.a, report.b)
    run = tracker.runs[-1]
    assert run.name == "router-calibration"
    assert run.tags["base_model_version"] == "1"
    assert run.metrics["nll_after"] == report.nll_after
    saved = json.loads((tmp_path / "models" / "router" / REPORT_NAME).read_text())
    assert saved["model_version"] == "2"


def test_calibrating_again_replaces_the_correction(tmp_path: Path) -> None:
    write_samples(tmp_path)
    tracker = registered_tracker()
    config = app_config(config_dict(tmp_path))
    calibrate(config, lambda _config: tracker)

    report = calibrate(config, lambda _config: tracker)

    assert (report.base_version, report.model_version) == ("2", "3")
    model, _ = tracker.load_model("test-router", "candidate")
    assert isinstance(model.model, Pipeline)


def test_calibrate_requires_samples(tmp_path: Path) -> None:
    config = app_config(config_dict(tmp_path))

    with pytest.raises(CalibrationStageError, match="run the sample stage first"):
        calibrate(config, lambda _config: registered_tracker())


def write_app_toml(tmp_path: Path) -> Path:
    return write_config(tmp_path / "app.toml", config_dict(tmp_path))


def test_cli_calibrate_reports_missing_samples(tmp_path: Path) -> None:
    assert (
        main(["--config", str(write_app_toml(tmp_path)), "run", "router-calibrate"])
        == 1
    )


def test_cli_calibrate_registers_a_model_mlflow_can_load(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    from prokopton.ml.training.tracking import MlflowTracker

    write_samples(tmp_path)
    config = app_config(config_dict(tmp_path))
    tracker = MlflowTracker(
        config.mlflow.tracking_uri, config.mlflow.artifact_dir, "test-router"
    )
    with tracker.run("register"):
        tracker.log_model(make_model(), None, "test-router", "candidate")

    assert (
        main(["--config", str(write_app_toml(tmp_path)), "run", "router-calibrate"])
        == 0
    )

    model = mlflow.pyfunc.load_model("models:/test-router@candidate")
    query = pd.DataFrame({"sequence": [MOTIFS["Hymenoptera"] * 12]})
    assert list(model.predict(query)) == ["Hymenoptera"]
    loaded, version = tracker.load_model("test-router", "candidate")
    assert version == "2"
    assert isinstance(loaded, LengthCalibratedClassifier)
