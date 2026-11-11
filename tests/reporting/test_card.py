"""Tests for prokopton.ml.reporting.card."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from prokopton.ml.cli import main
from prokopton.ml.config import (
    AppConfig,
    ClassifierCalibrationConfig,
    ClassifierEvaluationConfig,
    ClassifierSampleConfig,
    ClassifierTrainingConfig,
    UnknownClassifierError,
)
from prokopton.ml.evaluation.scores import (
    AmbiguityBin,
    CalibrationBin,
    ClassScore,
    CoveragePoint,
    GroupScore,
)
from prokopton.ml.pipeline.calibrate import CalibrationReport
from prokopton.ml.pipeline.evaluate import EvaluationReport, FragmentScore, SplitScores
from prokopton.ml.pipeline.sample import SampleReport, SplitSampleReport
from prokopton.ml.pipeline.train import TrainReport
from prokopton.ml.reporting.card import CardError, build_card

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)
METRICS = {
    "accuracy": 0.97,
    "top5_accuracy": 0.99,
    "macro_f1": 0.89,
    "ece": 0.006,
    "mean_confidence": 0.968,
}
COMMON: dict[str, Any] = {
    "app": "testapp",
    "release": "GB999",
    "gene": "CO1",
    "prokopton_ml_version": "0.1.0",
    "started_at": NOW,
    "finished_at": NOW,
}


def config_dict(tmp_path: Path) -> dict[str, Any]:
    paths = ["raw", "interim", "processed", "splits", "leakage", "samples", "models"]
    return {
        "name": "testapp",
        "title": "DNA barcoding",
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
            "card": {
                "output_dir": str(tmp_path / "cards"),
                "notes": str(tmp_path / "notes.md"),
                "title": "router model card",
                "task": "Predict the broad group (for animals, the order)",
                "intended_use": "For COI of at least {min_length} bases.",
                "limitations": ["**Only COI** is supported."],
            }
        },
    }


def sample_report() -> SampleReport:
    split = SplitSampleReport(
        eligible_records=100,
        sampled_records=90,
        classes=2,
        classes_below_cap=1,
        mean_species_per_class=12.0,
        output="x.parquet",
    )
    return SampleReport(
        **COMMON,
        rules=ClassifierSampleConfig(),
        labels=["Diptera", "Hymenoptera"],
        dropped_classes={"Rare": 3},
        excluded_near_duplicates=7,
        splits={"train": split, "val_known": split},
    )


def train_report(version: str = "4") -> TrainReport:
    metrics = {
        f"{name}_{key}": value
        for name in (
            "val_known",
            "val_novel",
            "val_known_fragments",
            "val_novel_fragments",
        )
        for key, value in METRICS.items()
    }
    return TrainReport(
        **COMMON,
        rules=ClassifierTrainingConfig(fragments_per_record=1),
        sample_sha256={"train": "a" * 64},
        candidates_tried=3,
        best_model="logistic",
        best_k=8,
        best_max_ambiguous_fraction=0.1,
        best_alpha=1e-7,
        metrics={**metrics, "selection_score": 0.89},
        registered_model="test-router",
        model_version=version,
        alias="candidate",
        parent_run_id="train-run",
    )


def calibration_report(
    base: str = "4", version: str = "5", b: float = -0.009
) -> CalibrationReport:
    return CalibrationReport(
        **COMMON,
        rules=ClassifierCalibrationConfig(),
        registered_model="test-router",
        base_version=base,
        model_version=version,
        alias="candidate",
        a=-0.281,
        b=b,
        nll_before=0.25,
        nll_after=0.21,
        before={"val_novel": {**METRICS, "ece": 0.04, "mean_confidence": 0.92}},
        after={"val_novel": METRICS},
        run_id="calibration-run",
    )


def split_scores() -> SplitScores:
    return SplitScores(
        records=1000,
        metrics=METRICS,
        by_ambiguity=[
            AmbiguityBin(bin="0", records=990, accuracy=0.97, macro_f1=0.9),
            AmbiguityBin(bin="> 0.1", records=0, accuracy=0.0, macro_f1=0.0),
        ],
        calibration=[
            CalibrationBin(
                lower=0.5, upper=0.6, records=40, mean_confidence=0.55, accuracy=0.5
            ),
            CalibrationBin(
                lower=0.9, upper=1.0, records=960, mean_confidence=0.98, accuracy=0.99
            ),
        ],
        coverage=[CoveragePoint(threshold=0.9, coverage=0.94, accuracy=0.997)],
    )


def evaluation_report(version: str = "5") -> EvaluationReport:
    fragments = [
        FragmentScore(length=length, records=100, too_short=0, metrics=METRICS)
        for length in (None, 50, 100, 300)
    ]
    classes = {
        "Diptera": ClassScore(precision=0.99, recall=0.98, f1=0.985, support=500),
        "Rare": ClassScore(precision=0.5, recall=0.4, f1=0.44, support=25),
        "Tiny": ClassScore(precision=0.0, recall=0.0, f1=0.0, support=3),
    }
    return EvaluationReport(
        **COMMON,
        rules=ClassifierEvaluationConfig(),
        registered_model="test-router",
        alias="candidate",
        model_version=version,
        data_sha256={"records": "b" * 64},
        unlabelled_records={"test_known": 6},
        unseen_class_records={"test_novel": 2},
        splits={name: split_scores() for name in ("test_known", "test_novel_clean")},
        fragments={"test_known": fragments, "test_novel_clean": fragments},
        per_class={"test_known": classes, "test_novel_clean": classes},
        run_id="evaluation-run",
    )


def write_reports(
    tmp_path: Path,
    *,
    train_version: str = "4",
    calibration: CalibrationReport | None = None,
    evaluation_version: str = "5",
) -> AppConfig:
    config = AppConfig.model_validate(config_dict(tmp_path))
    sample_dir = tmp_path / "samples" / "midori2_gb999" / "router"
    model_dir = tmp_path / "models" / "router"
    sample_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    (sample_dir / "_sample_report.json").write_text(sample_report().model_dump_json())
    (model_dir / "_train_report.json").write_text(
        train_report(train_version).model_dump_json()
    )
    if calibration is not None:
        (model_dir / "_calibration_report.json").write_text(
            calibration.model_dump_json()
        )
    (model_dir / "_evaluation_report.json").write_text(
        evaluation_report(evaluation_version).model_dump_json()
    )
    (tmp_path / "notes.md").write_text("Hand-written notes.\n")
    return config


def test_card_describes_the_evaluated_calibrated_model(tmp_path: Path) -> None:
    config = write_reports(tmp_path, calibration=calibration_report())

    files = build_card(config)

    card = files.card.read_text()
    assert files.model_version == "5"
    assert card.startswith("# DNA barcoding: router model card")
    assert "| Version | 5 (alias `@candidate`) |" in card
    assert "Logistic regression on 8-mer frequencies, rescaled to unit length" in card
    assert "Length-aware temperature, from v4" in card
    assert "T = exp(-0.2810 - 0.0090 * ln(length))" in card
    assert "| Novel species (clean) | 1,000 | 97.0% |" in card
    assert "![Accuracy by sequence length](router-fragments.svg)" in card
    assert "| Whole |" in card
    # Classes with under 20 test records are left out of the hardest list
    assert "| Rare | 25 |" in card
    assert "| Tiny |" not in card
    assert "## Development notes\n\nHand-written notes." in card
    assert "(1 of them), and 7 validation records" in card
    assert "In the final round of the search, 3 candidate models were" in card
    assert "the broad group (for animals, the order)" in card
    assert "1 random fragment of 75-600 bases" in card
    assert "| Validation, novel species | 97.0% | 92.0% → 96.8% |" in card
    assert "(at 100 bases, 97% accurate for novel species (clean))" in card
    assert "## Intended use\n\nFor COI of at least 100 bases." in card
    assert "- **Only COI** is supported." in card
    # These need per-group scores and the strand check
    assert "## By group of organisms" not in card
    assert "## Which strand?" not in card
    assert "`calibration-run`" in card
    assert files.fragments_chart.read_text().startswith("<svg")
    assert files.reliability_chart.read_text().startswith("<svg")


def test_card_for_a_model_without_novel_species(tmp_path: Path) -> None:
    config = write_reports(tmp_path, calibration=calibration_report())
    path = tmp_path / "models" / "router" / "_evaluation_report.json"
    report = evaluation_report()
    known = report.splits["test_known"].model_copy(
        update={
            "by_group": [
                GroupScore(group="Metazoa", records=900, accuracy=0.99, macro_f1=0.98),
                GroupScore(group="Fungi", records=100, accuracy=0.9, macro_f1=0.85),
            ]
        }
    )
    fragments = [
        point.model_copy(update={"forward_strand_share": 0.995})
        for point in report.fragments["test_known"]
    ]
    path.write_text(
        report.model_copy(
            update={
                "splits": {"test_known": known},
                "fragments": {"test_known": fragments},
                "per_class": {"test_known": report.per_class["test_known"]},
            }
        ).model_dump_json()
    )

    card = build_card(config).card.read_text()

    assert "| Test set | 1,000 | 97.0% |" in card
    assert "Records were split into training, validation and test sets" in card
    assert "| Test set | Fungi | 100 | 90.0% | 0.850 |" in card
    assert "## Which strand?" in card
    assert "| Whole | 99.5% |" in card
    assert "lowest F1 on test set" in card


def test_card_formula_with_a_positive_length_term(tmp_path: Path) -> None:
    config = write_reports(tmp_path, calibration=calibration_report(b=0.02))

    card = build_card(config).card.read_text()

    assert "T = exp(-0.2810 + 0.0200 * ln(length))" in card


def test_card_for_an_uncalibrated_model(tmp_path: Path) -> None:
    config = write_reports(tmp_path, evaluation_version="4")

    card = build_card(config).card.read_text()

    assert "This model's confidence is not calibrated." in card
    assert "| Calibration | None |" in card


def test_card_for_an_app_without_that_classifier(tmp_path: Path) -> None:
    config = write_reports(tmp_path, calibration=calibration_report())

    with pytest.raises(UnknownClassifierError, match="marker"):
        build_card(config, "marker")


def test_card_without_notes(tmp_path: Path) -> None:
    settings = config_dict(tmp_path)
    write_reports(tmp_path, calibration=calibration_report())
    del settings["router"]["card"]["notes"]

    card = build_card(AppConfig.model_validate(settings)).card.read_text()

    assert "Development notes" not in card


@pytest.mark.parametrize(
    ("calibration", "evaluation_version", "message"),
    [
        (calibration_report(base="3"), "5", "Calibration is of v3"),
        (calibration_report(), "4", "Evaluation is of v4"),
    ],
)
def test_card_refuses_reports_of_different_models(
    tmp_path: Path,
    calibration: CalibrationReport,
    evaluation_version: str,
    message: str,
) -> None:
    config = write_reports(
        tmp_path, calibration=calibration, evaluation_version=evaluation_version
    )

    with pytest.raises(CardError, match=message):
        build_card(config)


def test_card_requires_the_reports(tmp_path: Path) -> None:
    config = AppConfig.model_validate(config_dict(tmp_path))

    with pytest.raises(CardError, match="run the pipeline first"):
        build_card(config)


def test_card_requires_the_notes_file_it_names(tmp_path: Path) -> None:
    config = write_reports(tmp_path, calibration=calibration_report())
    (tmp_path / "notes.md").unlink()

    with pytest.raises(CardError, match=r"notes\.md not found"):
        build_card(config)


def write_app_toml(tmp_path: Path) -> Path:
    data = config_dict(tmp_path)
    lines = [f'name = "{data["name"]}"', f'title = "{data["title"]}"']
    for section, values in (
        ("data.source", data["data"]["source"]),
        ("data.paths", data["data"]["paths"]),
        ("validation", data["validation"]),
        ("router.card", data["router"]["card"]),
    ):
        lines.append(f"\n[{section}]")
        # These values are written the same way in TOML and JSON
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    path = tmp_path / "app.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_card(tmp_path: Path) -> None:
    write_reports(tmp_path, calibration=calibration_report())

    assert main(["card", str(write_app_toml(tmp_path))]) == 0
    assert (tmp_path / "cards" / "router.md").exists()


def test_cli_card_reports_missing_reports(tmp_path: Path) -> None:
    assert main(["card", str(write_app_toml(tmp_path))]) == 1
