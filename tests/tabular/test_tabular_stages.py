"""Tests for the tabular classifier's settings and stages, on a small made-up
table (the Palmer Penguins example runs the real thing in CI)."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig, UnknownClassifierError
from prokopton.ml.tabular import stages
from prokopton.ml.tabular.config import (
    TableConfig,
    TableSplitConfig,
    TabularClassifierConfig,
    table_classifier,
)
from prokopton.ml.tabular.stages import (
    TableError,
    build_pipeline,
    evaluate,
    read_split,
    read_table,
    split,
    train,
    write_card,
)
from tabular_world import app_data


def app(tmp_path: Path, **model: Any) -> AppConfig:
    return AppConfig.model_validate(app_data(tmp_path, **model))


# --- Settings ---


def test_a_table_needs_input_columns_used_once() -> None:
    with pytest.raises(ValidationError, match="at least one numeric"):
        TableConfig(path=Path("t.csv"), target="kind")
    with pytest.raises(ValidationError, match="used once"):
        TableConfig(path=Path("t.csv"), target="kind", numeric=("kind",))
    with pytest.raises(ValidationError, match="used once"):
        TableConfig(path=Path("t.csv"), target="k", numeric=("a",), categorical=("a",))
    table = TableConfig(path=Path("t"), target="k", numeric=("a",), categorical=("b",))
    assert table.columns == ("a", "b")


def test_split_fractions_must_leave_records_to_train_on() -> None:
    with pytest.raises(ValidationError, match="must be less than 1"):
        TableSplitConfig(validation_fraction=0.5, test_fraction=0.5)


def test_the_settings_are_read_from_the_app(tmp_path: Path) -> None:
    config = app(tmp_path)

    settings = table_classifier(config, "kind")
    assert isinstance(settings, TabularClassifierConfig)
    assert settings.training.c_values == (0.01, 1.0)
    with pytest.raises(UnknownClassifierError, match="no 'other' tabular"):
        table_classifier(config, "other")


# --- Reading the table ---


def test_the_table_is_read_with_numbers_and_missing_values(tmp_path: Path) -> None:
    table = table_classifier(app(tmp_path), "kind").table

    frame = read_table(table)

    assert list(frame.columns) == ["size", "weight", "colour", "kind"]
    assert frame["size"].dtype == float
    assert frame["weight"].isna().sum() == 9
    assert frame["colour"].isna().sum() == 6
    assert frame["kind"].isna().sum() == 1


def test_a_missing_table_or_column_is_an_error(tmp_path: Path) -> None:
    table = table_classifier(app(tmp_path), "kind").table
    with pytest.raises(TableError, match="not found"):
        read_table(table.model_copy(update={"path": tmp_path / "none.csv"}))
    with pytest.raises(TableError, match="has no column height"):
        read_table(table.model_copy(update={"numeric": ("size", "height")}))


def test_a_numeric_column_must_hold_numbers(tmp_path: Path) -> None:
    table = table_classifier(app(tmp_path), "kind").table

    with pytest.raises(TableError, match="column 'colour' is not numeric"):
        read_table(table.model_copy(update={"numeric": ("colour",), "categorical": ()}))


# --- Stages ---


def test_split_is_stratified_and_leaves_out_records_without_a_target(
    tmp_path: Path,
) -> None:
    config = app(tmp_path)

    report = split(config, "kind")

    assert report.left_out == 1
    assert report.records == {"train": 54, "validation": 18, "test": 18}
    assert report.classes["test"] == {"large": 6, "medium": 6, "small": 6}
    assert report.release == "2026-1"
    assert len(read_split(config, "kind", "validation")) == 18
    again = split(config, "kind")
    assert again.records == report.records  # The seed fixes the split


def test_a_class_too_small_to_split_is_an_error(tmp_path: Path) -> None:
    config = app(tmp_path)
    path = table_classifier(config, "kind").table.path
    path.write_text(path.read_text("utf-8") + "9.5,19.0,blue,rare\n", "utf-8")

    with pytest.raises(TableError, match="cannot be split by 'kind'"):
        split(config, "kind")


def test_training_needs_the_splits(tmp_path: Path) -> None:
    with pytest.raises(TableError, match="run the split stage"):
        train(app(tmp_path), "kind", lambda _: FakeTracker())


def test_training_chooses_and_registers_the_best_candidate(tmp_path: Path) -> None:
    config = app(tmp_path)
    split(config, "kind")
    tracker = FakeTracker()

    report = train(config, "kind", lambda _: tracker)

    assert [c.c for c in report.candidates] == [0.01, 1.0]
    assert report.chosen_c == 1.0  # Too strong a penalty (0.01) does worse
    assert report.classes == ["large", "medium", "small"]
    assert report.registered_model == "things-kind"
    _, example, name, alias = tracker.models[0]
    assert (name, alias) == ("things-kind", "candidate")
    assert list(example.columns) == ["size", "weight", "colour"]
    assert tracker.runs[0].tags["release"] == "2026-1"
    assert [run.params["c"] for run in tracker.runs[1:]] == [0.01, 1.0]
    assert tracker.runs[0].metrics["chosen_accuracy"] == 1.0
    saved = json.loads((tmp_path / "models" / "kind" / "train.json").read_text())
    assert saved["model_version"] == "1"


def test_the_registered_name_can_be_set(tmp_path: Path) -> None:
    config = app(tmp_path, training={"c_values": [1.0], "registered_model": "my-kind"})
    split(config, "kind")

    assert train(config, "kind", lambda _: FakeTracker()).registered_model == "my-kind"


def test_evaluation_measures_the_registered_model_on_the_test_set(
    tmp_path: Path,
) -> None:
    config = app(tmp_path)
    split(config, "kind")
    tracker = FakeTracker()
    train(config, "kind", lambda _: tracker)

    report = evaluate(config, "kind", lambda _: tracker)

    assert report.records == 18
    assert report.metrics["accuracy"] == 1.0
    assert report.confusion["small"] == {"large": 0, "medium": 0, "small": 6}
    assert report.model_version == "1"


def test_the_card_needs_every_report(tmp_path: Path) -> None:
    config = app(tmp_path)
    split(config, "kind")

    with pytest.raises(TableError, match="run the train stage first"):
        write_card(config, "kind")


def test_the_card_describes_the_model_data_and_results(tmp_path: Path) -> None:
    config = app(tmp_path, card=str(tmp_path / "cards" / "kind.md"))
    split(config, "kind")
    tracker = FakeTracker()
    train(config, "kind", lambda _: tracker)
    evaluate(config, "kind", lambda _: tracker)

    path = write_card(config, "kind")

    assert path == tmp_path / "cards" / "kind.md"
    card = path.read_text("utf-8")
    assert "# Model card: kind (Things)" in card
    assert "C = 1, chosen on the validation set from 0.01, 1" in card
    assert "1 records without a target were left out" in card
    assert "train 54, validation 18, test 18" in card
    assert "accuracy 1.000" in card
    assert "| small | 0 | 0 | 6 |" in card


def test_a_pipeline_of_only_numbers_or_only_categories(tmp_path: Path) -> None:
    table = table_classifier(app(tmp_path), "kind").table
    frame = read_table(table).dropna(subset=["kind"])

    for columns in ({"categorical": ()}, {"numeric": ()}):
        only = table.model_copy(update=columns)
        pipeline = build_pipeline(only, 1.0, 200)
        pipeline.fit(frame[list(only.columns)], frame["kind"])
        assert len(pipeline.classes_) == 3


def test_the_stages_run_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "prokopton.toml"
    data = app_data(tmp_path)
    model = data["models"]["kind"]
    lines = [
        "schema = 4",
        'name = "things"',
        'title = "Things"',
        "[data]",
        'release = "2026-1"',
        "[paths]",
        f'models_dir = "{tmp_path / "models"}"',
        "[models.kind]",
        'kind = "core.tabular-classifier"',
        "[models.kind.table]",
        f'path = "{model["table"]["path"]}"',
        'target = "kind"',
        'numeric = ["size", "weight"]',
        'categorical = ["colour"]',
        "[pipeline]",
        'stages = ["split", "train", "evaluate", "card"]',
    ]
    for name in ("split", "train", "evaluate", "card"):
        lines += [f"[stages.{name}]", f'kind = "core.tabular-{name}"', 'model = "kind"']
    path.write_text("\n".join(lines) + "\n", "utf-8")
    monkeypatch.chdir(tmp_path)
    tracker = FakeTracker()
    monkeypatch.setattr(stages, "mlflow_tracker", lambda config, model: tracker)

    assert main(["run", "--all"]) == 0
    assert main(["stages"]) == 0
    assert "core.tabular-card (kind)  [last run" in capsys.readouterr().out
    assert (tmp_path / "models" / "kind" / "card.md").is_file()


def test_the_classifier_is_registered_in_mlflow_and_loads_again(
    tmp_path: Path,
) -> None:
    mlflow = pytest.importorskip("mlflow")
    data = app_data(tmp_path)
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    data["mlflow"] = {"tracking_uri": tracking, "artifact_dir": str(tmp_path / "art")}
    config = AppConfig.model_validate(data)
    split(config, "kind")

    report = train(config, "kind")
    tested = evaluate(config, "kind")

    assert report.model_version == tested.model_version == "1"
    mlflow.set_tracking_uri(tracking)
    loaded = mlflow.sklearn.load_model("models:/things-kind@candidate")
    record = read_split(config, "kind", "test").head(1)
    assert loaded.predict(record[["size", "weight", "colour"]])[0] in {
        "small",
        "medium",
        "large",
    }
