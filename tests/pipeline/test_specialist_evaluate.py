"""Tests for prokopton.ml.pipeline.specialist_evaluate."""

import json
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline import specialist_evaluate
from prokopton.ml.pipeline.specialist_evaluate import (
    REPORT_NAME,
    SetScore,
    SpecialistEvaluationError,
    Thresholds,
    _choose,
    evaluate_specialist,
    is_right,
    score_set,
)
from prokopton.ml.specialists.identify import Identification
from prokopton.ml.training import tracking
from specialist_world import config_dict, prepared


def test_tunes_on_validation_and_scores_the_test_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = prepared(tmp_path)
    monkeypatch.setattr(specialist_evaluate, "EXAMPLES", 1)
    tracker = FakeTracker()

    report = evaluate_specialist(config, "culicidae", lambda _: tracker)

    assert set(report.validation) == {
        "val_known",
        "val_novel_clean",
        "val_novel_leaky",
        "val_novel",
        "val_background",
    }
    assert set(report.test) == {
        name.replace("val", "test") for name in report.validation
    }
    known = report.test["test_known"]
    # Up to 3 records per species: 3 Ae. aegypti and 2 Ae. albopictus
    assert (known.records, known.species) == (5, 2)
    assert known.exact == 1.0
    assert report.test["test_background"].right == 1.0
    assert report.test["test_novel_clean"].outcomes == {"genus": 2}
    # The leaky held-out species is named as its near-identical relative
    assert report.test["test_novel_leaky"].wrongly_named == 1.0
    assert report.test["test_novel"].wrongly_named_by_species == 0.5
    assert report.constraints_met
    assert report.chosen in [point.thresholds for point in report.grid]
    # 5 species x 3 margins x 3 member thresholds
    assert len(report.grid) == 45
    saved = json.loads(
        (tmp_path / "models" / "specialists" / "culicidae" / REPORT_NAME).read_text()
    )
    assert saved["specialist"] == "culicidae"
    run = tracker.runs[0]
    assert run.name == "culicidae-evaluation"
    assert run.metrics["test_known.exact"] == 1.0
    assert "sha256.neighbours" in run.tags
    assert run.params["grid_points"] == 45
    # Two leaky records are wrong, but only one example is kept
    [leaky] = report.wrong["test_novel_leaky"]
    assert leaky.best_species == "Culex pipiens"
    assert report.wrong["test_known"] == []


def test_parallel_workers_give_the_same_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = prepared(tmp_path)
    monkeypatch.setattr(specialist_evaluate, "CHUNK", 1)
    monkeypatch.setattr(specialist_evaluate, "PROGRESS_EVERY", 1)
    one = evaluate_specialist(config, "culicidae", lambda _: FakeTracker())
    two_workers = config_dict(tmp_path, workers=2)
    parallel = AppConfig.model_validate(two_workers)

    two = evaluate_specialist(parallel, "culicidae", lambda _: FakeTracker())

    assert two.test == one.test
    assert two.chosen == one.chosen


def test_reports_when_no_thresholds_meet_the_constraints(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = prepared(tmp_path, mislabelled=True)

    report = evaluate_specialist(config, "culicidae", lambda _: FakeTracker())

    assert not report.constraints_met
    assert (report.background_ok, report.known_ok) == (True, False)
    assert "No thresholds met every constraint" in caplog.text
    wrong = report.wrong["val_known"][0]
    assert (wrong.species, wrong.outcome, wrong.best_species) == (
        "Aedes mislabelled",
        "species",
        "Aedes aegypti",
    )
    # No reference of its own species was among the candidates
    assert wrong.own_species_identity is None


def test_needs_its_inputs(tmp_path: Path) -> None:
    config = AppConfig.model_validate(config_dict(tmp_path))

    with pytest.raises(SpecialistEvaluationError, match="reference stages first"):
        evaluate_specialist(config, "culicidae", lambda _: FakeTracker())


def answer(outcome: str, label: str | None, *species: str) -> Identification:
    return Identification(outcome=outcome, label=label, identity=0.99, species=species)  # type: ignore[arg-type]


def row(species: str, role: str = "target", genus: str = "Aedes") -> dict[str, Any]:
    return {"record_id": species, "species": species, "genus": genus, "role": role}


def test_is_right_at_the_level_given() -> None:
    aegypti = row("Aedes aegypti")

    assert is_right(answer("species", "Aedes aegypti"), aegypti)
    assert is_right(answer("closest_species", "Aedes aegypti"), aegypti)
    assert is_right(answer("species_group", None, "Aedes aegypti", "Aedes x"), aegypti)
    assert not is_right(answer("complex", "C", "Aedes y"), aegypti)
    assert is_right(answer("genus", "Aedes"), aegypti)
    assert is_right(answer("genus_label_only", None), aegypti)
    assert not is_right(answer("genus_label_only", "Culex"), aegypti)
    assert is_right(answer("family", "Culicidae"), aegypti)
    assert not is_right(answer("not_member", None), aegypti)
    assert is_right(answer("not_member", None), row("Musca", role="background"))


def test_scores_are_averaged_per_record_and_per_species() -> None:
    rows = [row("Aedes aegypti")] * 3 + [row("Aedes vexans")]
    found = [answer("species", "Aedes aegypti")] * 3 + [
        answer("species", "Aedes aegypti")
    ]

    score = score_set(found, rows)

    assert (score.right, score.right_by_species) == (0.75, 0.5)
    assert (score.wrongly_named, score.wrongly_named_by_species) == (0.25, 0.5)
    assert score.outcomes == {"species": 4}
    assert score_set([], []).right_by_species == 0.0


def test_choosing_puts_the_background_first_then_known_species(
    tmp_path: Path,
) -> None:
    rules = AppConfig.model_validate(config_dict(tmp_path)).specialist("culicidae")
    bad = score_set([answer("species", "Aedes x")], [row("Aedes aegypti")])
    good = score_set([answer("species", "Aedes aegypti")], [row("Aedes aegypti")])
    kept_out = score_set([answer("not_member", None)], [row("M", role="background")])
    let_in = score_set([answer("family", "Culicidae")], [row("M", role="background")])
    novel_right = score_set([answer("genus", "Aedes")], [row("Aedes novus")])
    novel_left_out = score_set([answer("not_member", None)], [row("Aedes novus")])
    first = Thresholds(species_identity=0.98, group_margin=0.01, member_identity=0.9)
    second, third, fourth = (
        first.model_copy(update={"species_identity": value})
        for value in (0.985, 0.99, 0.995)
    )
    scored: list[tuple[Thresholds, dict[str, SetScore]]] = [
        # The best objective, but lets in the background
        (first, {"val_known": good, "val_background": let_in}),
        # Keeps out the background, but names a known species wrongly
        (second, {"val_known": bad, "val_background": kept_out}),
        # Ties with the next on the objective, but gets fewer novel right
        (
            third,
            {
                "val_known": good,
                "val_background": kept_out,
                "val_novel": novel_left_out,
            },
        ),
        (
            fourth,
            {"val_known": good, "val_background": kept_out, "val_novel": novel_right},
        ),
    ]

    grid, point = _choose(rules, scored)

    assert point.thresholds == fourth
    assert [(p.background_ok, p.known_ok) for p in grid] == [
        (False, True),
        (True, False),
        (True, True),
        (True, True),
    ]
    _, point = _choose(rules, scored[:2])
    assert point.thresholds == second


def write_app_toml(tmp_path: Path) -> Path:
    data = config_dict(tmp_path)
    lines = [f'name = "{data["name"]}"', f'title = "{data["title"]}"']
    specialist = dict(data["specialists"]["culicidae"])
    evaluation = specialist.pop("evaluation")
    for section, values in (
        ("data.source", data["data"]["source"]),
        ("data.paths", data["data"]["paths"]),
        ("validation", data["validation"]),
        ("specialists.culicidae", specialist),
        ("specialists.culicidae.evaluation", evaluation),
    ):
        lines.append(f"\n[{section}]")
        # These values are written the same way in TOML and JSON
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    path = tmp_path / "app.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_evaluate_a_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = write_app_toml(tmp_path)
    monkeypatch.setattr(
        specialist_evaluate, "specialist_tracker", lambda *_: FakeTracker()
    )

    assert main(["evaluate", str(config_path), "--specialist", "culicidae"]) == 1
    prepared(tmp_path)
    assert main(["evaluate", str(config_path), "--specialist", "culicidae"]) == 0


def test_specialist_tracker_names_its_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[tuple[Any, ...]] = []
    monkeypatch.setattr(tracking, "MlflowTracker", lambda *args: made.append(args))
    config = AppConfig.model_validate(config_dict(tmp_path))

    tracking.specialist_tracker(config, "culicidae")

    assert made[0][2] == "testapp-culicidae-specialist"
