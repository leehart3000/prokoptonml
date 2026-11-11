"""Tests for prokopton.ml.pipeline.specialist_evaluate."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline import specialist_evaluate
from prokopton.ml.pipeline.reference import build_reference
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

rng = np.random.default_rng(5)


def random_sequence(length: int = 300) -> str:
    return "".join(rng.choice(list("ACGT"), length))


def mutate(sequence: str, share: float) -> str:
    bases = list(sequence)
    for position in rng.choice(len(bases), int(len(bases) * share), replace=False):
        bases[position] = str(rng.choice([b for b in "ACGT" if b != bases[position]]))
    return "".join(bases)


AEGYPTI = random_sequence()
ALBOPICTUS = random_sequence()
PIPIENS = random_sequence()
MUSCA = random_sequence()


def records(*, mislabelled: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(count: int, split: str, species: str, base: str, share: float) -> None:
        family = "Muscidae" if species.startswith("Musca") else "Culicidae"
        for _ in range(count):
            rows.append(
                {
                    "record_id": f"r{len(rows)}",
                    "split": split,
                    "order": "Diptera",
                    "family": family,
                    "genus": species.split()[0],
                    "species": species,
                    "species_named": True,
                    "ambiguous_fraction": 0.0,
                    "sequence": mutate(base, share),
                }
            )

    add(6, "train", "Aedes aegypti", AEGYPTI, 0.003)
    add(6, "train", "Aedes albopictus", ALBOPICTUS, 0.003)
    add(6, "train", "Culex pipiens", PIPIENS, 0.003)
    add(6, "train", "Musca domestica", MUSCA, 0.003)
    for stage in ("val", "test"):
        add(5, f"{stage}_known", "Aedes aegypti", AEGYPTI, 0.005)  # 3 drawn
        add(2, f"{stage}_known", "Aedes albopictus", ALBOPICTUS, 0.005)
        # A held-out species 4% from Ae. aegypti: the genus at best
        add(2, f"{stage}_novel", "Aedes novus", AEGYPTI, 0.04)
        # A held-out species near-identical to Cx. pipiens: leaky
        add(2, f"{stage}_novel", "Culex novus", PIPIENS, 0.005)
        add(3, f"{stage}_novel", "Musca domestica", MUSCA, 0.01)
    if mislabelled:
        # Named as another species, but with Ae. aegypti's sequence
        add(1, "val_known", "Aedes mislabelled", AEGYPTI, 0.0)
    rows.append({**rows[0], "record_id": "skip1", "split": "val_known"})
    rows[-1]["ambiguous_fraction"] = 0.5
    rows.append({**rows[0], "record_id": "skip2", "split": "val_known"})
    rows[-1]["species_named"] = False
    rows.append({**rows[0], "record_id": "skip3", "split": "val_known"})
    rows[-1]["order"] = "Lepidoptera"
    rows[-1]["family"] = "Noctuidae"
    return rows


def config_dict(tmp_path: Path, **evaluation: Any) -> dict[str, Any]:
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
        "specialists": {
            "culicidae": {
                "target": "Culicidae",
                "background": "Diptera",
                "k": 6,
                "evaluation": {"workers": 1, **evaluation},
            }
        },
    }


def write_data(tmp_path: Path, *, mislabelled: bool = False) -> list[dict[str, Any]]:
    rows = records(mislabelled=mislabelled)
    folder = Path("midori2_gb999") / "gene=CO1"
    split = tmp_path / "splits" / folder / "records.parquet"
    split.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split)
    leaky = [row["record_id"] for row in rows if row["species"] == "Culex novus"]
    leakage = tmp_path / "leakage" / folder / "near_duplicates.parquet"
    leakage.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"record_id": r} for r in leaky]), leakage)
    return rows


def prepared(
    tmp_path: Path, *, mislabelled: bool = False, **evaluation: Any
) -> AppConfig:
    write_data(tmp_path, mislabelled=mislabelled)
    config = AppConfig.model_validate(config_dict(tmp_path, **evaluation))
    build_reference(config, "culicidae")
    return config


def test_tunes_on_validation_and_scores_the_test_sets(tmp_path: Path) -> None:
    config = prepared(tmp_path)
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
    assert "No thresholds met the constraints" in caplog.text


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


def test_chooses_the_best_anyway_when_no_thresholds_meet_the_constraints(
    tmp_path: Path,
) -> None:
    rules = AppConfig.model_validate(config_dict(tmp_path)).specialist("culicidae")
    bad = score_set([answer("species", "Aedes x")], [row("Aedes aegypti")])
    first = Thresholds(species_identity=0.98, group_margin=0.01, member_identity=0.9)
    second = first.model_copy(update={"species_identity": 0.99})
    good_known = score_set([answer("species", "Aedes aegypti")], [row("Aedes aegypti")])
    scored: list[tuple[Thresholds, dict[str, SetScore]]] = [
        (first, {"val_known": bad}),
        (second, {"val_known": bad, "val_novel": bad}),
    ]

    grid, chosen, met = _choose(rules, scored)

    assert not met
    assert chosen == first  # objective 0 - 0 beats 0 - 1
    assert [point.eligible for point in grid] == [False, False]
    _, chosen, met = _choose(rules, [(second, {"val_known": good_known})])
    assert (chosen, met) == (second, True)


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
