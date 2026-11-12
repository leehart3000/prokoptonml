"""Tests for prokopton.ml.reporting.specialist_card."""

from pathlib import Path
from typing import Any

import pytest

from fakes import FakeTracker
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.reference import reference_path
from prokopton.ml.pipeline.specialist_evaluate import (
    REPORT_NAME,
    evaluate_specialist,
)
from prokopton.ml.reporting.specialist_card import (
    SpecialistCardError,
    build_specialist_card,
)
from specialist_world import config_dict, prepared


def card_settings(tmp_path: Path, notes: Path | None = None) -> dict[str, Any]:
    return {
        "output_dir": str(tmp_path / "cards"),
        "notes": str(notes) if notes else None,
        "title": "Culicidae card",
        "task": "Identify mosquitoes",
        "intended_use": "For testing.",
        "limitations": ["Only a test."],
    }


def tuned(tmp_path: Path, notes: Path | None = None) -> AppConfig:
    """A config whose thresholds are the ones its evaluation chose."""
    config = prepared(tmp_path)
    report = evaluate_specialist(config, "culicidae", lambda _: FakeTracker())
    data = config_dict(tmp_path)
    data["specialists"]["culicidae"].update(
        report.chosen.model_dump(), card=card_settings(tmp_path, notes)
    )
    return AppConfig.model_validate(data)


def test_writes_the_card_and_chart(tmp_path: Path) -> None:
    notes = tmp_path / "notes.md"
    notes.write_text("How it was made.\n")
    config = tuned(tmp_path, notes)

    files = build_specialist_card(config, "culicidae")

    text = files.card.read_text()
    assert files.card == tmp_path / "cards" / "culicidae.md"
    assert text.startswith("# Test app: Culicidae card")
    assert "- **Task:** Identify mosquitoes" in text
    assert "| Known species | 5 | 2 |" in text
    assert "Both limits were met." in text
    assert "## Development notes\n\nHow it was made." in text
    assert "- Only a test." in text
    assert "![Species threshold trade-off](culicidae-thresholds.svg)" in text
    assert "Answers for known species: species 5." in text
    assert "1 record of other Diptera in 1 family as the background" in text
    assert "Species complexes" not in text
    assert files.thresholds_chart.read_text().startswith("<svg")


def test_says_when_a_limit_was_missed(tmp_path: Path) -> None:
    config = tuned(tmp_path)
    data = config.model_dump(mode="json")
    data["specialists"]["culicidae"]["complexes"] = [
        {
            "name": "Aedes test complex",
            "members": ["Aedes aegypti", "Aedes x"],
            "resolve": "Use a nuclear assay.",
            "reference": "Someone (2020)",
        }
    ]
    config = AppConfig.model_validate(data)
    neighbours = reference_path(config, "culicidae").parent / "_reference_report.json"
    neighbours.write_text(
        neighbours.read_text()
        .replace('"species_with_neighbours": 0', '"species_with_neighbours": 3')
        .replace('"contradicted_skipped": 0', '"contradicted_skipped": 2')
    )
    path = reference_path(config, "culicidae").parent / REPORT_NAME
    path.write_text(
        path.read_text()
        .replace('"known_ok": true', '"known_ok": false')
        .replace('"constraints_met": true', '"constraints_met": false')
    )

    text = build_specialist_card(config, "culicidae").card.read_text()

    assert "known species limit met: False" in text
    assert "**0 of 3 named species** have another named species" in text
    assert "2 records labelled as one species but far more like another" in text
    assert "- **Aedes test complex** (2 names): Use a nuclear assay." in text


def test_refuses_thresholds_that_were_not_chosen(tmp_path: Path) -> None:
    config = tuned(tmp_path)
    data = config.model_dump(mode="json")
    data["specialists"]["culicidae"]["species_identity"] = 0.999

    with pytest.raises(SpecialistCardError, match="set the chosen values"):
        build_specialist_card(AppConfig.model_validate(data), "culicidae")


def test_needs_valid_reports_and_notes(tmp_path: Path) -> None:
    config = AppConfig.model_validate(config_dict(tmp_path))
    with pytest.raises(SpecialistCardError, match="run the reference"):
        build_specialist_card(config, "culicidae")

    config = tuned(tmp_path, tmp_path / "missing.md")
    with pytest.raises(SpecialistCardError, match=r"missing\.md not found"):
        build_specialist_card(config, "culicidae")

    path = reference_path(config, "culicidae").parent / REPORT_NAME
    path.write_text("{}")
    with pytest.raises(SpecialistCardError, match="not a valid report"):
        build_specialist_card(config, "culicidae")


def test_cli_writes_a_specialist_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tuned(tmp_path)
    monkeypatch.setattr("prokopton.ml.cli.load_config", lambda *_, **__: config)

    assert main(["card", "unused.toml", "--specialist", "culicidae"]) == 0
    assert (tmp_path / "cards" / "culicidae.md").exists()
