"""Tests for rewriting a schema 1 config file as schema 2."""

import tomllib
from pathlib import Path

import pytest

from prokopton.ml import migrate
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.migrate import MigrationError, is_schema_1, migrate_text

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HEADER = """\
name = "testapp"
title = "Test app"

[data.source]
release = "GB999"
doi = "10.0000/example"
licence = "CC-BY-4.0"
archive_url = "https://example.org/fake.zip"
checksums_url = "https://example.org/sums.txt"
archive_name = "fake.zip"
member_template = "x/{gene}.fasta.gz"
genes = ["CO1"]

[data.paths]
raw_dir = "r"
interim_dir = "i"
processed_dir = "p"
splits_dir = "s"
leakage_dir = "l"
samples_dir = "a"
models_dir = "m"

[validation]
phylum_reference = "tests/fixtures/phylum_lineages_gb273.tsv"
"""


def test_the_dnabarcode_config_keeps_its_settings_and_comments() -> None:
    text = (FIXTURES / "dnabarcode.toml").read_text("utf-8")

    result = migrate_text(text)

    data = tomllib.loads(result)
    assert data["schema"] == 2
    assert not is_schema_1(data)
    assert {"router", "marker", "specialists"}.isdisjoint(data)
    assert data["models"]["culicidae"]["kind"] == "bio.specialist"
    assert len(data["pipeline"]["stages"]) == 19
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    assert all(line in result for line in comments)
    assert AppConfig.model_validate(data) == AppConfig.model_validate(
        tomllib.loads(text)
    )


ROUTER_TRAINING = "\n[router.training]\nk_values = [6]\n"
SPECIALIST = """
[specialists.f]
target = "F"
background = "B"

[[specialists.f.complexes]]
name = "c"
members = ["a b", "a c"]
resolve = "r"
"""


def test_a_model_with_only_sub_tables_gets_its_own_table() -> None:
    data = tomllib.loads(migrate_text(HEADER + ROUTER_TRAINING + SPECIALIST))

    assert data["models"]["router"] == {
        "kind": "bio.kmer-classifier",
        "training": {"k_values": [6]},
    }
    assert data["models"]["f"]["complexes"][0]["name"] == "c"


def test_a_model_table_after_its_sub_tables_keeps_its_place() -> None:
    text = HEADER + (
        '\n[[specialists.f.excluded]]\naccession = "A1"\nreason = "r"\n'
        '\n[specialists.f]\ntarget = "F"\nbackground = "B"\n'
    )

    result = migrate_text(text)

    assert result.count("[models.f]") == 1
    assert tomllib.loads(result)["models"]["f"]["excluded"][0]["accession"] == "A1"


def test_a_config_without_a_router_table_gets_the_default_router() -> None:
    data = tomllib.loads(migrate_text(HEADER))

    assert data["models"]["router"] == {"kind": "bio.kmer-classifier"}


def test_schema_2_is_not_rewritten() -> None:
    with pytest.raises(MigrationError, match="already schema 2"):
        migrate_text("schema = 2\n" + HEADER)


def test_invalid_files_are_refused() -> None:
    with pytest.raises(MigrationError, match="Not a valid TOML"):
        migrate_text("name = \n")
    with pytest.raises(MigrationError, match="Not a valid config"):
        migrate_text('name = "x"\n')
    with pytest.raises(MigrationError, match="Not a valid config"):
        migrate_text(HEADER + "\n[specialists]\nx = 1\n")
    with pytest.raises(MigrationError, match="rewritten config is not valid"):
        migrate_text(HEADER + '\n[router]\nkind = "bio.kmer-classifier"\n')


def test_a_rewrite_that_changes_settings_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def extra_stage(converted: dict[str, object]) -> str:
        return '\n[stages.extra]\nkind = "bio.extra"\n'

    monkeypatch.setattr(migrate, "_pipeline_text", extra_stage)

    with pytest.raises(MigrationError, match="would not give the same settings"):
        migrate_text(HEADER)


def test_the_command_prints_or_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(HEADER, "utf-8")

    assert main(["--config", str(path), "migrate-config"]) == 0
    assert "schema = 2" in capsys.readouterr().out
    assert path.read_text("utf-8") == HEADER

    assert main(["--config", str(path), "migrate-config", "--write"]) == 0
    assert "schema = 2" in path.read_text("utf-8")
    assert main(["--config", str(path), "migrate-config"]) == 1


def test_commands_warn_about_schema_1(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(HEADER, "utf-8")

    assert main(["--config", str(path), "run", "nope-anchors"]) == 1

    assert "is schema 1" in caplog.text
    assert "no stage named 'nope-anchors'" in caplog.text


def test_commands_leave_unreadable_files_to_the_loader(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "app.toml"
    path.write_text("name = \n", "utf-8")

    assert main(["--config", str(path), "run", "nope-anchors"]) == 1
    assert main(["--config", str(tmp_path / "missing.toml"), "run", "x"]) == 1
    assert "is schema 1" not in caplog.text
