"""Tests for rewriting a schema 3 config file as schema 4."""

import tomllib
from pathlib import Path

import pytest

from prokopton.ml import migrate
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.migrate import MigrationError, migrate_text, schema_of

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_the_dnabarcode_config_keeps_its_settings_and_comments() -> None:
    text = (FIXTURES / "dnabarcode-schema-3.toml").read_text("utf-8")

    result = migrate_text(text)

    data = tomllib.loads(result)
    assert data["schema"] == 4
    assert {"validation", "split", "leakage"}.isdisjoint(data)
    assert data["data"] == {"release": "GB273"}
    assert data["paths"] == {"models_dir": "data/models/dnabarcode"}
    assert set(data["bio"]) == {"source", "paths", "validation", "split", "leakage"}
    assert "release" not in data["bio"]["source"]
    assert "models_dir" not in data["bio"]["paths"]
    assert data["models"]["culicidae"]["kind"] == "bio.specialist"
    assert len(data["pipeline"]["stages"]) == 19
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    assert all(line in result for line in comments)
    assert AppConfig.model_validate(data) == AppConfig.model_validate(
        tomllib.loads(text)
    )
    with (FIXTURES / "dnabarcode.toml").open("rb") as handle:
        assert AppConfig.model_validate(data) == AppConfig.model_validate(
            tomllib.load(handle)
        )


SCHEMA_3 = """\
# An app
schema = 3
name = "testapp"
title = "Test app"

# Where the data comes from
[data.source]
release = "GB999" # This release
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

[split]
salt = "s"

[models.router]
kind = "bio.kmer-classifier"
"""


def test_schema_3_tables_move_into_the_bio_section() -> None:
    result = migrate_text(SCHEMA_3)

    data = tomllib.loads(result)
    assert data["schema"] == 4
    assert data["data"] == {"release": "GB999"}
    assert data["paths"] == {"models_dir": "m"}
    assert data["bio"]["source"]["doi"] == "10.0000/example"
    assert data["bio"]["paths"]["raw_dir"] == "r"
    assert data["bio"]["split"] == {"salt": "s"}
    assert data["models"]["router"] == {"kind": "bio.kmer-classifier"}
    # The new tables come before the first one, after the comments above it
    assert result.index("[data]") < result.index("# Where the data comes from")
    assert 'release = "GB999" # This release' in result
    assert "\n[bio.source]\n" in result


def test_schema_4_is_not_rewritten() -> None:
    with pytest.raises(MigrationError, match="already schema 4"):
        migrate_text(SCHEMA_3.replace("schema = 3", "schema = 4"))


@pytest.mark.parametrize("schema", ["", "schema = 2\n"])
def test_schemas_1_and_2_are_no_longer_rewritten(schema: str) -> None:
    text = schema + SCHEMA_3.replace("schema = 3\n", "")

    with pytest.raises(MigrationError, match=r"with prokopton-ml 0\.2\.0a7"):
        migrate_text(text)


def test_invalid_files_are_refused() -> None:
    with pytest.raises(MigrationError, match="Not a valid TOML"):
        migrate_text("name = \n")
    with pytest.raises(MigrationError, match="Not a valid config"):
        migrate_text('schema = 3\nname = "x"\n')
    with pytest.raises(MigrationError, match="Not a valid config"):
        migrate_text(SCHEMA_3 + "\n[sharks]\nx = 1\n")


@pytest.mark.parametrize(
    ("added", "message"),
    [
        ('\n[stages.extra]\nkind = "bio.extra"\n', "would not give the same"),
        ("\nname = \n", "rewritten config is not valid"),
    ],
)
def test_a_rewrite_that_changes_settings_is_refused(
    monkeypatch: pytest.MonkeyPatch, added: str, message: str
) -> None:
    rewrite = migrate._to_schema_4
    monkeypatch.setattr(migrate, "_to_schema_4", lambda text: rewrite(text) + added)

    with pytest.raises(MigrationError, match=message):
        migrate_text(SCHEMA_3)


def test_the_command_prints_or_writes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(SCHEMA_3, "utf-8")

    assert main(["--config", str(path), "migrate-config"]) == 0
    assert "schema = 4" in capsys.readouterr().out
    assert path.read_text("utf-8") == SCHEMA_3

    assert main(["--config", str(path), "migrate-config", "--write"]) == 0
    assert "schema = 4" in path.read_text("utf-8")
    assert main(["--config", str(path), "migrate-config"]) == 1


def test_commands_warn_about_schema_3(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(SCHEMA_3, "utf-8")

    assert main(["--config", str(path), "run", "nope-anchors"]) == 1

    assert "is schema 3" in caplog.text
    assert "no stage named 'nope-anchors'" in caplog.text


def test_commands_refuse_schemas_1_and_2(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "app.toml"
    path.write_text(SCHEMA_3.replace("schema = 3\n", ""), "utf-8")

    assert main(["--config", str(path), "stages"]) == 1

    assert "schema 1, which is no longer read" in caplog.text


def test_commands_leave_unreadable_files_to_the_loader(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "app.toml"
    path.write_text("name = \n", "utf-8")

    assert main(["--config", str(path), "run", "nope-anchors"]) == 1
    assert main(["--config", str(tmp_path / "missing.toml"), "run", "x"]) == 1
    assert "is schema" not in caplog.text


def test_schema_numbers_are_read_as_written() -> None:
    assert schema_of({"schema": 2}) == 2
    assert schema_of({"schema_version": 3}) == 3
    assert schema_of({"schema": "two"}) == 1
