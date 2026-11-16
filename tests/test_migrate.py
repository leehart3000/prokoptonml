"""Tests for rewriting a schema 1, 2 or 3 config file as schema 4."""

import tomllib
from pathlib import Path

import pytest

from prokopton.ml import migrate
from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.migrate import (
    MigrationError,
    _to_schema_2,
    is_schema_1,
    migrate_text,
    schema_of,
)

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
    assert data["schema"] == 4
    assert not is_schema_1(data)
    assert {"router", "marker", "specialists", "validation"}.isdisjoint(data)
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


def test_schema_4_is_not_rewritten() -> None:
    with pytest.raises(MigrationError, match="already schema 4"):
        migrate_text("schema = 4\n" + HEADER)


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
    assert "schema = 4" in capsys.readouterr().out
    assert path.read_text("utf-8") == HEADER

    assert main(["--config", str(path), "migrate-config", "--write"]) == 0
    assert "schema = 4" in path.read_text("utf-8")
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


INFERENCE = """
# How submissions are checked
[inference]
# The longest sequence
max_length = 5000
barcode_gene = "CO1"
# Consult specialists at 20%
route_threshold = 0.2
# A comment at the end of the table

# The next table's own comment
[mlflow]
tracking_uri = "sqlite:///x.db"
"""
TWO_SPECIALISTS = (
    SPECIALIST.replace(
        "[[specialists.f.complexes]]",
        'router_label = "Diptera"\n\n[[specialists.f.complexes]]',
    )
    + """
[specialists.g]
target = "G"
background = "B"
router_label = "Diptera"
"""
)


def test_inference_settings_move_with_their_comments() -> None:
    text = HEADER + '\n[marker]\nlabel = "gene"\n' + TWO_SPECIALISTS + INFERENCE
    schema_2 = _to_schema_2(text, tomllib.loads(text))
    assert schema_of(tomllib.loads(schema_2)) == 2

    result = migrate_text(schema_2)

    data = tomllib.loads(result)
    assert "inference" not in data
    assert data["serving"]["chain"] == ["input", "marker", "router", "f", "g"]
    assert data["serving"]["steps"]["input"]["max_length"] == 5000
    assert data["serving"]["steps"]["marker"]["barcode_gene"] == "CO1"
    assert data["serving"]["steps"]["g"]["route_threshold"] == 0.2
    serving = result[result.index("[serving]") :]
    assert serving.count("# Consult specialists at 20%") == 2
    assert "# The longest sequence\nmax_length = 5000" in serving
    head = result[: result.index("[serving]")]
    assert "# How submissions are checked" in head[head.rindex("[stages.") :]
    assert "# A comment at the end of the table" in head[head.rindex("[stages.") :]
    assert "# The next table's own comment\n[mlflow]" in result
    assert result.count("schema = 4") == 1


def test_schema_numbers_are_read_as_written() -> None:
    assert schema_of({"schema": 2}) == 2
    assert schema_of({"schema_version": 3}) == 3
    assert schema_of({"schema": "two"}) == 1
