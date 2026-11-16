"""Tests for the command line: finding the config, listing and running stages."""

import logging
import os
from pathlib import Path

import pytest

from prokopton.ml import cli
from prokopton.ml.cli import CONFIG_NAME, main
from prokopton.ml.stages import Registry, StageKind

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def app_toml(raw_dir: str = "r") -> str:
    phylum = (FIXTURES / "phylum_lineages_gb273.tsv").as_posix()
    return f"""\
schema = 2
name = "testapp"
title = "Test app"

[data.source]
release = "GB999"
doi = "10.0000/example"
licence = "CC-BY-4.0"
archive_url = "https://example.org/fake.zip"
checksums_url = "https://example.org/sums.txt"
archive_name = "fake.zip"
member_template = "x/{{gene}}.fasta.gz"
genes = ["CO1"]

[data.paths]
raw_dir = "{raw_dir}"
interim_dir = "i"
processed_dir = "p"
splits_dir = "s"
leakage_dir = "l"
samples_dir = "a"
models_dir = "m"

[validation]
phylum_reference = "{phylum}"

[models.router]
kind = "bio.kmer-classifier"

[pipeline]
stages = ["one", "two"]

[stages.one]
kind = "test.ok"

[stages.two]
kind = "test.fails"
"""


def fails(_: object) -> str:
    raise KeyError("this stage failed")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / CONFIG_NAME).write_text(app_toml(), "utf-8")
    monkeypatch.chdir(tmp_path)
    kinds = [
        StageKind("test.ok", "Works.", lambda c: f"ran {c.name}"),
        StageKind("test.fails", "Fails.", fails, errors=(KeyError,)),
    ]
    monkeypatch.setattr(cli, "registry", lambda: Registry(kinds))
    return tmp_path


def test_stages_are_listed_with_their_last_run(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["stages"]) == 0
    assert "  1  one" in capsys.readouterr().out.replace("[never run]", "")

    assert main(["run", "one"]) == 0
    assert main(["stages"]) == 0
    listing = capsys.readouterr().out
    assert "last run" in listing
    assert "settings changed" not in listing

    (project / CONFIG_NAME).write_text(app_toml(raw_dir="raw"), "utf-8")
    assert main(["stages"]) == 0
    assert "settings changed since" in capsys.readouterr().out


def test_runs_stop_at_a_failure(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    assert main(["run", "--all"]) == 1
    assert "this stage failed" in caplog.text
    assert main(["run", "--from", "one", "--all"]) == 1
    assert main(["run", "--from", "two"]) == 1
    assert main(["run", "one", "one"]) == 0
    assert "ran 2 stages" in caplog.text


def test_the_config_is_found_in_a_parent_folder(
    project: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    (project / "deeper").mkdir()
    os.chdir(project / "deeper")

    assert main(["stages"]) == 0

    assert Path.cwd() == project
    assert "working folder" in caplog.text


def test_a_missing_config_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["stages"]) == 1
    assert f"No {CONFIG_NAME}" in caplog.text


def test_a_given_config_is_used_as_it_is(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = project / "elsewhere"
    other.mkdir()
    (other / "app.toml").write_text(app_toml(), "utf-8")
    monkeypatch.chdir(other)

    assert main(["--config", "app.toml", "run", "one"]) == 0
    assert Path.cwd() == other
    assert (other / "m" / "testapp" / "stages" / "one.json").exists()
