"""Tests for prokopton.ml.config, including the real app config files."""

from pathlib import Path

import pytest
from prokopton.core.config import ConfigError, load_config
from pydantic import ValidationError

from prokopton.ml.config import AppConfig, Midori2SourceConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE_FIELDS: dict[str, object] = {
    "release": "GB273",
    "doi": "10.5281/zenodo.22685701",
    "licence": "CC-BY-4.0",
    "archive_url": "https://example.org/archive.zip",
    "checksums_url": "https://example.org/release_sha256.txt",
    "archive_name": "archive.zip",
    "member_template": "FORMAT/uniq/{gene}.fasta.gz",
    "genes": ["CO1"],
}


def test_dnabarcode_config_is_valid() -> None:
    config = load_config(REPO_ROOT / "apps/dnabarcode/dnabarcode.toml", AppConfig)

    assert config.name == "dnabarcode"
    assert config.data.source.release == "GB273"
    assert config.data.source.genes[0] == "CO1"
    assert len(config.data.source.genes) == 15
    assert config.validation.min_length == 100
    assert config.validation.phylum_reference.is_file()


def test_source_config_accepts_valid_fields() -> None:
    source = Midori2SourceConfig.model_validate(SOURCE_FIELDS)

    assert source.genes == ("CO1",)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("archive_url", "http://example.org/archive.zip", "URL must start with"),
        ("checksums_url", "ftp://example.org/sums.txt", "URL must start with"),
        ("member_template", "FORMAT/uniq/CO1.fasta.gz", "must contain '{gene}'"),
        ("release", "273", "String should match pattern"),
        ("genes", [], "at least 1 item"),
    ],
)
def test_source_config_rejects_invalid_fields(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        Midori2SourceConfig.model_validate({**SOURCE_FIELDS, field: value})


def test_env_override_changes_paths(tmp_path: Path) -> None:
    environ = {"PROKOPTON_DATA__PATHS__RAW_DIR": str(tmp_path)}

    config = load_config(
        REPO_ROOT / "apps/dnabarcode/dnabarcode.toml",
        AppConfig,
        env_prefix="PROKOPTON_",
        environ=environ,
    )

    assert config.data.paths.raw_dir == tmp_path


def test_invalid_app_name_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "app.toml"
    path.write_text('name = "Bad Name"\ntitle = "x"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(path, AppConfig)
