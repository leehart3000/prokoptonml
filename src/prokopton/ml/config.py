"""Configuration models for ProkoptonML applications.

Each application (for example dnabarcode) is described by one TOML file that is
validated against ``AppConfig``. Relative paths in the file are resolved from
the current working directory, normally the repository root.
"""

from pathlib import Path
from typing import Literal

from prokopton.core.config import ConfigModel
from pydantic import Field, field_validator

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "AppConfig",
    "DataConfig",
    "DataPathsConfig",
    "Midori2SourceConfig",
    "ValidationConfig",
]

ALLOWED_URL_SCHEMES: tuple[str, ...] = ("https://", "file://")
"""URL schemes accepted for downloads: HTTPS, plus local files for testing."""


def _check_url(value: str) -> str:
    if not value.startswith(ALLOWED_URL_SCHEMES):
        msg = f"URL must start with one of {ALLOWED_URL_SCHEMES}: {value!r}"
        raise ValueError(msg)
    return value


class Midori2SourceConfig(ConfigModel):
    """Where to get a MIDORI2 release, and which genes to ingest from it."""

    type: Literal["midori2"] = "midori2"
    release: str = Field(pattern=r"^GB\d+$")
    doi: str
    licence: str
    archive_url: str
    checksums_url: str
    archive_name: str
    member_template: str
    genes: tuple[str, ...] = Field(min_length=1)

    @field_validator("archive_url", "checksums_url")
    @classmethod
    def _validate_url(cls, value: str) -> str:
        return _check_url(value)

    @field_validator("member_template")
    @classmethod
    def _validate_member_template(cls, value: str) -> str:
        if "{gene}" not in value:
            msg = "member_template must contain '{gene}'"
            raise ValueError(msg)
        return value


class DataPathsConfig(ConfigModel):
    """Local folders for downloaded (raw), converted (interim) and validated
    (processed) data."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path


class ValidationConfig(ConfigModel):
    """Rules for the validation stage.

    ``max_ambiguous_fraction`` defaults to 1.0 (no filtering): the stage only
    reports how many records exceed each of ``ambiguity_report_thresholds``.
    The ambiguity threshold is treated as a hyperparameter and applied by the
    training stage to training data only, so evaluation still uses realistic,
    imperfect sequences.
    """

    phylum_reference: Path
    min_length: int = Field(default=100, ge=1)
    max_length: int = Field(default=2000, ge=1)
    max_ambiguous_fraction: float = Field(default=1.0, ge=0.0, le=1.0)
    ambiguity_report_thresholds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    short_length: int = Field(default=200, ge=1)
    top_shared_examples: int = Field(default=10, ge=0)


class DataConfig(ConfigModel):
    """Data settings for an application."""

    source: Midori2SourceConfig
    paths: DataPathsConfig


class AppConfig(ConfigModel):
    """Top-level configuration for a ProkoptonML application."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    data: DataConfig
    validation: ValidationConfig
