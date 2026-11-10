"""Configuration models for ProkoptonML applications.

Each application (for example dnabarcode) is described by one TOML file that is
validated against ``AppConfig``. Relative paths in the file are resolved from
the current working directory, normally the repository root.
"""

from pathlib import Path
from typing import Literal

from prokopton.core.config import ConfigModel
from pydantic import Field, field_validator, model_validator

__all__ = [
    "ALLOWED_URL_SCHEMES",
    "AppConfig",
    "DataConfig",
    "DataPathsConfig",
    "LeakageConfig",
    "Midori2SourceConfig",
    "RouterConfig",
    "RouterSampleConfig",
    "SplitConfig",
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
    splits_dir: Path
    leakage_dir: Path
    samples_dir: Path


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


class SplitConfig(ConfigModel):
    """How records are assigned to training, validation and test sets.

    Assignment uses a hash of each record's key (the species name for novel
    holdouts, the sequence otherwise) mixed with ``salt``, so it is reproducible
    and stable when data is added. Changing ``salt`` reshuffles every split.
    """

    salt: str = Field(default="prokopton-v1", min_length=1)
    novel_val_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    novel_test_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    val_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    test_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    novel_species_genes: tuple[str, ...] = ("CO1",)
    report_families: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_fractions(self) -> "SplitConfig":
        if self.novel_val_fraction + self.novel_test_fraction > 1.0:
            msg = "novel_val_fraction + novel_test_fraction must not exceed 1"
            raise ValueError(msg)
        if self.val_fraction + self.test_fraction >= 1.0:
            msg = "val_fraction + test_fraction must be below 1 (train needs data)"
            raise ValueError(msg)
        return self


DEFAULT_VSEARCH_COMMAND: tuple[str, ...] = (
    "pixi",
    "exec",
    "--channel",
    "conda-forge",
    "--channel",
    "bioconda",
    "--spec",
    "vsearch>=2.28,<3",
    "vsearch",
)
"""Runs VSEARCH from bioconda in a temporary environment, without changing the
project's own environment."""


class LeakageConfig(ConfigModel):
    """How the leakage stage finds near-duplicates of training sequences.

    Records in ``query_splits`` with a match of at least ``min_identity`` in
    ``reference_split`` are listed, so evaluation can exclude them from
    novel-species scores. The default VSEARCH settings stop at the first
    match and give up after a few non-matching candidates, which found about
    95% of the leaks found by a thorough search in about a third of the time
    (see ``analysis/README.md``).
    """

    genes: tuple[str, ...] = ("CO1",)
    query_splits: tuple[str, ...] = ("val_novel", "test_novel")
    reference_split: str = "train"
    min_identity: float = Field(default=0.99, gt=0.0, le=1.0)
    maxaccepts: int = Field(default=1, ge=1)
    maxrejects: int = Field(default=8, ge=1)
    threads: int = Field(default=8, ge=1)
    chunk_size: int = Field(default=250_000, ge=1)
    vsearch_command: tuple[str, ...] = Field(
        default=DEFAULT_VSEARCH_COMMAND, min_length=1
    )


class RouterSampleConfig(ConfigModel):
    """How the router's training and validation samples are drawn.

    Each class (an animal order, or a group for non-animals) is capped, so a
    few huge orders do not dominate. Within a class, records are taken from as
    many different species as possible. ``max_ambiguous_fraction`` should be
    the loosest value that training will try; stricter thresholds are applied
    later, when training.
    """

    salt: str = Field(default="prokopton-sample-v1", min_length=1)
    train_cap: int = Field(default=2000, ge=1)
    eval_cap: int = Field(default=200, ge=1)
    min_class_records: int = Field(default=20, ge=1)
    max_ambiguous_fraction: float = Field(default=0.10, ge=0.0, le=1.0)
    eval_splits: tuple[str, ...] = ("val_known", "val_novel")
    exclude_near_duplicates: bool = True


class RouterConfig(ConfigModel):
    """Settings for the router: the model that predicts group and order."""

    gene: str = "CO1"
    sample: RouterSampleConfig = Field(default_factory=RouterSampleConfig)


class AppConfig(ConfigModel):
    """Top-level configuration for a ProkoptonML application."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    data: DataConfig
    validation: ValidationConfig
    split: SplitConfig = Field(default_factory=SplitConfig)
    leakage: LeakageConfig = Field(default_factory=LeakageConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
