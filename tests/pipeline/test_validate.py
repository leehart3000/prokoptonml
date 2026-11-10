"""Tests for prokopton.ml.pipeline.validate."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig, ValidationConfig
from prokopton.ml.data.taxonomy import NO_PHYLUM, OTHER_EUKARYOTES, UNMAPPED
from prokopton.ml.pipeline.validate import (
    REASONS,
    REPORT_NAME,
    DataValidationError,
    validate,
    validate_gene,
)

GROUPS = {6656: "Metazoa", 4890: "Fungi", 2763: OTHER_EUKARYOTES}
GOOD = "ACGT" * 30  # 120 bases


def make_records(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Build a records table with the ingestion stage's columns and dtypes."""
    defaults: dict[str, object] = {
        "record_id": "id",
        "accession": "ACC1.1",
        "start": "1",
        "end": "120",
        "gene": "CO1",
        "phylum_taxid": 6656,
        "phylum": "Arthropoda",
        "order": "Diptera",
        "species": "Aedes aegypti",
        "sequence": GOOD,
    }
    frame = pd.DataFrame([{**defaults, **row} for row in rows])
    frame["length"] = frame["sequence"].str.len().astype("Int32")
    frame["phylum_taxid"] = frame["phylum_taxid"].astype("Int64")
    for column in ("sequence", "species", "order", "phylum", "accession", "gene"):
        frame[column] = frame[column].astype("string")
    return frame


RULES = ValidationConfig(
    phylum_reference=Path("unused.tsv"),
    min_length=100,
    max_length=200,
    max_ambiguous_fraction=0.05,
    short_length=150,
    top_shared_examples=5,
)


def test_adds_derived_columns() -> None:
    records = make_records(
        [
            {"phylum_taxid": 6656},
            {"phylum_taxid": 2763, "sequence": "ACGN" * 30},
            {"phylum_taxid": None, "sequence": "TTTT" * 30},
            {"phylum_taxid": 999, "sequence": "GGGG" * 30},
        ]
    )

    kept, _ = validate_gene(
        records, GROUPS, RULES.model_copy(update={"max_ambiguous_fraction": 1.0})
    )

    assert list(kept["group"]) == ["Metazoa", OTHER_EUKARYOTES, NO_PHYLUM, UNMAPPED]
    assert kept["ambiguous_fraction"].tolist() == [0.0, 0.25, 0.0, 0.0]
    assert kept["species_sharing"].tolist() == [1, 1, 1, 1]


def test_rules_remove_records_under_first_broken_rule() -> None:
    records = make_records(
        [
            {"sequence": GOOD},
            {"sequence": "ACGX" * 30},  # invalid character (and ambiguous)
            {"sequence": "ACGT" * 20},  # 80 bases: too short
            {"sequence": "ACGT" * 60},  # 240 bases: too long
            {"sequence": "ACGN" * 30},  # 25% ambiguous
        ]
    )

    kept, report = validate_gene(records, GROUPS, RULES)

    assert len(kept) == 1
    assert report.input_records == 5
    assert report.kept_records == 1
    assert report.removed == {
        "invalid_characters": 1,
        "too_short": 1,
        "too_long": 1,
        "too_ambiguous": 1,
        "order_conflict": 0,
    }
    assert list(report.removed) == list(REASONS)
    assert report.ambiguity_above[">0%"] == 2
    assert report.ambiguity_above[">5%"] == 2
    assert report.length_histogram["0-99"] == 1
    assert report.length_histogram["100-199"] == 3
    assert report.length_histogram["200-299"] == 1


def test_order_conflicts_remove_all_copies_but_missing_order_is_not_a_conflict() -> (
    None
):
    conflict = "ACCA" * 30
    records = make_records(
        [
            {"sequence": conflict, "order": "Diptera", "accession": "A1.1"},
            {"sequence": conflict, "order": "Coleoptera", "accession": "A2.1"},
            {"sequence": conflict, "order": None, "accession": "A3.1"},
            {"sequence": GOOD, "order": "Diptera"},
            {"sequence": GOOD, "order": None},
        ]
    )

    kept, report = validate_gene(records, GROUPS, RULES)

    assert report.removed["order_conflict"] == 3
    assert report.order_conflict_sequences == 1
    assert report.order_conflict_records == 3
    assert report.order_conflict_short_records == 3
    assert set(kept["sequence"]) == {GOOD}


def test_species_sharing_is_kept_and_reported() -> None:
    records = make_records(
        [
            {"species": "Anopheles gambiae", "accession": "B1.1"},
            {"species": "Anopheles coluzzii", "accession": "B2.1"},
            {"species": "Anopheles gambiae", "accession": "B3.1"},
            {"species": "Aedes aegypti", "sequence": "TTGA" * 30},
        ]
    )

    kept, report = validate_gene(records, GROUPS, RULES)

    assert len(kept) == 4
    assert kept["species_sharing"].tolist() == [2, 2, 2, 1]
    assert report.named_species_records == 4
    assert report.named_species == 3
    assert report.shared_species_sequences == 1
    assert report.shared_species_records == 3
    example = report.top_shared_examples[0]
    assert example.species_count == 2
    assert example.species == ["Anopheles coluzzii", "Anopheles gambiae"]
    assert example.records == 3
    assert example.length == 120
    assert example.accessions == ["B1.1", "B2.1", "B3.1"]
    assert report.groups == {"Metazoa": 4}


def test_placeholder_names_do_not_count_as_shared_species() -> None:
    placeholders = [
        "Chironomidae sp. BIOUG02146_D07",
        "Chironomidae sp. BIOUG02146_D08",
        "Aedes cf. vexans",
        "Culex sp.",
    ]
    records = make_records(
        [{"species": name} for name in placeholders]
        + [{"species": "Culex pipiens pallens"}]
    )

    kept, report = validate_gene(records, GROUPS, RULES)

    assert kept["species_named"].tolist() == [False, False, False, False, True]
    assert kept["species_sharing"].tolist() == [1, 1, 1, 1, 1]
    assert report.shared_species_sequences == 0
    assert report.top_shared_examples == []


def test_example_species_lists_are_capped() -> None:
    names = [
        f"Aedes species{chr(97 + i)}{chr(97 + j)}" for i in range(5) for j in range(5)
    ]
    records = make_records([{"species": name} for name in names])

    _, report = validate_gene(records, GROUPS, RULES)

    example = report.top_shared_examples[0]
    assert example.species_count == 25
    assert len(example.species) == 20


def test_no_shared_examples_when_limit_is_zero() -> None:
    records = make_records([{"species": "Aedes aegypti"}, {"species": "Aedes vexans"}])

    _, report = validate_gene(
        records, GROUPS, RULES.model_copy(update={"top_shared_examples": 0})
    )

    assert report.shared_species_sequences == 1
    assert report.top_shared_examples == []


def make_config(tmp_path: Path) -> AppConfig:
    reference = tmp_path / "phyla.tsv"
    reference.write_text(
        "phylum\tphylum_taxid\tco1_records\tall_records\tkingdom\tmajor_group\n"
        "Arthropoda\t6656\t1\t1\tMetazoa\tOpisthokonta\n",
        encoding="utf-8",
    )
    return AppConfig.model_validate(
        {
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
                "paths": {
                    "raw_dir": str(tmp_path / "raw"),
                    "interim_dir": str(tmp_path / "interim"),
                    "processed_dir": str(tmp_path / "processed"),
                    "splits_dir": str(tmp_path / "splits"),
                },
            },
            "validation": {"phylum_reference": str(reference)},
        }
    )


def test_validate_reads_and_writes_parquet(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    input_path = tmp_path / "interim" / "midori2_gb999" / "gene=CO1" / "records.parquet"
    input_path.parent.mkdir(parents=True)
    make_records([{}, {"sequence": "ACGT" * 10}]).to_parquet(input_path, index=False)

    report = validate(config)

    assert report.genes["CO1"].kept_records == 1
    output = pd.read_parquet(report.genes["CO1"].output)
    assert output["group"].tolist() == ["Metazoa"]
    saved = json.loads(
        (tmp_path / "processed" / "midori2_gb999" / REPORT_NAME).read_text("utf-8")
    )
    assert saved["genes"]["CO1"]["removed"]["too_short"] == 1
    assert saved["rules"]["min_length"] == 100


def test_validate_requires_ingested_data(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="run the ingest stage first"):
        validate(make_config(tmp_path))


def write_app_toml(tmp_path: Path, config: AppConfig) -> Path:
    """Write the test app's config as a TOML file for the CLI."""
    config_path = tmp_path / "app.toml"
    config_path.write_text(
        f"""
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
raw_dir = "{tmp_path / "raw"}"
interim_dir = "{tmp_path / "interim"}"
processed_dir = "{tmp_path / "processed"}"
splits_dir = "{tmp_path / "splits"}"

[validation]
phylum_reference = "{config.validation.phylum_reference}"
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_validate_succeeds(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    input_path = tmp_path / "interim" / "midori2_gb999" / "gene=CO1" / "records.parquet"
    input_path.parent.mkdir(parents=True)
    make_records([{}]).to_parquet(input_path, index=False)

    assert main(["validate", str(write_app_toml(tmp_path, config))]) == 0
    assert (tmp_path / "processed" / "midori2_gb999" / REPORT_NAME).exists()


def test_cli_validate_reports_missing_data(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    assert main(["validate", str(write_app_toml(tmp_path, config))]) == 1
