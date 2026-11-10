"""Tests for prokopton.ml.pipeline.split."""

import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig, SplitConfig
from prokopton.ml.pipeline.split import (
    REPORT_NAME,
    SPLITS,
    SplitStageError,
    SplitSummary,
    assign_splits,
    find_novel_sequences,
    hash_fraction,
    hash_key,
    split,
)

ALL_NOVEL_TEST = SplitConfig(
    novel_val_fraction=0.0, novel_test_fraction=1.0, val_fraction=0.0, test_fraction=0.0
)


def random_sequence(rng: random.Random) -> str:
    return "".join(rng.choice("ACGT") for _ in range(60))


def run(
    species: list[str], named: list[bool], sequences: list[str], rules: SplitConfig
) -> list[str]:
    """Run both passes on one batch."""
    novel = find_novel_sequences([(species, named, sequences)], rules)
    return assign_splits(sequences, novel, rules)


def test_hash_fraction_is_stable_and_in_range() -> None:
    values = [hash_fraction(f"key{i}", "salt") for i in range(1000)]

    assert all(0.0 <= value < 1.0 for value in values)
    assert hash_fraction("key1", "salt") == values[1]
    assert hash_fraction("key1", "other salt") != values[1]
    assert hash_key("key1", "salt") == hash_key("key1", "salt")
    assert 0.45 < sum(values) / len(values) < 0.55


@pytest.mark.parametrize(
    "fields",
    [
        {"novel_val_fraction": 0.6, "novel_test_fraction": 0.5},
        {"val_fraction": 0.5, "test_fraction": 0.5},
    ],
)
def test_split_config_rejects_impossible_fractions(fields: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate(fields)


def test_named_species_held_out_and_identical_sequences_follow() -> None:
    shared = "TTGA" * 30

    result = run(
        ["Aedes vexans", "Aedes sp.", "Culex sp."],
        [True, False, False],
        [shared, shared, "GGCC" * 30],
        ALL_NOVEL_TEST,
    )

    assert result == ["test_novel", "test_novel", "train"]


def test_test_novel_wins_when_a_sequence_is_in_both_novel_splits() -> None:
    rules = SplitConfig(novel_val_fraction=0.5, novel_test_fraction=0.5)
    names = [
        f"Aedes species{chr(97 + i)}{chr(97 + j)}" for i in range(6) for j in range(6)
    ]

    result = run(names, [True] * len(names), ["ACGT" * 30] * len(names), rules)

    assert set(result) == {"test_novel"}


def test_novel_sequences_are_found_across_batches() -> None:
    shared = "TTGA" * 30
    batches = [
        (["Aedes vexans"], [True], [shared]),
        (["Aedes sp."], [False], [shared]),
    ]

    novel = find_novel_sequences(batches, ALL_NOVEL_TEST)

    assert assign_splits([shared, "GGCC" * 30], novel, ALL_NOVEL_TEST) == [
        "test_novel",
        "train",
    ]


def test_unnamed_or_missing_species_are_never_held_out_by_name() -> None:
    sequences = ["ACGT" * 30, "TTGA" * 30]

    novel = find_novel_sequences(
        [(["Aedes sp.", None], [False, None], sequences)], ALL_NOVEL_TEST
    )

    assert novel == {}


def test_known_split_proportions_and_identical_sequences_together() -> None:
    rng = random.Random(2)
    sequences = [random_sequence(rng) for _ in range(5000)]
    sequences.append(sequences[0])

    result = assign_splits(sequences, {}, SplitConfig())
    shares = {name: result.count(name) / len(result) for name in SPLITS}

    assert 0.77 < shares["train"] < 0.83
    assert 0.08 < shares["val_known"] < 0.12
    assert 0.08 < shares["test_known"] < 0.12
    assert result[0] == result[-1]


def test_assignment_is_reproducible_and_salt_dependent() -> None:
    rng = random.Random(3)
    species = [f"Aedes s{i}x" for i in range(500)]
    sequences = [random_sequence(rng) for _ in species]
    named = [True] * len(species)

    first = run(species, named, sequences, SplitConfig(salt="a"))
    second = run(species, named, sequences, SplitConfig(salt="a"))
    other = run(species, named, sequences, SplitConfig(salt="b"))

    assert first == second
    assert first != other
    assert {"val_novel", "test_novel", "train"} <= set(first)


def test_summary_counts_splits_groups_and_families() -> None:
    rules = ALL_NOVEL_TEST.model_copy(update={"report_families": ("Culicidae",)})
    shared = "TTGA" * 30
    species = ["Aedes vexans", "Aedes sp.", "Musca sp.", "Fungus sp."]
    named = [True, False, False, False]
    sequences = [shared, shared, "GGCC" * 30, "CCAA" * 30]
    groups = ["Metazoa", "Metazoa", "Metazoa", "Fungi"]
    families = ["Culicidae", "Culicidae", "Muscidae", None]
    splits = run(species, named, sequences, rules)
    summary = SplitSummary(rules)

    summary.update(splits[:2], species[:2], named[:2], groups[:2], families[:2])
    summary.update(splits[2:], species[2:], named[2:], [None, "Fungi"], families[2:])
    report = summary.report()

    assert report.records == {
        "train": 2,
        "val_known": 0,
        "test_known": 0,
        "val_novel": 0,
        "test_novel": 2,
    }
    assert report.named_species["test_novel"] == 1
    assert report.moved_by_identical_sequence == 1
    assert report.groups["Fungi"]["train"] == 1
    assert report.groups["Unknown"]["train"] == 1
    assert report.groups["Metazoa"]["test_novel"] == 2
    assert report.families["Culicidae"]["records"]["test_novel"] == 2
    assert report.families["Culicidae"]["named_species"]["test_novel"] == 1


def make_config(
    tmp_path: Path, genes: tuple[str, ...] = ("CO1",), **split_fields: object
) -> AppConfig:
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
                    "genes": list(genes),
                },
                "paths": {
                    "raw_dir": str(tmp_path / "raw"),
                    "interim_dir": str(tmp_path / "interim"),
                    "processed_dir": str(tmp_path / "processed"),
                    "splits_dir": str(tmp_path / "splits"),
                },
            },
            "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
            "split": {"report_families": ["Culicidae"], **split_fields},
        }
    )


def write_processed(tmp_path: Path, gene: str = "CO1") -> None:
    rng = random.Random(4)
    path = tmp_path / "processed" / "midori2_gb999" / f"gene={gene}" / "records.parquet"
    path.parent.mkdir(parents=True)
    rows = [
        {
            "record_id": f"id{i}",
            "species": f"Aedes s{i}x",
            "species_named": True,
            "family": "Culicidae",
            "group": "Metazoa",
            "sequence": random_sequence(rng),
        }
        for i in range(250)
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_split_reads_and_writes_parquet(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_processed(tmp_path)

    report = split(config)

    output = pq.read_table(report.genes["CO1"].output)
    assert output.num_rows == 250
    assert output.column_names[-1] == "split"
    assert output.column("record_id").to_pylist()[:2] == ["id0", "id1"]
    assert set(output.column("split").to_pylist()) <= set(SPLITS)
    saved = json.loads(
        (tmp_path / "splits" / "midori2_gb999" / REPORT_NAME).read_text("utf-8")
    )
    assert sum(saved["genes"]["CO1"]["records"].values()) == 250
    assert saved["rules"]["salt"] == "prokopton-v1"


def test_genes_without_novel_holdout_only_get_known_splits(tmp_path: Path) -> None:
    config = make_config(
        tmp_path, genes=("Cytb",), novel_val_fraction=0.5, novel_test_fraction=0.5
    )
    write_processed(tmp_path, gene="Cytb")

    report = split(config)

    assert report.genes["Cytb"].records["val_novel"] == 0
    assert report.genes["Cytb"].records["test_novel"] == 0
    assert report.genes["Cytb"].records["train"] > 0


def test_split_requires_validated_data(tmp_path: Path) -> None:
    with pytest.raises(SplitStageError, match="run the validate stage first"):
        split(make_config(tmp_path))


def write_app_toml(tmp_path: Path) -> Path:
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
phylum_reference = "{tmp_path / "phyla.tsv"}"
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_split_succeeds(tmp_path: Path) -> None:
    write_processed(tmp_path)

    assert main(["split", str(write_app_toml(tmp_path))]) == 0


def test_cli_split_reports_missing_data(tmp_path: Path) -> None:
    assert main(["split", str(write_app_toml(tmp_path))]) == 1
