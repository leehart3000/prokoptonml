"""Tests for prokopton.ml.pipeline.leakage, using a stand-in for VSEARCH."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig, LeakageConfig
from prokopton.ml.pipeline.leakage import (
    REPORT_NAME,
    LeakageStageError,
    find_near_duplicates,
    leakage,
)

FAKE_VSEARCH = (sys.executable, str(Path(__file__).with_name("fake_vsearch.py")))

A = "ACGTACGTAC"
B = "TTGACCATGG"
C = "GGGGCCCCAA"

ROWS = [
    # Novel records: A has an exact copy in training, B a near-copy, C none
    ("n1", "test_novel", A),
    ("n2", "val_novel", A),
    ("n3", "test_novel", B),
    ("n4", "test_novel", C),
    # Training records, in an order that spreads matches across chunks
    ("t1", "train", B[:-1] + "A"),
    ("t2", "train", "AAAAAAAAAA"),
    ("t3", "train", A),
    # Other splits are ignored
    ("k1", "test_known", C),
]


def write_split(path: Path, rows: list[tuple[str, str, str]] = ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [{"record_id": r, "split": s, "sequence": q} for r, s, q in rows]
    )
    pq.write_table(table, path)


def rules(**fields: object) -> LeakageConfig:
    return LeakageConfig.model_validate(
        {"vsearch_command": list(FAKE_VSEARCH), "chunk_size": 2, **fields}
    )


def test_finds_exact_and_near_copies_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    write_split(path)

    rows, report = find_near_duplicates(path, tmp_path / "work", rules())

    found = {
        row["record_id"]: (row["best_identity"], row["train_record_id"]) for row in rows
    }
    assert found == {"n1": (100.0, "t3"), "n2": (100.0, "t3"), "n3": (99.5, "t1")}
    assert report.query_records == {"val_novel": 1, "test_novel": 3}
    assert report.unique_query_sequences == 3
    assert report.near_duplicates == {"val_novel": 1, "test_novel": 2}
    assert report.near_duplicate_share == {"val_novel": 1.0, "test_novel": 0.6667}
    assert report.reference_records == 3
    assert report.chunks == 2
    assert list((tmp_path / "work").iterdir()) == []


def test_no_queries_means_no_search(tmp_path: Path) -> None:
    path = tmp_path / "records.parquet"
    write_split(path, [("t1", "train", A)])

    rows, report = find_near_duplicates(path, tmp_path / "work", rules())

    assert rows == []
    assert report.chunks == 0
    assert report.near_duplicate_share == {"val_novel": 0.0, "test_novel": 0.0}


@pytest.mark.parametrize(
    "command",
    [
        [sys.executable, "-c", "import sys; sys.exit(1)"],
        ["/nonexistent/vsearch"],
    ],
    ids=["fails", "missing"],
)
def test_vsearch_problems_raise_stage_error(tmp_path: Path, command: list[str]) -> None:
    path = tmp_path / "records.parquet"
    write_split(path)

    with pytest.raises(LeakageStageError, match="VSEARCH failed"):
        find_near_duplicates(path, tmp_path / "work", rules(vsearch_command=command))


def make_config(tmp_path: Path) -> AppConfig:
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
                    "leakage_dir": str(tmp_path / "leakage"),
                    "samples_dir": str(tmp_path / "samples"),
                },
            },
            "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
            "leakage": {"vsearch_command": list(FAKE_VSEARCH), "chunk_size": 2},
        }
    )


def split_path(tmp_path: Path) -> Path:
    return tmp_path / "splits" / "midori2_gb999" / "gene=CO1" / "records.parquet"


def test_leakage_writes_table_and_report(tmp_path: Path) -> None:
    write_split(split_path(tmp_path))

    report = leakage(make_config(tmp_path))

    table = pq.read_table(report.genes["CO1"].output)
    assert sorted(table.column("record_id").to_pylist()) == ["n1", "n2", "n3"]
    saved = json.loads(
        (tmp_path / "leakage" / "midori2_gb999" / REPORT_NAME).read_text("utf-8")
    )
    assert saved["genes"]["CO1"]["near_duplicates"]["test_novel"] == 2
    assert saved["rules"]["min_identity"] == 0.99
    assert not (tmp_path / "leakage" / "midori2_gb999" / "gene=CO1" / "_work").exists()


def test_leakage_requires_split_data(tmp_path: Path) -> None:
    with pytest.raises(LeakageStageError, match="run the split stage first"):
        leakage(make_config(tmp_path))


def write_app_toml(tmp_path: Path) -> Path:
    command = ", ".join(f'"{part}"' for part in FAKE_VSEARCH)
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
leakage_dir = "{tmp_path / "leakage"}"
samples_dir = "{tmp_path / "samples"}"

[validation]
phylum_reference = "{tmp_path / "phyla.tsv"}"

[leakage]
vsearch_command = [{command}]
chunk_size = 2
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_leakage_succeeds(tmp_path: Path) -> None:
    write_split(split_path(tmp_path))

    assert main(["leakage", str(write_app_toml(tmp_path))]) == 0


def test_cli_leakage_reports_missing_data(tmp_path: Path) -> None:
    assert main(["leakage", str(write_app_toml(tmp_path))]) == 1
