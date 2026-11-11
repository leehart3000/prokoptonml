"""Tests for prokopton.ml.pipeline.reference."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.reference import (
    REPORT_NAME,
    ReferenceStageError,
    build_reference,
    first_per_key,
    reference_path,
)


def test_first_per_key_keeps_the_lowest_priorities_up_to_each_cap() -> None:
    keys = np.array([0, 0, 0, 1, 1, 1])
    priority = np.array([0.3, 0.1, 0.2, 0.9, 0.8, 0.7])
    caps = np.array([2, 2, 2, 1, 1, 1])

    chosen = first_per_key(keys, priority, caps)

    assert np.flatnonzero(chosen).tolist() == [1, 2, 5]
    assert (
        first_per_key(np.array([], dtype=np.int64), np.array([]), np.array([])).size
        == 0
    )


def records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(count: int, order: str, family: str, species: str, **fields: Any) -> None:
        for _ in range(count):
            rows.append(
                {
                    "record_id": f"r{len(rows)}",
                    "split": fields.get("split", "train"),
                    "order": order,
                    "family": family,
                    "genus": species.split()[0],
                    "species": species,
                    "species_named": fields.get("named", True),
                    "ambiguous_fraction": fields.get("ambiguous", 0.0),
                    "sequence": "ACGT" * 30,
                }
            )

    add(25, "Diptera", "Culicidae", "Aedes aegypti")  # capped at 20
    add(3, "Diptera", "Culicidae", "Anopheles gambiae")
    add(2, "Diptera", "Culicidae", "Culex sp. BIOUG1", named=False)
    add(25, "Diptera", "Chaoboridae", "Chaoborus flavicans")  # by species: 20
    add(5, "Diptera", "Chaoboridae", "Chaoborus crystallinus")
    add(150, "Diptera", "Muscidae", "Musca domestica")  # by family: 100
    add(10, "Lepidoptera", "Noctuidae", "Spodoptera frugiperda")  # not Diptera
    add(4, "Diptera", "Culicidae", "Aedes aegypti", split="test_known")
    add(2, "Diptera", "Culicidae", "Aedes vexans", ambiguous=0.5)
    return rows


def config_dict(tmp_path: Path) -> dict[str, Any]:
    paths = ["raw", "interim", "processed", "splits", "leakage", "samples", "models"]
    return {
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
            "paths": {f"{name}_dir": str(tmp_path / name) for name in paths},
        },
        "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
        "specialists": {
            "culicidae": {
                "target": "Culicidae",
                "background": "Diptera",
                "close_relatives": ["Chaoboridae"],
                "per_species": 20,
                "background_per_family": 100,
            }
        },
    }


def write_records(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "midori2_gb999" / "gene=CO1" / "records.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(records()), path)


def test_reference_keeps_capped_targets_and_background(tmp_path: Path) -> None:
    write_records(tmp_path)
    config = AppConfig.model_validate(config_dict(tmp_path))

    report = build_reference(config, "culicidae")

    rows = pq.read_table(reference_path(config, "culicidae")).to_pylist()
    by_species = Counter(row["species"] for row in rows)
    assert by_species == {
        "Aedes aegypti": 20,
        "Anopheles gambiae": 3,
        "Culex sp. BIOUG1": 2,
        "Chaoborus flavicans": 20,
        "Chaoborus crystallinus": 5,
        "Musca domestica": 100,
    }
    assert {row["role"] for row in rows if row["family"] == "Culicidae"} == {"target"}
    assert (report.target_records, report.target_named_species) == (25, 2)
    assert report.target_placeholder_records == 2
    assert (report.background_records, report.close_relative_records) == (125, 25)
    assert report.background_families == 2
    saved = json.loads(
        (reference_path(config, "culicidae").parent / REPORT_NAME).read_text()
    )
    assert saved["specialist"] == "culicidae"


def test_reference_is_reproducible(tmp_path: Path) -> None:
    write_records(tmp_path)
    config = AppConfig.model_validate(config_dict(tmp_path))
    build_reference(config, "culicidae")
    first = (
        pq.read_table(reference_path(config, "culicidae"))
        .column("record_id")
        .to_pylist()
    )

    build_reference(config, "culicidae")

    again = (
        pq.read_table(reference_path(config, "culicidae"))
        .column("record_id")
        .to_pylist()
    )
    assert again == first


def test_reference_requires_split_data(tmp_path: Path) -> None:
    config = AppConfig.model_validate(config_dict(tmp_path))

    with pytest.raises(ReferenceStageError, match="run the split stage first"):
        build_reference(config, "culicidae")


def write_app_toml(tmp_path: Path) -> Path:
    data = config_dict(tmp_path)
    lines = [f'name = "{data["name"]}"', f'title = "{data["title"]}"']
    for section, values in (
        ("data.source", data["data"]["source"]),
        ("data.paths", data["data"]["paths"]),
        ("validation", data["validation"]),
        ("specialists.culicidae", data["specialists"]["culicidae"]),
    ):
        lines.append(f"\n[{section}]")
        # These values are written the same way in TOML and JSON
        lines += [f"{key} = {json.dumps(value)}" for key, value in values.items()]
    path = tmp_path / "app.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_cli_reference(tmp_path: Path) -> None:
    write_records(tmp_path)
    config_path = write_app_toml(tmp_path)

    assert main(["reference", str(config_path), "--specialist", "culicidae"]) == 0
    assert main(["reference", str(config_path), "--specialist", "sharks"]) == 1


def test_reference_writes_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_records(tmp_path)
    config = AppConfig.model_validate(config_dict(tmp_path))
    build_reference(config, "culicidae")
    whole = pq.read_table(reference_path(config, "culicidae")).to_pylist()
    monkeypatch.setattr("prokopton.ml.pipeline.reference.BATCH_SIZE", 7)

    build_reference(config, "culicidae")

    assert pq.read_table(reference_path(config, "culicidae")).to_pylist() == whole
