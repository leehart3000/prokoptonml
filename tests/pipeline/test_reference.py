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
    CONTRADICTED_NAME,
    REPORT_NAME,
    SERVING_SPLITS,
    ReferenceStageError,
    build_reference,
    first_per_key,
    reference_path,
    serving_reference_path,
)
from prokopton.ml.specialists.identify import NEIGHBOURS_NAME


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


rng = np.random.default_rng(11)


def records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(count: int, order: str, family: str, species: str, **fields: Any) -> None:
        for _ in range(count):
            sequence = fields.get("sequence") or "".join(rng.choice(list("ACGT"), 200))
            rows.append(
                {
                    "record_id": f"r{len(rows)}",
                    "split": fields.get("split", "train"),
                    "order": order,
                    "family": family,
                    "genus": fields.get("genus", species.split()[0]),
                    "species": species,
                    "species_named": fields.get("named", True),
                    "ambiguous_fraction": fields.get("ambiguous", 0.0),
                    "sequence": sequence,
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
    add(3, "Diptera", "Culicidae", "Anopheles gambiae", sequence="ACGT" * 50)
    # A placeholder without a genus is left out
    add(2, "Diptera", "Culicidae", "Culicidae sp. BOLD1", named=False, genus=None)
    return rows


TEST_RECORD_IDS = {f"r{n}" for n in range(220, 224)}
SERVING_SPLITS_SORTED = tuple(sorted(SERVING_SPLITS))


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
                # Random test sequences have no real neighbours to vote
                "curate": False,
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
        "Anopheles gambiae": 4,
        "Culex sp. BIOUG1": 2,
        "Chaoborus flavicans": 20,
        "Chaoborus crystallinus": 5,
        "Musca domestica": 100,
    }
    assert {row["role"] for row in rows if row["family"] == "Culicidae"} == {"target"}
    assert (report.target_records, report.target_named_species) == (26, 2)
    assert report.target_placeholder_records == 2
    assert (report.background_records, report.close_relative_records) == (125, 25)
    assert report.background_families == 2
    # Three copies of one sequence: one kept, two skipped
    assert report.duplicates_skipped == 2
    assert report.species_with_neighbours == 0
    assert (reference_path(config, "culicidae").parent / NEIGHBOURS_NAME).exists()
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


def test_serving_reference_uses_every_split(tmp_path: Path) -> None:
    write_records(tmp_path)
    config = AppConfig.model_validate(config_dict(tmp_path))

    report = build_reference(config, "culicidae", serving=True)

    assert report.splits == SERVING_SPLITS_SORTED
    output = serving_reference_path(config, "culicidae")
    assert report.output == str(output)
    rows = pq.read_table(output).to_pylist()
    # The 4 test records of Ae. aegypti compete for its 20 places
    assert {r["record_id"] for r in rows} & TEST_RECORD_IDS
    assert not reference_path(config, "culicidae").exists()


def test_serving_can_keep_more_per_species(tmp_path: Path) -> None:
    write_records(tmp_path)
    data = config_dict(tmp_path)
    data["specialists"]["culicidae"]["serving_per_species"] = 25
    config = AppConfig.model_validate(data)

    build_reference(config, "culicidae", serving=True)

    rows = pq.read_table(serving_reference_path(config, "culicidae")).to_pylist()
    # 25 of the 29 Ae. aegypti records, instead of 20
    assert sum(r["species"] == "Aedes aegypti" for r in rows) == 25


def test_reference_requires_split_data(tmp_path: Path) -> None:
    config = AppConfig.model_validate(config_dict(tmp_path))

    with pytest.raises(ReferenceStageError, match="run the split stage first"):
        build_reference(config, "culicidae")


def test_reference_needs_target_records(tmp_path: Path) -> None:
    write_records(tmp_path)
    data = config_dict(tmp_path)
    data["specialists"]["culicidae"]["target"] = "Simuliidae"
    config = AppConfig.model_validate(data)

    with pytest.raises(ReferenceStageError, match="No records of Simuliidae"):
        build_reference(config, "culicidae")
    assert not reference_path(config, "culicidae").exists()


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


def test_reference_leaves_out_records_that_look_like_another_species(
    tmp_path: Path,
) -> None:
    aegypti, pipiens = ("".join(rng.choice(list("ACGT"), 300)) for _ in range(2))

    def like(base: str) -> str:
        bases = list(base)
        for position in rng.choice(len(bases), 1, replace=False):
            bases[position] = "A" if bases[position] != "A" else "C"
        return "".join(bases)

    rows = [
        {
            "record_id": f"r{n}",
            "split": "train",
            "order": "Diptera",
            "family": "Culicidae",
            "genus": species.split()[0],
            "species": species,
            "species_named": True,
            "ambiguous_fraction": 0.0,
            "sequence": like(base) if n < 24 else base,
        }
        for n, (species, base) in enumerate(
            [("Aedes aegypti", aegypti)] * 12
            + [("Culex pipiens", pipiens)] * 12
            # Labelled Cx. pipiens, but exactly an Ae. aegypti sequence
            + [("Culex pipiens", aegypti)]
        )
    ]
    path = tmp_path / "splits" / "midori2_gb999" / "gene=CO1" / "records.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    data = config_dict(tmp_path)
    data["specialists"]["culicidae"]["curate"] = True
    config = AppConfig.model_validate(data)

    report = build_reference(config, "culicidae")

    assert report.contradicted_skipped == 1
    assert report.target_records == 24
    kept = pq.read_table(reference_path(config, "culicidae")).column("record_id")
    assert "r24" not in kept.to_pylist()
    listed = json.loads(
        (reference_path(config, "culicidae").parent / CONTRADICTED_NAME).read_text()
    )
    assert listed[0]["record_id"] == "r24"
    assert listed[0]["looks_like"] == "Aedes aegypti"


def test_reference_leaves_out_records_listed_by_hand(tmp_path: Path) -> None:
    write_records(tmp_path)
    data = config_dict(tmp_path)
    data["specialists"]["culicidae"]["excluded"] = [
        {"accession": "r0", "reason": "A test of the list"}
    ]
    config = AppConfig.model_validate(data)

    build_reference(config, "culicidae")

    kept = pq.read_table(reference_path(config, "culicidae")).column("record_id")
    assert "r0" not in kept.to_pylist()
