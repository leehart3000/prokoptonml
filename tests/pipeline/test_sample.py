"""Tests for prokopton.ml.pipeline.sample."""

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
from prokopton.ml.pipeline.sample import (
    REPORT_NAME,
    SampleStageError,
    label_for,
    sample,
    select_capped,
)


@pytest.mark.parametrize(
    ("group", "order", "expected"),
    [
        ("Metazoa", "Diptera", "Diptera"),
        ("Metazoa", None, None),
        ("Fungi", "Hypocreales", "Fungi"),
        ("Viridiplantae", None, "Viridiplantae"),
        ("Other eukaryotes", "Peronosporales", "Other eukaryotes"),
        ("No phylum", "Diptera", None),
        ("Unmapped", "Diptera", None),
    ],
)
def test_label_for(group: str, order: str | None, expected: str | None) -> None:
    assert label_for(group, order) == expected


def test_select_capped_covers_species_before_repeating() -> None:
    # Class 0: species 0 has 5 records, species 1 and 2 have 1 each
    classes = np.array([0, 0, 0, 0, 0, 0, 0])
    species = np.array([0, 0, 0, 0, 0, 1, 2])
    priority = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.8])

    chosen = select_capped(classes, species, priority, cap=4)

    # One from each species first (indices 0, 5, 6), then species 0 again (index 1)
    assert np.flatnonzero(chosen).tolist() == [0, 1, 5, 6]


def test_select_capped_caps_each_class_separately() -> None:
    classes = np.array([0, 0, 0, 1, 1, 2])
    species = np.array([0, 1, 2, 3, 4, 5])
    priority = np.array([0.5, 0.1, 0.9, 0.3, 0.2, 0.4])

    chosen = select_capped(classes, species, priority, cap=2)

    assert Counter(classes[chosen].tolist()) == {0: 2, 1: 2, 2: 1}
    assert not chosen[2]  # class 0's highest-priority-number record is left out


def test_select_capped_empty() -> None:
    empty = np.array([], dtype=np.int64)

    assert select_capped(empty, empty, np.array([], dtype=np.float64), cap=3).size == 0


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(
        split: str, group: str, order: str | None, count: int, **fields: Any
    ) -> None:
        for i in range(count):
            rows.append(
                {
                    "record_id": f"{split}-{group}-{order}-{len(rows)}",
                    "split": split,
                    "group": group,
                    "order": order,
                    "family": "Fam",
                    "genus": "Gen",
                    "species": fields.get("species", f"{order} species {i % 3}"),
                    "species_named": True,
                    "length": 120,
                    "ambiguous_fraction": fields.get("ambiguous_fraction", 0.0),
                    "sequence": "ACGT" * 30,
                }
            )

    add("train", "Metazoa", "Diptera", 30)
    add("train", "Metazoa", "Rare", 5)  # below min_class_records
    add("train", "Metazoa", None, 4)  # no order: not eligible
    add("train", "Fungi", "Hypocreales", 25)
    add("train", "Metazoa", "Diptera", 3, ambiguous_fraction=0.5)  # too ambiguous
    add("val_known", "Metazoa", "Diptera", 12)
    add("val_known", "Metazoa", "Rare", 3)  # dropped class
    add("val_novel", "Metazoa", "Diptera", 4)
    add("test_known", "Metazoa", "Diptera", 6)  # never sampled
    return rows


def write_inputs(tmp_path: Path, rows: list[dict[str, Any]], leaky: list[str]) -> None:
    folder = "midori2_gb999"
    split_path = tmp_path / "splits" / folder / "gene=CO1" / "records.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split_path)
    leakage_path = (
        tmp_path / "leakage" / folder / "gene=CO1" / "near_duplicates.parquet"
    )
    leakage_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{"record_id": r} for r in leaky] or [{"record_id": "none"}]
        ),
        leakage_path,
    )


def make_config(tmp_path: Path, **sample_fields: object) -> AppConfig:
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
                    "models_dir": str(tmp_path / "models"),
                },
            },
            "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
            "router": {
                "sample": {
                    "train_cap": 20,
                    "eval_cap": 10,
                    "min_class_records": 10,
                    **sample_fields,
                }
            },
        }
    )


def read_labels(path: str) -> Counter[str]:
    return Counter(pq.read_table(path).column("label").to_pylist())


def test_sample_caps_classes_and_drops_rare_ones(tmp_path: Path) -> None:
    rows = make_rows()
    leaky = [rows[-7]["record_id"]]  # one val_novel record has a near-copy
    write_inputs(tmp_path, rows, leaky)

    report = sample(make_config(tmp_path))

    assert report.labels == ["Diptera", "Fungi"]
    assert report.dropped_classes == {"Rare": 5}
    assert report.excluded_near_duplicates == 1
    assert read_labels(report.splits["train"].output) == {"Diptera": 20, "Fungi": 20}
    assert read_labels(report.splits["val_known"].output) == {"Diptera": 10}
    assert read_labels(report.splits["val_novel"].output) == {"Diptera": 3}
    assert report.splits["train"].eligible_records == 60
    assert report.splits["train"].classes_below_cap == 0
    assert report.splits["val_novel"].classes_below_cap == 1
    assert report.splits["train"].mean_species_per_class == 3.0
    saved = json.loads(
        (tmp_path / "samples" / "midori2_gb999" / "router" / REPORT_NAME).read_text(
            "utf-8"
        )
    )
    assert saved["rules"]["train_cap"] == 20


def test_near_duplicates_can_be_kept(tmp_path: Path) -> None:
    rows = make_rows()
    write_inputs(tmp_path, rows, [rows[-7]["record_id"]])

    report = sample(make_config(tmp_path, exclude_near_duplicates=False))

    assert report.excluded_near_duplicates == 0
    assert report.splits["val_novel"].sampled_records == 4


def test_sample_is_reproducible(tmp_path: Path) -> None:
    write_inputs(tmp_path, make_rows(), [])
    config = make_config(tmp_path)

    first = (
        pq.read_table(sample(config).splits["train"].output)
        .column("record_id")
        .to_pylist()
    )
    second = (
        pq.read_table(sample(config).splits["train"].output)
        .column("record_id")
        .to_pylist()
    )

    assert first == second


def test_sample_requires_split_data(tmp_path: Path) -> None:
    with pytest.raises(SampleStageError, match="run the split stage first"):
        sample(make_config(tmp_path))


def test_sample_requires_leakage_data_when_excluding(tmp_path: Path) -> None:
    split_path = tmp_path / "splits" / "midori2_gb999" / "gene=CO1" / "records.parquet"
    split_path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(make_rows()), split_path)

    with pytest.raises(SampleStageError, match="run the leakage stage first"):
        sample(make_config(tmp_path))


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
leakage_dir = "{tmp_path / "leakage"}"
samples_dir = "{tmp_path / "samples"}"
models_dir = "{tmp_path / "models"}"

[validation]
phylum_reference = "{tmp_path / "phyla.tsv"}"

[router.sample]
train_cap = 20
eval_cap = 10
min_class_records = 10
""",
        encoding="utf-8",
    )
    return config_path


def test_cli_sample_succeeds(tmp_path: Path) -> None:
    write_inputs(tmp_path, make_rows(), [])

    assert main(["sample", str(write_app_toml(tmp_path))]) == 0


def test_cli_sample_reports_missing_data(tmp_path: Path) -> None:
    assert main(["sample", str(write_app_toml(tmp_path))]) == 1


def test_sample_writes_in_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_inputs(tmp_path, make_rows(), [])
    config = make_config(tmp_path)
    whole = pq.read_table(sample(config).splits["train"].output)
    monkeypatch.setattr("prokopton.ml.pipeline.sample.BATCH_SIZE", 7)

    output = sample(config).splits["train"].output

    assert pq.read_table(output).equals(whole)
    assert pq.ParquetFile(output).num_row_groups > 1


def marker_config(tmp_path: Path, **sample_fields: object) -> AppConfig:
    settings = make_config(tmp_path).model_dump(mode="json")
    settings["marker"] = {
        "label": "gene",
        "genes": ["CO1", "Cytb"],
        "sample": {
            "diversity": "order",
            "train_cap": 10,
            "eval_cap": 5,
            "min_class_records": 5,
            "eval_splits": ["val_known"],
            **sample_fields,
        },
    }
    return AppConfig.model_validate(settings)


def write_cytb(tmp_path: Path) -> None:
    rows = [
        {
            "record_id": f"cytb-{split}-{order}-{i}",
            "split": split,
            "group": "Metazoa",
            "order": order,
            "family": "Fam",
            "genus": "Gen",
            "species": f"{order} species {i}",
            "species_named": True,
            "length": 120,
            "ambiguous_fraction": 0.0,
            "sequence": "GGCC" * 30,
        }
        for split, counts in {"train": {"Perciformes": 30, "Diptera": 3}}.items()
        for order, count in counts.items()
        for i in range(count)
    ]
    rows += [
        {**rows[0], "record_id": f"cytb-val-{i}", "split": "val_known"}
        for i in range(6)
    ]
    path = tmp_path / "splits" / "midori2_gb999" / "gene=Cytb" / "records.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_marker_sample_labels_genes_and_spreads_across_orders(tmp_path: Path) -> None:
    write_inputs(tmp_path, make_rows(), ["train-Metazoa-Diptera-0"])
    write_cytb(tmp_path)

    report = sample(marker_config(tmp_path), "marker")

    assert (report.model, report.gene) == ("marker", "CO1,Cytb")
    assert report.labels == ["CO1", "Cytb"]
    train = pq.read_table(report.splits["train"].output).to_pylist()
    cytb_orders = Counter(r["order"] for r in train if r["label"] == "Cytb")
    # Spread across orders: all 3 Diptera records are kept before more Perciformes
    assert cytb_orders == {"Diptera": 3, "Perciformes": 7}
    # CO1 records with no order are still eligible: their group is used
    co1_orders = Counter(r["order"] for r in train if r["label"] == "CO1")
    assert co1_orders[None] >= 1
    assert set(report.splits) == {"train", "val_known"}
    assert report.splits["val_known"].output.endswith("marker/val_known.parquet")
    # Only CO1 has leakage data; its near-duplicates are still left out
    assert report.excluded_near_duplicates == 1


def test_marker_sample_requires_every_gene(tmp_path: Path) -> None:
    write_inputs(tmp_path, make_rows(), [])

    with pytest.raises(SampleStageError, match="gene=Cytb"):
        sample(marker_config(tmp_path), "marker")


def test_cli_sample_for_an_app_without_that_classifier(tmp_path: Path) -> None:
    write_inputs(tmp_path, make_rows(), [])

    assert main(["sample", str(write_app_toml(tmp_path)), "--model", "marker"]) == 1
