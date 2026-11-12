"""Shared data for the specialist evaluation and card tests.

A tiny world: three mosquito species in training, held-out species that are
clean (4% from a known species) or leaky (near-identical to one), and
house flies as the background.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.reference import build_reference

__all__ = ["AEGYPTI", "config_dict", "prepared", "write_data"]

rng = np.random.default_rng(5)


def random_sequence(length: int = 300) -> str:
    return "".join(rng.choice(list("ACGT"), length))


def mutate(sequence: str, share: float) -> str:
    bases = list(sequence)
    for position in rng.choice(len(bases), int(len(bases) * share), replace=False):
        bases[position] = str(rng.choice([b for b in "ACGT" if b != bases[position]]))
    return "".join(bases)


AEGYPTI = random_sequence()
ALBOPICTUS = random_sequence()
PIPIENS = random_sequence()
MUSCA = random_sequence()


def records(*, mislabelled: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(count: int, split: str, species: str, base: str, share: float) -> None:
        family = "Muscidae" if species.startswith("Musca") else "Culicidae"
        for _ in range(count):
            rows.append(
                {
                    "record_id": f"r{len(rows)}",
                    "split": split,
                    "order": "Diptera",
                    "family": family,
                    "genus": species.split()[0],
                    "species": species,
                    "species_named": True,
                    "ambiguous_fraction": 0.0,
                    "sequence": mutate(base, share),
                }
            )

    add(6, "train", "Aedes aegypti", AEGYPTI, 0.003)
    add(6, "train", "Aedes albopictus", ALBOPICTUS, 0.003)
    add(6, "train", "Culex pipiens", PIPIENS, 0.003)
    add(6, "train", "Musca domestica", MUSCA, 0.003)
    for stage in ("val", "test"):
        add(5, f"{stage}_known", "Aedes aegypti", AEGYPTI, 0.005)  # 3 drawn
        add(2, f"{stage}_known", "Aedes albopictus", ALBOPICTUS, 0.005)
        # A held-out species 4% from Ae. aegypti: the genus at best
        add(2, f"{stage}_novel", "Aedes novus", AEGYPTI, 0.04)
        # A held-out species near-identical to Cx. pipiens: leaky
        add(2, f"{stage}_novel", "Culex novus", PIPIENS, 0.005)
        add(3, f"{stage}_novel", "Musca domestica", MUSCA, 0.01)
    if mislabelled:
        # Named as another species, but with Ae. aegypti's sequence
        add(1, "val_known", "Aedes mislabelled", AEGYPTI, 0.0)
    rows.append({**rows[0], "record_id": "skip1", "split": "val_known"})
    rows[-1]["ambiguous_fraction"] = 0.5
    rows.append({**rows[0], "record_id": "skip2", "split": "val_known"})
    rows[-1]["species_named"] = False
    rows.append({**rows[0], "record_id": "skip3", "split": "val_known"})
    rows[-1]["order"] = "Lepidoptera"
    rows[-1]["family"] = "Noctuidae"
    return rows


def config_dict(tmp_path: Path, **evaluation: Any) -> dict[str, Any]:
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
                "k": 6,
                "evaluation": {"workers": 1, **evaluation},
            }
        },
    }


def write_data(tmp_path: Path, *, mislabelled: bool = False) -> list[dict[str, Any]]:
    rows = records(mislabelled=mislabelled)
    folder = Path("midori2_gb999") / "gene=CO1"
    split = tmp_path / "splits" / folder / "records.parquet"
    split.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), split)
    leaky = [row["record_id"] for row in rows if row["species"] == "Culex novus"]
    leakage = tmp_path / "leakage" / folder / "near_duplicates.parquet"
    leakage.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"record_id": r} for r in leaky]), leakage)
    return rows


def prepared(
    tmp_path: Path, *, mislabelled: bool = False, **evaluation: Any
) -> AppConfig:
    write_data(tmp_path, mislabelled=mislabelled)
    config = AppConfig.model_validate(config_dict(tmp_path, **evaluation))
    build_reference(config, "culicidae")
    return config
