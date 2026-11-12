"""Reference stage: build a specialist's reference set.

A specialist identifies sequences by comparing them with reference
sequences, so its "training" is choosing those references well. From the
training split of the specialist's gene (never validation or test), it
keeps:

- **target** records (for example, family Culicidae): up to ``per_species``
  per species, including placeholder names such as "Aedes sp. BIOUG...",
  which are reported as records labelled only to genus (placeholders
  without a genus are left out if ``placeholders_need_genus`` is on);
- **background** records from the wider group (for example, order Diptera),
  so the specialist can tell when a sequence is not in its group: up to
  ``per_species`` per species for close relatives, and up to
  ``background_per_family`` per family otherwise.

Identical sequences within a species (or family) are kept once, so copies
of one haplotype do not use up its places. Within each species or family,
records are then chosen by a salted hash, so the choice is reproducible.
The data is streamed twice (to choose, then to write).

Finally, each named target species' nearest other named species is found
(its barcode gap in our data; see ``Specialist.find_neighbours``).

Outputs ``<models_dir>/specialists/<name>/reference.parquet``,
``neighbours.json`` and ``_reference_report.json``.
"""

import hashlib
import json
import logging
from array import array
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, SpecialistConfig
from prokopton.ml.pipeline.split import hash_fraction
from prokopton.ml.specialists.identify import (
    NEIGHBOURS_NAME,
    Specialist,
    write_neighbours,
)

__all__ = [
    "CONTRADICTED_NAME",
    "REFERENCE_NAME",
    "REFERENCE_SCHEMA",
    "REPORT_NAME",
    "SERVING_SPLITS",
    "ReferenceReport",
    "ReferenceStageError",
    "build_reference",
    "first_per_key",
    "reference_path",
    "serving_reference_path",
]

logger = logging.getLogger(__name__)

REFERENCE_NAME = "reference.parquet"
REPORT_NAME = "_reference_report.json"
CONTRADICTED_NAME = "_contradicted.json"
SERVING_SPLITS = ("train", "val_known", "val_novel", "test_known", "test_novel")
"""The splits a serving reference set is drawn from (all of them)."""
BATCH_SIZE = 100_000
_COLUMNS = [
    "record_id",
    "split",
    "order",
    "family",
    "genus",
    "species",
    "species_named",
    "ambiguous_fraction",
    "sequence",
]
REFERENCE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("role", pa.string()),
        ("order", pa.string()),
        ("family", pa.string()),
        ("genus", pa.string()),
        ("species", pa.string()),
        ("species_named", pa.bool_()),
        ("sequence", pa.string()),
    ]
)


class ReferenceStageError(RuntimeError):
    """Raised when the reference stage cannot run."""


class ReferenceReport(BaseModel):
    """Provenance and contents of a specialist's reference set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app: str
    specialist: str
    release: str
    gene: str
    prokopton_ml_version: str
    rules: SpecialistConfig
    started_at: datetime
    finished_at: datetime
    target_records: int
    target_species: int
    target_named_species: int
    target_placeholder_records: int
    background_records: int
    background_families: int
    close_relative_records: int
    duplicates_skipped: int
    contradicted_skipped: int = 0
    """Named references left out because they look like another species
    (listed in ``_contradicted.json``)."""
    species_with_neighbours: int
    species_within_species_identity: int
    """Named species whose nearest other species is at least
    ``species_identity`` identical: the marker cannot separate them well."""
    output: str
    splits: tuple[str, ...] = ("train",)


def reference_path(config: AppConfig, name: str) -> Path:
    """Where a specialist's reference set (for evaluation) is written."""
    return config.data.paths.models_dir / "specialists" / name / REFERENCE_NAME


def serving_reference_path(config: AppConfig, name: str) -> Path:
    """Where a specialist's serving reference set (all splits) is written."""
    folder = config.data.paths.models_dir / "specialists" / name / "serving"
    return folder / REFERENCE_NAME


def first_per_key(
    keys: np.ndarray, priority: np.ndarray, caps: np.ndarray
) -> np.ndarray:
    """Choose, within each key, the records with the lowest priority numbers.

    Args:
        keys: A key id for each record (such as its species).
        priority: Tie-breaking number for each record (lower is chosen first).
        caps: The most records to keep for each record's key.

    Returns:
        A boolean mask of the chosen records.
    """
    count = keys.size
    if count == 0:
        return np.zeros(0, dtype=bool)
    order = np.lexsort((priority, keys))
    sorted_keys = keys[order]
    new_key = np.ones(count, dtype=bool)
    new_key[1:] = sorted_keys[1:] != sorted_keys[:-1]
    start = np.maximum.accumulate(np.where(new_key, np.arange(count), 0))
    chosen = np.zeros(count, dtype=bool)
    chosen[order[(np.arange(count) - start) < caps[order]]] = True
    return chosen


def _role(row: dict[str, Any], rules: SpecialistConfig) -> str | None:
    if row[rules.target_rank] == rules.target:
        return "target"
    if row[rules.background_rank] == rules.background:
        return "background"
    return None


def _rows(
    path: Path, rules: SpecialistConfig, splits: frozenset[str]
) -> Iterator[dict[str, Any]]:
    """Yield the rows of ``splits`` that can be references, with their role.

    Records listed in ``excluded`` (by accession) are left out.
    """
    excluded = {record.accession for record in rules.excluded}
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=BATCH_SIZE, columns=_COLUMNS
    ):
        for row in batch.to_pylist():
            if row["split"] not in splits:
                continue
            # Record IDs start with the accession and its version
            if ".".join(row["record_id"].split(".")[:2]) in excluded:
                continue
            if row["ambiguous_fraction"] > rules.max_ambiguous_fraction:
                continue
            role = _role(row, rules)
            if (
                role == "target"
                and rules.placeholders_need_genus
                and not row["species_named"]
                and not row["genus"]
            ):
                continue
            if role is not None:
                row["role"] = role
                yield row


def build_reference(
    config: AppConfig, name: str, *, serving: bool = False
) -> ReferenceReport:
    """Choose and write a specialist's reference set.

    The reference set for evaluation uses the training split only, so the
    validation and test records stay unseen. With ``serving``, every split
    is used instead (more species and more variation for real queries), and
    the set is written to ``serving_reference_path``.

    Raises:
        ReferenceStageError: If the split data is missing.
    """
    started_at = datetime.now(UTC)
    rules = config.specialist(name)
    folder = f"midori2_{config.data.source.release.lower()}"
    path = (
        config.data.paths.splits_dir / folder / f"gene={rules.gene}" / "records.parquet"
    )
    if not path.exists():
        msg = f"{path} not found; run the split stage first"
        raise ReferenceStageError(msg)

    # Pass 1: give each record a key (species, or family for most of the
    # background)
    # and that key's cap. Repeated sequences within a key get key -1 and
    # cap 0, so they are never chosen and take no one's place
    key_ids: dict[tuple[str, str], int] = {}
    seen: set[tuple[int, bytes]] = set()
    duplicates = 0
    keys = array("q")
    caps = array("q")
    priority = array("d")
    splits = frozenset(SERVING_SPLITS if serving else ("train",))
    per_species = (
        rules.serving_per_species or rules.per_species if serving else rules.per_species
    )
    for row in _rows(path, rules, splits):
        by_species = row["role"] == "target" or row["family"] in rules.close_relatives
        key = (
            ("species", row["species"] or "")
            if by_species
            else ("family", row["family"] or "")
        )
        key_id = key_ids.setdefault(key, len(key_ids))
        digest = hashlib.blake2b(
            row["sequence"].upper().encode("ascii"), digest_size=12
        ).digest()
        if (key_id, digest) in seen:
            duplicates += 1
            keys.append(-1)
            caps.append(0)
            priority.append(0.0)
            continue
        seen.add((key_id, digest))
        keys.append(key_id)
        caps.append(per_species if by_species else rules.background_per_family)
        priority.append(hash_fraction(row["record_id"], rules.salt))
    chosen = first_per_key(
        np.frombuffer(keys, dtype=np.int64),
        np.frombuffer(priority, dtype=np.float64),
        np.frombuffer(caps, dtype=np.int64),
    )

    # Pass 2: write the chosen rows
    output = (serving_reference_path if serving else reference_path)(config, name)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    counts: Counter[str] = Counter()
    target_species: set[str] = set()
    named_species: set[str] = set()
    families: set[str] = set()
    with pq.ParquetWriter(partial, REFERENCE_SCHEMA, compression="zstd") as writer:
        rows: list[dict[str, Any]] = []
        for position, row in enumerate(_rows(path, rules, splits)):
            if not chosen[position]:
                continue
            rows.append({column: row[column] for column in REFERENCE_SCHEMA.names})
            if row["role"] == "target":
                counts["target"] += 1
                target_species.add(row["species"] or "")
                if row["species_named"]:
                    named_species.add(row["species"])
                else:
                    counts["placeholder"] += 1
            else:
                counts["background"] += 1
                families.add(row["family"] or "")
                if row["family"] in rules.close_relatives:
                    counts["close"] += 1
            if len(rows) >= BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA))
                rows.clear()
        writer.write_table(pa.Table.from_pylist(rows, schema=REFERENCE_SCHEMA))
    if not counts["target"]:
        partial.unlink()
        msg = f"No records of {rules.target} found in {path} ({sorted(splits)})"
        raise ReferenceStageError(msg)
    partial.replace(output)

    # Leave out references that look mislabelled
    contradicted = []
    if rules.curate:
        contradicted = Specialist.from_parquet(output, rules).contradicted()
        dropped = {c.record_id for c in contradicted}
        if dropped:
            kept = [
                row
                for row in pq.read_table(output).to_pylist()
                if row["record_id"] not in dropped
            ]
            with pq.ParquetWriter(output, REFERENCE_SCHEMA, compression="zstd") as out:
                out.write_table(pa.Table.from_pylist(kept, schema=REFERENCE_SCHEMA))
            counts["target"] -= len(dropped)
        (output.parent / CONTRADICTED_NAME).write_text(
            json.dumps([asdict(c) for c in contradicted], indent=1) + "\n", "utf-8"
        )
        logger.info(
            "Left out %d references that look like another species", len(dropped)
        )

    # The barcode gap of each named target species
    logger.info("Finding the nearest other species for %d species", len(named_species))
    neighbours = Specialist.from_parquet(output, rules).find_neighbours()
    write_neighbours(output.parent / NEIGHBOURS_NAME, neighbours)

    report = ReferenceReport(
        app=config.name,
        specialist=name,
        release=config.data.source.release,
        gene=rules.gene,
        prokopton_ml_version=__version__,
        rules=rules,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        target_records=counts["target"],
        target_species=len(target_species),
        target_named_species=len(named_species),
        target_placeholder_records=counts["placeholder"],
        background_records=counts["background"],
        background_families=len(families),
        close_relative_records=counts["close"],
        duplicates_skipped=duplicates,
        contradicted_skipped=len(contradicted),
        species_with_neighbours=len(neighbours),
        species_within_species_identity=sum(
            n.identity >= rules.species_identity for n in neighbours.values()
        ),
        output=str(output),
        splits=tuple(sorted(splits)),
    )
    report_path = output.parent / REPORT_NAME
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Reference for %s: %d target records (%d named species), %d background "
        "records in %d families, %d repeated sequences skipped -> %s",
        name,
        report.target_records,
        report.target_named_species,
        report.background_records,
        report.background_families,
        report.duplicates_skipped,
        output,
    )
    logger.info(
        "Barcode gaps: %d of %d named species have another named species at "
        "least %.1f%% identical",
        report.species_within_species_identity,
        report.species_with_neighbours,
        rules.species_identity * 100,
    )
    return report
