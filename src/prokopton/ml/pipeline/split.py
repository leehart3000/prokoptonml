"""Split stage: assign every record to a training, validation or test set.

There are five splits:

- ``val_novel`` and ``test_novel``: every record of a held-out *named* species,
  to measure performance on species that are not in the training data;
- ``train``, ``val_known`` and ``test_known``: the remaining records, to measure
  performance on new specimens of species that are in the training data.

Assignment is by a salted hash, so it is reproducible and stable when new data
is added. Novel holdouts are chosen by species name. Any record whose exact
sequence also occurs in a novel split joins that split, so identical sequences
under another name (e.g. "Aedes sp.") cannot leak into training. The remaining
records are split by a hash of the sequence, so identical sequences always
stay together. Near-identical sequences are not grouped; the leakage audit
measures how much that matters.

Data is streamed in batches, so memory use stays small whatever the gene size:
pass 1 finds the held-out sequences (kept as 64-bit fingerprints), and pass 2
assigns every record and writes it out with a ``split`` column.

Outputs ``<splits_dir>/midori2_<release>/gene=<gene>/records.parquet`` and
``_split_report.json``.
"""

import hashlib
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, SplitConfig

__all__ = [
    "REPORT_NAME",
    "SPLITS",
    "GeneSplitReport",
    "SplitReport",
    "SplitStageError",
    "SplitSummary",
    "assign_splits",
    "find_novel_sequences",
    "hash_fraction",
    "hash_key",
    "split",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_split_report.json"
SPLITS = ("train", "val_known", "test_known", "val_novel", "test_novel")
BATCH_SIZE = 100_000
_NOVEL_PRIORITY = {"val_novel": 1, "test_novel": 2}
_SCALE = 2**64


class SplitStageError(RuntimeError):
    """Raised when the split stage cannot run."""


def hash_key(key: str, salt: str) -> int:
    """Return a stable, salted 64-bit fingerprint of ``key``."""
    digest = hashlib.blake2b(f"{salt}\x00{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def hash_fraction(key: str, salt: str) -> float:
    """Map ``key`` to a stable, uniformly spread number in ``[0, 1)``."""
    return hash_key(key, salt) / _SCALE


def _novel_split(
    species: str, rules: SplitConfig, cache: dict[str, str | None]
) -> str | None:
    """Return the novel split for a named species, or ``None`` if it is not held out."""
    if species not in cache:
        fraction = hash_fraction(species, f"{rules.salt}:species")
        if fraction < rules.novel_val_fraction:
            cache[species] = "val_novel"
        elif fraction < rules.novel_val_fraction + rules.novel_test_fraction:
            cache[species] = "test_novel"
        else:
            cache[species] = None
    return cache[species]


def _known_split(key: int, rules: SplitConfig) -> str:
    fraction = key / _SCALE
    train_cut = 1.0 - rules.val_fraction - rules.test_fraction
    if fraction < train_cut:
        return "train"
    if fraction < train_cut + rules.val_fraction:
        return "val_known"
    return "test_known"


def find_novel_sequences(
    batches: Iterable[
        tuple[Sequence[str | None], Sequence[bool | None], Sequence[str]]
    ],
    rules: SplitConfig,
) -> dict[int, str]:
    """Pass 1: return ``{sequence fingerprint: novel split}`` for held-out sequences.

    Each batch is ``(species, species_named, sequence)`` columns. When a
    sequence belongs to both novel splits, ``test_novel`` wins.
    """
    salt = f"{rules.salt}:sequence"
    cache: dict[str, str | None] = {}
    novel: dict[int, str] = {}
    for species_column, named_column, sequence_column in batches:
        for species, named, sequence in zip(
            species_column, named_column, sequence_column, strict=True
        ):
            if not named or species is None:
                continue
            split_name = _novel_split(species, rules, cache)
            if split_name is None:
                continue
            key = hash_key(sequence, salt)
            current = novel.get(key)
            if (
                current is None
                or _NOVEL_PRIORITY[split_name] > _NOVEL_PRIORITY[current]
            ):
                novel[key] = split_name
    return novel


def assign_splits(
    sequences: Sequence[str], novel: dict[int, str], rules: SplitConfig
) -> list[str]:
    """Pass 2: return the split for each sequence in a batch."""
    salt = f"{rules.salt}:sequence"
    result = []
    for sequence in sequences:
        key = hash_key(sequence, salt)
        result.append(novel.get(key) or _known_split(key, rules))
    return result


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeneSplitReport(_Report):
    """Split sizes for one gene."""

    records: dict[str, int]
    named_species: dict[str, int]
    groups: dict[str, dict[str, int]]
    moved_by_identical_sequence: int
    families: dict[str, dict[str, dict[str, int]]]
    output: str | None = None


class SplitReport(_Report):
    """Provenance and sizes of a split run."""

    app: str
    release: str
    prokopton_ml_version: str
    rules: SplitConfig
    started_at: datetime
    finished_at: datetime
    genes: dict[str, GeneSplitReport]


def _ordered(counts: Counter[str]) -> dict[str, int]:
    return {name: counts[name] for name in SPLITS}


class SplitSummary:
    """Accumulates split counts batch by batch."""

    def __init__(self, rules: SplitConfig) -> None:
        self._rules = rules
        self._cache: dict[str, str | None] = {}
        self.records: Counter[str] = Counter()
        self.named: dict[str, set[str]] = defaultdict(set)
        self.groups: dict[str, Counter[str]] = defaultdict(Counter)
        self.family_records: dict[str, Counter[str]] = defaultdict(Counter)
        self.family_named: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self.moved = 0

    def update(
        self,
        splits: Sequence[str],
        species: Sequence[str | None],
        named: Sequence[bool | None],
        groups: Sequence[str | None],
        families: Sequence[str | None],
    ) -> None:
        """Add one batch's assignments to the totals."""
        reported = set(self._rules.report_families)
        for split_name, name, is_named, group, family in zip(
            splits, species, named, groups, families, strict=True
        ):
            self.records[split_name] += 1
            self.groups[group or "Unknown"][split_name] += 1
            named_species = name if is_named and name is not None else None
            if named_species is not None:
                self.named[split_name].add(named_species)
            if split_name in _NOVEL_PRIORITY and (
                named_species is None
                or _novel_split(named_species, self._rules, self._cache) is None
            ):
                self.moved += 1
            if family in reported:
                self.family_records[family][split_name] += 1
                if named_species is not None:
                    self.family_named[family][split_name].add(named_species)

    def report(self) -> GeneSplitReport:
        """Return the totals as a report."""
        return GeneSplitReport(
            records=_ordered(self.records),
            named_species={name: len(self.named[name]) for name in SPLITS},
            groups={
                group: _ordered(counts) for group, counts in sorted(self.groups.items())
            },
            moved_by_identical_sequence=self.moved,
            families={
                family: {
                    "records": _ordered(self.family_records[family]),
                    "named_species": {
                        name: len(self.family_named[family][name]) for name in SPLITS
                    },
                }
                for family in self._rules.report_families
            },
        )


def _split_gene(input_path: Path, output: Path, rules: SplitConfig) -> GeneSplitReport:
    """Stream one gene's records through both passes and write the output."""
    source = pq.ParquetFile(input_path)
    pass_one = (
        (
            batch.column("species").to_pylist(),
            batch.column("species_named").to_pylist(),
            batch.column("sequence").to_pylist(),
        )
        for batch in source.iter_batches(
            batch_size=BATCH_SIZE, columns=["species", "species_named", "sequence"]
        )
    )
    novel = find_novel_sequences(pass_one, rules)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    schema = source.schema_arrow.append(pa.field("split", pa.string()))
    summary = SplitSummary(rules)
    with pq.ParquetWriter(partial, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=BATCH_SIZE):
            splits = assign_splits(batch.column("sequence").to_pylist(), novel, rules)
            writer.write_batch(
                pa.RecordBatch.from_arrays(
                    [*batch.columns, pa.array(splits, type=pa.string())], schema=schema
                )
            )
            summary.update(
                splits,
                batch.column("species").to_pylist(),
                batch.column("species_named").to_pylist(),
                batch.column("group").to_pylist(),
                batch.column("family").to_pylist(),
            )
    partial.replace(output)
    return summary.report()


def split(config: AppConfig) -> SplitReport:
    """Assign splits for every gene of an application and write the results.

    Raises:
        SplitStageError: If a gene has not been validated yet.
    """
    started_at = datetime.now(UTC)
    source = config.data.source
    folder = f"midori2_{source.release.lower()}"
    input_root = config.data.paths.processed_dir / folder
    output_root = config.data.paths.splits_dir / folder

    genes: dict[str, GeneSplitReport] = {}
    for gene in source.genes:
        input_path = input_root / f"gene={gene}" / "records.parquet"
        if not input_path.exists():
            msg = f"{input_path} not found; run the validate stage first"
            raise SplitStageError(msg)
        rules = config.split
        if gene not in rules.novel_species_genes:
            rules = rules.model_copy(
                update={"novel_val_fraction": 0.0, "novel_test_fraction": 0.0}
            )
        output = output_root / f"gene={gene}" / "records.parquet"
        report = _split_gene(input_path, output, rules).model_copy(
            update={"output": str(output)}
        )
        genes[gene] = report
        logger.info("Split %s: %s -> %s", gene, report.records, output)

    result = SplitReport(
        app=config.name,
        release=source.release,
        prokopton_ml_version=__version__,
        rules=config.split,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        genes=genes,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return result
