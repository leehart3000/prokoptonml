"""Leakage stage: list novel-set records that have a near-copy in training.

Holding out whole species does not stop a held-out species from appearing in
training under another name, such as a placeholder ("Aedes sp. BIOUG...") that
differs by a base or two. The leakage audit (``analysis/03_leakage_audit.py``)
found this for about a third of novel-set records. This stage searches every
record of the novel splits against the training split with VSEARCH, and lists
those with a match of at least ``min_identity`` (by default 99%).

The split data is not changed. Evaluation reads the list to report
novel-species scores both with and without these records.

Memory stays small: identical query sequences are searched once, and the
training split is streamed to temporary FASTA files in chunks, each deleted
after it has been searched.

Outputs ``<leakage_dir>/midori2_<release>/gene=<gene>/near_duplicates.parquet``
and ``_leakage_report.json``.
"""

import logging
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, LeakageConfig

__all__ = [
    "NEAR_DUPLICATES_SCHEMA",
    "REPORT_NAME",
    "GeneLeakageReport",
    "LeakageReport",
    "LeakageStageError",
    "find_near_duplicates",
    "leakage",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_leakage_report.json"
BATCH_SIZE = 100_000

NEAR_DUPLICATES_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("split", pa.string()),
        ("best_identity", pa.float64()),
        ("train_record_id", pa.string()),
    ]
)


class LeakageStageError(RuntimeError):
    """Raised when the leakage stage cannot run."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeneLeakageReport(_Report):
    """Near-duplicate counts for one gene."""

    query_records: dict[str, int]
    unique_query_sequences: int
    near_duplicates: dict[str, int]
    near_duplicate_share: dict[str, float]
    reference_records: int
    chunks: int
    seconds: int
    output: str


class LeakageReport(_Report):
    """Provenance and results of a leakage run."""

    app: str
    release: str
    prokopton_ml_version: str
    rules: LeakageConfig
    started_at: datetime
    finished_at: datetime
    genes: dict[str, GeneLeakageReport]


def _run_vsearch(
    queries: Path, database: Path, hits: Path, rules: LeakageConfig
) -> None:
    command = [
        *rules.vsearch_command,
        "--usearch_global",
        str(queries),
        "--db",
        str(database),
        "--id",
        str(rules.min_identity),
        "--maxaccepts",
        str(rules.maxaccepts),
        "--maxrejects",
        str(rules.maxrejects),
        "--userout",
        str(hits),
        "--userfields",
        "query+target+id",
        "--threads",
        str(rules.threads),
        "--quiet",
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        msg = f"VSEARCH failed: {error}"
        raise LeakageStageError(msg) from error


def find_near_duplicates(
    path: Path, work_dir: Path, rules: LeakageConfig
) -> tuple[list[dict[str, object]], GeneLeakageReport]:
    """Search one gene's novel-split records against its training records.

    Returns:
        One row per near-duplicate query record, and a report (whose
        ``output`` is left empty for the caller to fill in).
    """
    started = time.monotonic()
    source = pq.ParquetFile(path)
    query_splits = set(rules.query_splits)

    # Pass 1: unique query sequences, each with the records that share it
    labels: dict[str, str] = {}
    records_by_label: dict[str, list[tuple[str, str]]] = {}
    query_records: Counter[str] = Counter()
    for batch in source.iter_batches(
        batch_size=BATCH_SIZE, columns=["record_id", "split", "sequence"]
    ):
        for record_id, split, sequence in zip(
            batch.column("record_id").to_pylist(),
            batch.column("split").to_pylist(),
            batch.column("sequence").to_pylist(),
            strict=True,
        ):
            if split not in query_splits:
                continue
            query_records[split] += 1
            label = labels.setdefault(sequence, f"q{len(labels)}")
            records_by_label.setdefault(label, []).append((record_id, split))

    work_dir.mkdir(parents=True, exist_ok=True)
    query_fasta = work_dir / "queries.fasta"
    with query_fasta.open("w", encoding="utf-8") as file:
        file.writelines(f">{label}\n{sequence}\n" for sequence, label in labels.items())
    labels.clear()

    # Pass 2: stream the reference split in chunks and search each one
    best: dict[str, tuple[float, str]] = {}
    chunk: list[tuple[str, str]] = []
    reference_records = 0
    chunks = 0

    def search_chunk() -> None:
        nonlocal chunks
        if not chunk or not records_by_label:
            chunk.clear()
            return
        chunks += 1
        database = work_dir / f"reference_{chunks}.fasta"
        hits = work_dir / f"hits_{chunks}.tsv"
        with database.open("w", encoding="utf-8") as file:
            file.writelines(f">r{index}\n{sequence}\n" for index, sequence in chunk)
        chunk_started = time.monotonic()
        _run_vsearch(query_fasta, database, hits, rules)
        with hits.open(encoding="utf-8") as file:
            for line in file:
                query, target, identity = line.rstrip("\n").split("\t")
                value = float(identity)
                if query not in best or value > best[query][0]:
                    best[query] = (value, target)
        database.unlink()
        hits.unlink()
        logger.info(
            "Searched reference chunk %d (%d sequences) in %.0f s",
            chunks,
            len(chunk),
            time.monotonic() - chunk_started,
        )
        chunk.clear()

    reference_ids: list[str] = []
    for batch in source.iter_batches(
        batch_size=BATCH_SIZE, columns=["record_id", "split", "sequence"]
    ):
        for record_id, split, sequence in zip(
            batch.column("record_id").to_pylist(),
            batch.column("split").to_pylist(),
            batch.column("sequence").to_pylist(),
            strict=True,
        ):
            if split != rules.reference_split:
                continue
            chunk.append((str(reference_records), sequence))
            reference_ids.append(record_id)
            reference_records += 1
            if len(chunk) >= rules.chunk_size:
                search_chunk()
    search_chunk()
    query_fasta.unlink()

    rows: list[dict[str, object]] = []
    for label, (identity, target) in best.items():
        for record_id, split in records_by_label[label]:
            rows.append(
                {
                    "record_id": record_id,
                    "split": split,
                    "best_identity": identity,
                    "train_record_id": reference_ids[int(target.removeprefix("r"))],
                }
            )
    found = Counter(str(row["split"]) for row in rows)
    report = GeneLeakageReport(
        query_records={split: query_records[split] for split in rules.query_splits},
        unique_query_sequences=len(records_by_label),
        near_duplicates={split: found[split] for split in rules.query_splits},
        near_duplicate_share={
            split: round(found[split] / query_records[split], 4)
            if query_records[split]
            else 0.0
            for split in rules.query_splits
        },
        reference_records=reference_records,
        chunks=chunks,
        seconds=round(time.monotonic() - started),
        output="",
    )
    return rows, report


def leakage(config: AppConfig) -> LeakageReport:
    """Find near-duplicates for every configured gene and write the results.

    Raises:
        LeakageStageError: If a gene has not been split yet, or VSEARCH fails.
    """
    started_at = datetime.now(UTC)
    rules = config.leakage
    folder = f"midori2_{config.data.source.release.lower()}"
    input_root = config.data.paths.splits_dir / folder
    output_root = config.data.paths.leakage_dir / folder

    genes: dict[str, GeneLeakageReport] = {}
    for gene in rules.genes:
        input_path = input_root / f"gene={gene}" / "records.parquet"
        if not input_path.exists():
            msg = f"{input_path} not found; run the split stage first"
            raise LeakageStageError(msg)
        gene_dir = output_root / f"gene={gene}"
        rows, report = find_near_duplicates(input_path, gene_dir / "_work", rules)
        output = gene_dir / "near_duplicates.parquet"
        pq.write_table(
            pa.Table.from_pylist(rows, schema=NEAR_DUPLICATES_SCHEMA), output
        )
        (gene_dir / "_work").rmdir()
        genes[gene] = report.model_copy(update={"output": str(output)})
        logger.info("Leakage %s: %s -> %s", gene, report.near_duplicates, output)

    result = LeakageReport(
        app=config.name,
        release=config.data.source.release,
        prokopton_ml_version=__version__,
        rules=rules,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        genes=genes,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return result
