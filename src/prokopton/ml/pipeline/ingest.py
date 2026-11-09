"""Ingestion stage: download, verify and convert source data to Parquet.

For each gene, the stage checks the file's SHA-256 against the release's
published checksums, then streams its records into a Parquet file at
``<interim_dir>/midori2_<release>/gene=<gene>/records.parquet``. A JSON report,
``_ingest_report.json``, records where the data came from, so every later stage
can be traced back to an exact, verified input (data lineage). Its name starts
with ``_`` so that dataset readers such as PyArrow and DuckDB skip it.
"""

import logging
from datetime import UTC, datetime
from itertools import batched
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig
from prokopton.ml.data.checksums import (
    parse_sha256_manifest,
    read_gzip_lines_from_zip,
    verify_zip_member,
)
from prokopton.ml.data.download import download_file
from prokopton.ml.data.fasta import parse_fasta
from prokopton.ml.sources.midori2 import RANKS, parse_midori2_records, record_to_row

__all__ = [
    "MIDORI2_SCHEMA",
    "REPORT_NAME",
    "GeneIngestReport",
    "IngestError",
    "IngestReport",
    "ingest",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_ingest_report.json"
CHECKSUMS_NAME = "release_sha256.txt"

MIDORI2_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("accession", pa.string()),
        ("start", pa.string()),
        ("end", pa.string()),
        ("gene", pa.string()),
        ("length", pa.int32()),
        ("sequence", pa.string()),
        *[
            field
            for rank in RANKS
            for field in ((rank, pa.string()), (f"{rank}_taxid", pa.int64()))
        ],
    ]
)
"""Column names and types of the Parquet files written by this stage."""


class IngestError(RuntimeError):
    """Raised when source data cannot be ingested."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class GeneIngestReport(_Report):
    """What was ingested for one gene."""

    member: str
    sha256: str
    records: int
    output: str


class IngestReport(_Report):
    """Provenance of an ingestion run."""

    app: str
    source: str
    release: str
    doi: str
    licence: str
    prokopton_ml_version: str
    started_at: datetime
    finished_at: datetime
    genes: dict[str, GeneIngestReport]


def _write_gene(
    archive: Path, member: str, gene: str, output: Path, batch_size: int
) -> int:
    """Stream one gene's records into a Parquet file and return the record count."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    lines = read_gzip_lines_from_zip(archive, member)
    rows = (record_to_row(r, gene) for r in parse_midori2_records(parse_fasta(lines)))
    count = 0
    with pq.ParquetWriter(partial, MIDORI2_SCHEMA, compression="zstd") as writer:
        for batch in batched(rows, batch_size, strict=False):
            table = pa.Table.from_pylist(list(batch), schema=MIDORI2_SCHEMA)
            writer.write_table(table)
            count += len(batch)
    partial.replace(output)
    return count


def ingest(config: AppConfig, *, batch_size: int = 100_000) -> IngestReport:
    """Download (if needed), verify and convert an application's source data.

    Raises:
        IngestError: If a gene's file is not listed in the published checksums.
        ChecksumMismatchError: If a file does not match its published checksum.
        DownloadError: If a required file cannot be downloaded.
    """
    started_at = datetime.now(UTC)
    source = config.data.source
    paths = config.data.paths

    archive = paths.raw_dir / source.archive_name
    checksums = paths.raw_dir / CHECKSUMS_NAME
    for url, path in ((source.archive_url, archive), (source.checksums_url, checksums)):
        if download_file(url, path):
            logger.info("Downloaded %s", path)
        else:
            logger.info("Using existing %s", path)

    manifest = parse_sha256_manifest(checksums.read_text(encoding="utf-8").splitlines())
    output_root = paths.interim_dir / f"midori2_{source.release.lower()}"

    genes: dict[str, GeneIngestReport] = {}
    for gene in source.genes:
        member = source.member_template.format(gene=gene)
        expected = manifest.get(member)
        if expected is None:
            msg = f"No published checksum for {member} in {checksums}"
            raise IngestError(msg)
        verify_zip_member(archive, member, expected)
        output = output_root / f"gene={gene}" / "records.parquet"
        records = _write_gene(archive, member, gene, output, batch_size)
        logger.info("Ingested %s: %d records -> %s", gene, records, output)
        genes[gene] = GeneIngestReport(
            member=member, sha256=expected, records=records, output=str(output)
        )

    report = IngestReport(
        app=config.name,
        source=source.type,
        release=source.release,
        doi=source.doi,
        licence=source.licence,
        prokopton_ml_version=__version__,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        genes=genes,
    )
    report_path = output_root / REPORT_NAME
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return report
