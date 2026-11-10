"""Command-line interface for ProkoptonML: ``prokopton-ml <command> <config>``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from prokopton.core.config import ConfigError, load_config

from prokopton.ml.config import AppConfig
from prokopton.ml.data.checksums import ChecksumManifestError, ChecksumMismatchError
from prokopton.ml.data.download import DownloadError
from prokopton.ml.data.taxonomy import ReferenceDataError
from prokopton.ml.pipeline.ingest import IngestError, ingest
from prokopton.ml.pipeline.split import SplitStageError, split
from prokopton.ml.pipeline.validate import DataValidationError, validate

__all__ = ["ENV_PREFIX", "main"]

ENV_PREFIX = "PROKOPTON_"
"""Prefix for environment variables that override config values, e.g.
``PROKOPTON_DATA__PATHS__RAW_DIR``."""

logger = logging.getLogger("prokopton.ml")


_COMMANDS = {
    "ingest": "Download, verify and convert an app's source data to Parquet.",
    "validate": "Check data quality, apply rules and write processed data.",
    "split": "Assign records to training, validation and test sets.",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prokopton-ml",
        description="Build, deploy and monitor scientific ML applications.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _COMMANDS.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path, help="Path to the app's TOML file.")
    return parser


def _run(command: str, config: AppConfig) -> str:
    """Run one pipeline stage and return a one-line summary."""
    if command == "ingest":
        ingest_report = ingest(config)
        total = sum(gene.records for gene in ingest_report.genes.values())
        return f"{len(ingest_report.genes)} genes, {total} records"
    if command == "validate":
        validation_report = validate(config)
        kept = sum(g.kept_records for g in validation_report.genes.values())
        seen = sum(g.input_records for g in validation_report.genes.values())
        return f"kept {kept} of {seen} records"
    split_report = split(config)
    totals = {
        name: sum(g.records[name] for g in split_report.genes.values())
        for name in next(iter(split_report.genes.values())).records
    }
    return f"records per split: {totals}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return an exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
        summary = _run(args.command, config)
    except (
        ConfigError,
        IngestError,
        DownloadError,
        ChecksumManifestError,
        ChecksumMismatchError,
        ReferenceDataError,
        DataValidationError,
        SplitStageError,
    ) as error:
        logger.error("%s", error)
        return 1
    logger.info("Done: %s", summary)
    return 0
