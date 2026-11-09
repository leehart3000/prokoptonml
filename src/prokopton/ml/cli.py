"""Command-line interface for ProkoptonML: ``prokopton-ml <command> <config>``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from prokopton.core.config import ConfigError, load_config

from prokopton.ml.config import AppConfig
from prokopton.ml.data.checksums import ChecksumManifestError, ChecksumMismatchError
from prokopton.ml.data.download import DownloadError
from prokopton.ml.pipeline.ingest import IngestError, ingest

__all__ = ["ENV_PREFIX", "main"]

ENV_PREFIX = "PROKOPTON_"
"""Prefix for environment variables that override config values, e.g.
``PROKOPTON_DATA__PATHS__RAW_DIR``."""

logger = logging.getLogger("prokopton.ml")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prokopton-ml",
        description="Build, deploy and monitor scientific ML applications.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    ingest_parser = commands.add_parser(
        "ingest", help="Download, verify and convert an app's source data to Parquet."
    )
    ingest_parser.add_argument("config", type=Path, help="Path to the app's TOML file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface and return an exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    try:
        config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
        report = ingest(config)
    except (
        ConfigError,
        IngestError,
        DownloadError,
        ChecksumManifestError,
        ChecksumMismatchError,
    ) as error:
        logger.error("%s", error)
        return 1
    total = sum(gene.records for gene in report.genes.values())
    logger.info("Done: %d genes, %d records", len(report.genes), total)
    return 0
