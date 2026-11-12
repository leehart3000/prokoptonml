"""Command-line interface for ProkoptonML: ``prokopton-ml <command> <config>``."""

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from prokopton.core.config import ConfigError, load_config

from prokopton.ml.config import AppConfig, UnknownClassifierError
from prokopton.ml.data.checksums import ChecksumManifestError, ChecksumMismatchError
from prokopton.ml.data.download import DownloadError
from prokopton.ml.data.taxonomy import ReferenceDataError
from prokopton.ml.inference.load import LoadError, load_identifier
from prokopton.ml.pipeline.calibrate import CalibrationStageError, calibrate
from prokopton.ml.pipeline.evaluate import EvaluationStageError, evaluate
from prokopton.ml.pipeline.ingest import IngestError, ingest
from prokopton.ml.pipeline.leakage import LeakageStageError, leakage
from prokopton.ml.pipeline.reference import ReferenceStageError, build_reference
from prokopton.ml.pipeline.sample import SampleStageError, sample
from prokopton.ml.pipeline.specialist_evaluate import (
    SpecialistEvaluationError,
    evaluate_specialist,
)
from prokopton.ml.pipeline.split import SplitStageError, split
from prokopton.ml.pipeline.train import TrainStageError, train
from prokopton.ml.pipeline.validate import DataValidationError, validate
from prokopton.ml.reporting.card import CardError, build_card
from prokopton.ml.reporting.specialist_card import (
    SpecialistCardError,
    build_specialist_card,
)

__all__ = ["ENV_PREFIX", "main"]

ENV_PREFIX = "PROKOPTON_"
"""Prefix for environment variables that override config values, e.g.
``PROKOPTON_DATA__PATHS__RAW_DIR``."""

logger = logging.getLogger("prokopton.ml")


_COMMANDS = {
    "ingest": "Download, verify and convert an app's source data to Parquet.",
    "validate": "Check data quality, apply rules and write processed data.",
    "split": "Assign records to training, validation and test sets.",
    "leakage": "List novel-set records with a near-copy in the training set.",
    "sample": "Draw balanced training and validation samples for a classifier.",
    "train": "Train a classifier's candidates, then track and register the best.",
    "calibrate": "Calibrate a registered classifier's confidence by length.",
    "evaluate": (
        "Measure a registered classifier on the test sets, or tune and test "
        "a specialist (with --specialist)."
    ),
    "card": (
        "Write a classifier's model card from the pipeline's reports, or a "
        "specialist's (with --specialist)."
    ),
    "reference": "Build a specialist's reference set from the training split.",
    "identify": (
        "Identify one sequence with the registered models and print the "
        "result, with its decision trace, as JSON."
    ),
}
_CLASSIFIER_COMMANDS = {"sample", "train", "calibrate", "evaluate", "card"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prokopton-ml",
        description="Build, deploy and monitor scientific ML applications.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in _COMMANDS.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config", type=Path, help="Path to the app's TOML file.")
        if name in _CLASSIFIER_COMMANDS:
            command.add_argument(
                "--model",
                choices=("router", "marker"),
                default="router",
                help="Which classifier to work on (default: router).",
            )
        if name == "identify":
            command.add_argument(
                "--input",
                type=Path,
                required=True,
                help="A file with one sequence (plain or FASTA); '-' for stdin.",
            )
        if name in ("reference", "evaluate", "card"):
            command.add_argument(
                "--specialist",
                required=name == "reference",
                default="",
                help="Which specialist (as named in the app's config).",
            )
    return parser


def _identify(config: AppConfig, source: Path) -> str:
    text = sys.stdin.read() if str(source) == "-" else source.read_text("utf-8")
    loaded = load_identifier(config)
    result = loaded.identifier.identify(text)
    output = {"versions": loaded.versions, **result.model_dump(mode="json")}
    print(json.dumps(output, indent=2))
    return f"{result.status}: {result.answer.text if result.answer else result.reason}"


def _run(
    command: str,
    config: AppConfig,
    model: str = "router",
    specialist: str = "",
    source: Path | None = None,
) -> str:
    """Run one pipeline stage and return a one-line summary."""
    if command == "identify" and source is not None:
        return _identify(config, source)
    if command == "reference":
        reference = build_reference(config, specialist)
        return (
            f"{reference.target_records} target records "
            f"({reference.target_named_species} named species), "
            f"{reference.background_records} background records"
        )
    if command == "ingest":
        ingest_report = ingest(config)
        total = sum(gene.records for gene in ingest_report.genes.values())
        return f"{len(ingest_report.genes)} genes, {total} records"
    if command == "validate":
        validation_report = validate(config)
        kept = sum(g.kept_records for g in validation_report.genes.values())
        seen = sum(g.input_records for g in validation_report.genes.values())
        return f"kept {kept} of {seen} records"
    if command == "leakage":
        leakage_report = leakage(config)
        found = {g: r.near_duplicates for g, r in leakage_report.genes.items()}
        return f"near-duplicates: {found}"
    if command == "sample":
        sample_report = sample(config, model)
        sizes = {name: r.sampled_records for name, r in sample_report.splits.items()}
        return f"{len(sample_report.labels)} classes; records per split: {sizes}"
    if command == "train":
        train_report = train(config, model=model)
        return (
            f"best {train_report.best_model} k={train_report.best_k}, "
            f"selection score {train_report.metrics['selection_score']:.4f}; "
            f"{train_report.registered_model} v{train_report.model_version} "
            f"@{train_report.alias}"
        )
    if command == "calibrate":
        calibration = calibrate(config, model=model)
        return (
            f"v{calibration.base_version} calibrated as v{calibration.model_version}; "
            f"NLL {calibration.nll_before:.4f} -> {calibration.nll_after:.4f}"
        )
    if command == "evaluate" and specialist:
        tuned = evaluate_specialist(config, specialist)
        chosen = tuned.chosen
        return (
            f"{specialist} thresholds: species {chosen.species_identity}, "
            f"group margin {chosen.group_margin}, member {chosen.member_identity}"
        )
    if command == "card" and specialist:
        written = build_specialist_card(config, specialist)
        return f"{specialist} model card written to {written.card}"
    if command == "card":
        files = build_card(config, model)
        return f"model card for v{files.model_version} written to {files.card}"
    if command == "evaluate":
        evaluation = evaluate(config, model=model)
        accuracy = {
            name: round(scores.metrics["accuracy"], 4)
            for name, scores in evaluation.splits.items()
        }
        return f"v{evaluation.model_version} accuracy: {accuracy}"
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
        summary = _run(
            args.command,
            config,
            getattr(args, "model", "router"),
            getattr(args, "specialist", ""),
            getattr(args, "input", None),
        )
    except (
        ConfigError,
        IngestError,
        DownloadError,
        ChecksumManifestError,
        ChecksumMismatchError,
        ReferenceDataError,
        DataValidationError,
        SplitStageError,
        LeakageStageError,
        SampleStageError,
        TrainStageError,
        EvaluationStageError,
        CalibrationStageError,
        CardError,
        UnknownClassifierError,
        ReferenceStageError,
        SpecialistEvaluationError,
        SpecialistCardError,
        LoadError,
        OSError,
    ) as error:
        logger.error("%s", error)
        return 1
    logger.info("Done: %s", summary)
    return 0
