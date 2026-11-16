"""Stage kinds for sequence data: ingest, validate, split and leakage of
MIDORI2 records, k-mer classifiers (sample, train, calibrate, evaluate, card)
and specialists (anchors, reference, evaluate, card).

Each wraps the function that does the work, and returns the one-line summary
that ``prokopton-ml run`` prints.
"""

from prokopton.ml.bio.config import specialist_settings
from prokopton.ml.bio.data.taxonomy import ReferenceDataError
from prokopton.ml.bio.pipeline.calibrate import CalibrationStageError, calibrate
from prokopton.ml.bio.pipeline.evaluate import EvaluationStageError, evaluate
from prokopton.ml.bio.pipeline.ingest import IngestError, ingest
from prokopton.ml.bio.pipeline.leakage import LeakageStageError, leakage
from prokopton.ml.bio.pipeline.reference import ReferenceStageError, build_reference
from prokopton.ml.bio.pipeline.sample import SampleStageError, sample
from prokopton.ml.bio.pipeline.specialist_evaluate import (
    SpecialistEvaluationError,
    evaluate_specialist,
)
from prokopton.ml.bio.pipeline.split import SplitStageError, split
from prokopton.ml.bio.pipeline.train import TrainStageError, train
from prokopton.ml.bio.pipeline.validate import DataValidationError, validate
from prokopton.ml.bio.reporting.card import CardError, build_card
from prokopton.ml.bio.reporting.specialist_card import (
    SpecialistCardError,
    build_specialist_card,
)
from prokopton.ml.bio.sources.refseq import AnchorError, fetch_anchors
from prokopton.ml.data.checksums import ChecksumManifestError, ChecksumMismatchError
from prokopton.ml.data.download import DownloadError
from prokopton.ml.stages import StageContext, StageKind

__all__ = ["bio_kinds"]

CLASSIFIER = "bio.kmer-classifier"
SPECIALIST = "bio.specialist"


def _model(context: StageContext) -> str:
    # Registry.check makes sure stages that need a model have one
    assert context.model is not None
    return context.model


def _ingest(context: StageContext) -> str:
    report = ingest(context.config)
    total = sum(gene.records for gene in report.genes.values())
    return f"{len(report.genes)} genes, {total} records"


def _validate(context: StageContext) -> str:
    report = validate(context.config)
    kept = sum(g.kept_records for g in report.genes.values())
    seen = sum(g.input_records for g in report.genes.values())
    return f"kept {kept} of {seen} records"


def _split(context: StageContext) -> str:
    report = split(context.config)
    totals = {
        name: sum(g.records[name] for g in report.genes.values())
        for name in next(iter(report.genes.values())).records
    }
    return f"records per split: {totals}"


def _leakage(context: StageContext) -> str:
    report = leakage(context.config)
    found = {g: r.near_duplicates for g, r in report.genes.items()}
    return f"near-duplicates: {found}"


def _sample(context: StageContext) -> str:
    report = sample(context.config, _model(context))
    sizes = {name: r.sampled_records for name, r in report.splits.items()}
    return f"{len(report.labels)} classes; records per split: {sizes}"


def _train(context: StageContext) -> str:
    report = train(context.config, model=_model(context))
    return (
        f"best {report.best_model} k={report.best_k}, "
        f"selection score {report.metrics['selection_score']:.4f}; "
        f"{report.registered_model} v{report.model_version} @{report.alias}"
    )


def _calibrate(context: StageContext) -> str:
    report = calibrate(context.config, model=_model(context))
    return (
        f"v{report.base_version} calibrated as v{report.model_version}; "
        f"NLL {report.nll_before:.4f} -> {report.nll_after:.4f}"
    )


def _evaluate(context: StageContext) -> str:
    report = evaluate(context.config, model=_model(context))
    accuracy = {
        name: round(scores.metrics["accuracy"], 4)
        for name, scores in report.splits.items()
    }
    return f"v{report.model_version} accuracy: {accuracy}"


def _card(context: StageContext) -> str:
    files = build_card(context.config, _model(context))
    return f"model card for v{files.model_version} written to {files.card}"


def _anchors(context: StageContext) -> str:
    name = _model(context)
    rules = specialist_settings(context.config, name)
    if rules.anchors is None or not rules.anchor_taxon:
        msg = f"Set anchors and anchor_taxon for the {name!r} specialist"
        raise AnchorError(msg)
    anchors = fetch_anchors(rules.anchor_taxon, rules.anchors)
    return f"{len(anchors)} {rules.gene} anchors written to {rules.anchors}"


def _reference(context: StageContext) -> str:
    reference = build_reference(context.config, _model(context))
    return (
        f"{reference.target_records} target records "
        f"({reference.target_named_species} named species), "
        f"{reference.background_records} background records"
    )


def _specialist_evaluate(context: StageContext) -> str:
    name = _model(context)
    chosen = evaluate_specialist(context.config, name).chosen
    return (
        f"{name} thresholds: species {chosen.species_identity}, "
        f"group margin {chosen.group_margin}, member {chosen.member_identity}"
    )


def _specialist_card(context: StageContext) -> str:
    name = _model(context)
    written = build_specialist_card(context.config, name)
    return f"{name} model card written to {written.card}"


def bio_kinds() -> list[StageKind]:
    """The stage kinds for sequence data."""
    return [
        StageKind(
            "bio.ingest",
            "Download, verify and convert MIDORI2 data to Parquet.",
            _ingest,
            errors=(
                IngestError,
                DownloadError,
                ChecksumManifestError,
                ChecksumMismatchError,
            ),
        ),
        StageKind(
            "bio.validate",
            "Check data quality, apply rules and write processed data.",
            _validate,
            uses=("data", "bio.source", "bio.paths", "bio.validation"),
            errors=(DataValidationError, ReferenceDataError),
        ),
        StageKind(
            "bio.split",
            "Assign records to training, validation and test sets.",
            _split,
            uses=("data", "bio.source", "bio.paths", "bio.split"),
            errors=(SplitStageError,),
        ),
        StageKind(
            "bio.leakage",
            "List novel-set records with a near-copy in the training set.",
            _leakage,
            uses=("data", "bio.source", "bio.paths", "bio.leakage"),
            errors=(LeakageStageError,),
        ),
        StageKind(
            "bio.classifier-sample",
            "Draw balanced training and validation samples for a classifier.",
            _sample,
            model_kind=CLASSIFIER,
            errors=(SampleStageError,),
        ),
        StageKind(
            "bio.classifier-train",
            "Train a classifier's candidates, then track and register the best.",
            _train,
            model_kind=CLASSIFIER,
            uses=("data", "paths", "bio.source", "bio.paths", "mlflow"),
            errors=(TrainStageError,),
        ),
        StageKind(
            "bio.classifier-calibrate",
            "Calibrate a registered classifier's confidence by length.",
            _calibrate,
            model_kind=CLASSIFIER,
            uses=("data", "paths", "bio.source", "bio.paths", "mlflow"),
            errors=(CalibrationStageError,),
        ),
        StageKind(
            "bio.classifier-evaluate",
            "Measure a registered classifier on the test sets.",
            _evaluate,
            model_kind=CLASSIFIER,
            uses=("data", "paths", "bio.source", "bio.paths", "mlflow"),
            errors=(EvaluationStageError,),
        ),
        StageKind(
            "bio.classifier-card",
            "Write a classifier's model card from the pipeline's reports.",
            _card,
            model_kind=CLASSIFIER,
            uses=("data", "paths", "bio.source", "bio.paths", "mlflow"),
            errors=(CardError,),
        ),
        StageKind(
            "bio.specialist-anchors",
            "Fetch the barcode genes of a specialist's RefSeq mitogenomes from "
            "NCBI, to check reference labels against.",
            _anchors,
            model_kind=SPECIALIST,
            uses=(),
            errors=(AnchorError,),
        ),
        StageKind(
            "bio.specialist-reference",
            "Build a specialist's reference set from the training split.",
            _reference,
            model_kind=SPECIALIST,
            errors=(ReferenceStageError,),
        ),
        StageKind(
            "bio.specialist-evaluate",
            "Tune a specialist's thresholds and test them.",
            _specialist_evaluate,
            model_kind=SPECIALIST,
            errors=(SpecialistEvaluationError,),
        ),
        StageKind(
            "bio.specialist-card",
            "Write a specialist's model card.",
            _specialist_card,
            model_kind=SPECIALIST,
            errors=(SpecialistCardError,),
        ),
    ]
