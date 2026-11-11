"""Build the router's model card from the pipeline's reports.

A model card explains what a model is for, how it was made, how well it
works and where it does not. This one is generated (never edited by hand)
from the sample, train, calibration and evaluation reports, so it always
matches the registered model. Hand-written notes can be included from a
Markdown file. Two SVG charts are written beside it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline import calibrate, evaluate, sample, train
from prokopton.ml.pipeline.calibrate import CalibrationReport
from prokopton.ml.pipeline.evaluate import EvaluationReport, FragmentScore
from prokopton.ml.pipeline.sample import SampleReport
from prokopton.ml.pipeline.train import TrainReport
from prokopton.ml.reporting.charts import Series, line_chart, reliability_chart

__all__ = ["CARD_NAME", "CardError", "CardFiles", "build_router_card"]

logger = logging.getLogger(__name__)

CARD_NAME = "router.md"
FRAGMENTS_CHART = "router-fragments.svg"
RELIABILITY_CHART = "router-reliability.svg"
_HARDEST = 10
_MIN_SUPPORT = 20

_SAMPLE_NAMES = {
    "train": "Training",
    "val_known": "Validation, known species",
    "val_novel": "Validation, novel species",
    "val_known_fragments": "Validation, known species, fragments",
    "val_novel_fragments": "Validation, novel species, fragments",
}

_SET_NAMES = {
    "test_known": "Known species",
    "test_novel": "Novel species (all)",
    "test_novel_clean": "Novel species (clean)",
    "test_novel_leaky": "Novel species (leaky)",
}


class CardError(RuntimeError):
    """Raised when the reports are missing or describe different models."""


@dataclass(frozen=True, slots=True)
class CardFiles:
    """The files written, and the model version the card describes."""

    card: Path
    fragments_chart: Path
    reliability_chart: Path
    model_version: str


@dataclass(frozen=True, slots=True)
class _Reports:
    sample: SampleReport
    train: TrainReport
    calibration: CalibrationReport | None
    evaluation: EvaluationReport


def _count(number: int, noun: str) -> str:
    return f"{number:,} {noun}{'' if number == 1 else 's'}"


def _pct(value: float, places: int = 1) -> str:
    return f"{value * 100:.{places}f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _read(config: AppConfig) -> _Reports:
    folder = f"midori2_{config.data.source.release.lower()}"
    router_dir = config.data.paths.models_dir / "router"
    paths = {
        "sample": config.data.paths.samples_dir
        / folder
        / "router"
        / sample.REPORT_NAME,
        "train": router_dir / train.REPORT_NAME,
        "evaluation": router_dir / evaluate.REPORT_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"{', '.join(missing)} not found; run the pipeline first"
        raise CardError(msg)
    calibration_path = router_dir / calibrate.REPORT_NAME
    reports = _Reports(
        sample=SampleReport.model_validate_json(paths["sample"].read_text("utf-8")),
        train=TrainReport.model_validate_json(paths["train"].read_text("utf-8")),
        calibration=(
            CalibrationReport.model_validate_json(calibration_path.read_text("utf-8"))
            if calibration_path.exists()
            else None
        ),
        evaluation=EvaluationReport.model_validate_json(
            paths["evaluation"].read_text("utf-8")
        ),
    )
    expected = reports.train.model_version
    if reports.calibration is not None:
        if reports.calibration.base_version != expected:
            msg = (
                f"Calibration is of v{reports.calibration.base_version}, but training "
                f"made v{expected}; run the calibrate stage again"
            )
            raise CardError(msg)
        expected = reports.calibration.model_version
    if reports.evaluation.model_version != expected:
        msg = (
            f"Evaluation is of v{reports.evaluation.model_version}, but the latest "
            f"model is v{expected}; run the evaluate stage again"
        )
        raise CardError(msg)
    return reports


def _fragment_points(points: list[FragmentScore]) -> list[FragmentScore]:
    """Fragments by length, with whole sequences last."""
    return sorted(points, key=lambda p: (p.length is None, p.length or 0))


def _fragments_chart(reports: _Reports, min_length: int) -> str:
    fragments = reports.evaluation.fragments
    names = [n for n in ("test_known", "test_novel_clean") if n in fragments]
    points = _fragment_points(fragments[names[0]])
    # Evenly spaced, so the short lengths (where accuracy changes most) are
    # readable
    x_values = [float(index) for index in range(len(points))]
    x_labels = [str(p.length) if p.length else "Whole" for p in points]
    lengths = [p.length for p in points]
    marker = float(lengths.index(min_length)) if min_length in lengths else None
    series = [
        Series(
            _SET_NAMES[name],
            [p.metrics["accuracy"] for p in _fragment_points(fragments[name])],
        )
        for name in names
    ]
    return line_chart(
        "How short can you go? Accuracy by sequence length",
        x_values,
        x_labels,
        series,
        x_label="Sequence length (bases)",
        y_label="Accuracy",
        marker_x=marker,
        marker_label=f"Minimum ({min_length})",
    )


def _reliability_chart(reports: _Reports) -> str:
    bins = reports.evaluation.splits["test_novel_clean"].calibration
    return reliability_chart(
        "Reliability: novel species (clean), whole sequences",
        [b.mean_confidence for b in bins],
        [b.accuracy for b in bins],
        [b.records for b in bins],
    )


def _training_data(config: AppConfig) -> str:
    source = config.data.source
    link = f"[doi:{source.doi}](https://doi.org/{source.doi})"
    return f"MIDORI2 {source.release} ({config.router.gene}), {link}, {source.licence}"


def _model_section(config: AppConfig, reports: _Reports) -> str:
    train_report = reports.train
    evaluation = reports.evaluation
    kind = {"logistic": "Logistic regression", "naive_bayes": "Naive Bayes"}
    model = kind.get(train_report.best_model, train_report.best_model)
    calibration = reports.calibration
    rows = [
        ["Registered model", f"`{evaluation.registered_model}`"],
        ["Version", f"{evaluation.model_version} (alias `@{evaluation.alias}`)"],
        ["Task", "Predict the group or insect order of a COI barcode sequence"],
        ["Input", "A DNA sequence (A, C, G, T; ambiguous bases allowed)"],
        ["Output", f"One of {len(reports.sample.labels)} labels, with a confidence"],
        [
            "Model",
            f"{model} on {train_report.best_k}-mer frequencies"
            + (
                ", rescaled to unit length"
                if train_report.best_model == "logistic"
                else ""
            )
            + f", alpha {train_report.best_alpha:g}",
        ],
        [
            "Calibration",
            (
                f"Length-aware temperature, from v{calibration.base_version}"
                if calibration
                else "None"
            ),
        ],
        [
            "Training data",
            _training_data(config),
        ],
        ["Built with", f"prokopton-ml {evaluation.prokopton_ml_version}"],
    ]
    return "## Model\n\n" + _table(["", ""], rows)


def _use_section(config: AppConfig) -> str:
    serving = config.router.serving
    return f"""## Intended use

The router is the first step of identification: it
decides which broad group (or, for insects, which order) a COI barcode most
likely belongs to, so a specialist model can identify it further. It is
meant for COI sequences of at least {serving.min_length} bases; shorter
sequences are declined, and sequences under {serving.short_length} bases are
shown with a note that accuracy is lower.

It is not meant for identifying species on its own, for other genes or
markers, for mixed samples, or for decisions with legal, medical or
regulatory consequences without expert review."""


def _data_section(reports: _Reports) -> str:
    sample_report = reports.sample
    rows = [
        [
            _SAMPLE_NAMES.get(name, name),
            f"{split.sampled_records:,}",
            f"{split.classes:,}",
            f"{split.mean_species_per_class:.0f}",
        ]
        for name, split in sample_report.splits.items()
    ]
    rules = sample_report.rules
    return (
        "## Data\n\n"
        "Records were split by species: *known* species appear in training, "
        "while *novel* species are held out entirely, so their scores show how "
        "the model handles species it has never seen. Novel-species records "
        "with a near-copy (99% identity or more) in training were found with "
        "VSEARCH; *clean* records have none, and are the honest measure.\n\n"
        f"Training samples were capped at {rules.train_cap:,} records per class "
        f"(validation: {rules.eval_cap:,}), keeping as many species as possible. "
        f"Classes with under {rules.min_class_records} records were left out "
        f"({len(sample_report.dropped_classes):,} of them), and "
        f"{sample_report.excluded_near_duplicates:,} validation records with a "
        "near-copy in training were removed.\n\n"
        + _table(["Sample", "Records", "Classes", "Species per class"], rows)
    )


def _training_section(reports: _Reports) -> str:
    train_report = reports.train
    metrics = train_report.metrics
    rules = train_report.rules
    sets = [
        ("Known species", "val_known"),
        ("Novel species", "val_novel"),
        ("Known species, fragments", "val_known_fragments"),
        ("Novel species, fragments", "val_novel_fragments"),
    ]
    rows = [
        [label, _pct(metrics[f"{key}_accuracy"]), f"{metrics[f'{key}_macro_f1']:.3f}"]
        for label, key in sets
        if f"{key}_accuracy" in metrics
    ]
    fragments = (
        "Each training record also gave "
        f"{_count(rules.fragments_per_record, 'random fragment')} of "
        f"{rules.fragment_min_length}-{rules.fragment_max_length} bases, so the "
        "model learns short inputs too. "
        if rules.fragments_per_record
        else ""
    )
    return (
        "## Training\n\n"
        f"{_count(train_report.candidates_tried, 'candidate model')} were "
        "compared on the "
        "validation samples (never the test sets), and the best was chosen by "
        "the mean macro-F1 across them. "
        f"{fragments}Every candidate is recorded in MLflow (run "
        f"`{train_report.parent_run_id}`).\n\n"
        + _table(["Validation set", "Accuracy", "Macro-F1"], rows)
    )


def _calibration_section(reports: _Reports) -> str:
    calibration = reports.calibration
    if calibration is None:
        return "## Calibration\n\nThis model's confidence is not calibrated."
    rows = [
        [
            _SAMPLE_NAMES.get(name, name),
            _pct(after["accuracy"]),
            f"{_pct(calibration.before[name]['mean_confidence'])} → "
            f"{_pct(after['mean_confidence'])}",
            f"{calibration.before[name]['ece']:.3f} → {after['ece']:.3f}",
        ]
        for name, after in calibration.after.items()
    ]
    return (
        "## Calibration\n\n"
        "Confidence was corrected with a temperature that depends on length, "
        f"T = exp({calibration.a:.4f} + {calibration.b:.4f} * ln(length)), "
        "fitted on the validation samples (whole sequences and new random "
        "fragments). It changes only confidence, never the prediction. "
        "*ECE* (expected calibration error) is the average gap between "
        "confidence and accuracy; 0 is perfect.\n\n"
        + _table(["Validation set", "Accuracy", "Mean confidence", "ECE"], rows)
    )


def _results_section(reports: _Reports) -> str:
    evaluation = reports.evaluation
    rows = [
        [
            _SET_NAMES.get(name, name),
            f"{scores.records:,}",
            _pct(scores.metrics["accuracy"]),
            _pct(scores.metrics["top5_accuracy"]),
            f"{scores.metrics['macro_f1']:.3f}",
            _pct(scores.metrics["mean_confidence"]),
            f"{scores.metrics['ece']:.3f}",
        ]
        for name, scores in evaluation.splits.items()
    ]
    unlabelled = sum(evaluation.unlabelled_records.values())
    unseen = sum(evaluation.unseen_class_records.values())
    return (
        "## Test results\n\n"
        "Measured once, on test sets the model and its selection never saw. "
        "*Accuracy* counts every record equally; *macro-F1* gives every class "
        "equal weight, so rare groups count as much as common ones.\n\n"
        + _table(
            [
                "Test set",
                "Records",
                "Accuracy",
                "Top-5",
                "Macro-F1",
                "Confidence",
                "ECE",
            ],
            rows,
        )
        + f"\n\n{unlabelled:,} test records had no group or order label, and "
        f"{unseen:,} belonged to classes too rare to train on; these were not "
        "scored."
    )


def _fragments_section(reports: _Reports) -> str:
    fragments = reports.evaluation.fragments
    names = [n for n in ("test_known", "test_novel_clean") if n in fragments]
    points = {name: _fragment_points(fragments[name]) for name in names}
    rows = []
    for index, point in enumerate(points[names[0]]):
        row = [str(point.length) if point.length else "Whole"]
        for name in names:
            metrics = points[name][index].metrics
            row += [_pct(metrics["accuracy"]), _pct(metrics["mean_confidence"])]
        rows.append(row)
    header = ["Length"]
    for name in names:
        header += [f"{_SET_NAMES[name]}: accuracy", "confidence"]
    fraction = reports.evaluation.rules.fragment_fraction
    return (
        "## How short can you go?\n\n"
        f"A random {_pct(fraction, 0)} of each test set was cut to each length "
        "at a random position, and predicted again.\n\n"
        f"![Accuracy by sequence length]({FRAGMENTS_CHART})\n\n" + _table(header, rows)
    )


def _reliability_section(reports: _Reports) -> str:
    novel = reports.evaluation.splits["test_novel_clean"]
    rows = [
        [_pct(point.threshold, 0), _pct(point.coverage), _pct(point.accuracy)]
        for point in novel.coverage
    ]
    return (
        "## Can the confidence be trusted?\n\n"
        "Each point is a group of predictions with similar confidence; on the "
        "diagonal, confidence matches how often the model is right.\n\n"
        f"![Reliability chart]({RELIABILITY_CHART})\n\n"
        "Accepting only predictions at or above a confidence threshold (novel "
        "species, clean):\n\n"
        + _table(["Threshold", "Share accepted", "Accuracy of those"], rows)
    )


def _ambiguity_section(reports: _Reports) -> str:
    splits = reports.evaluation.splits
    names = [n for n in ("test_known", "test_novel_clean") if n in splits]
    bins = {name: splits[name].by_ambiguity for name in names}
    rows = []
    for index, first in enumerate(bins[names[0]]):
        row = [first.bin]
        for name in names:
            found = bins[name][index]
            row += [
                f"{found.records:,}",
                _pct(found.accuracy) if found.records else "n/a",
            ]
        rows.append(row)
    header = ["Ambiguous share"]
    for name in names:
        header += [f"{_SET_NAMES[name]}: records", "accuracy"]
    return (
        "## Ambiguous bases\n\n"
        "Scores by the share of a sequence's bases that are ambiguous (such as "
        "`N`).\n\n" + _table(header, rows)
    )


def _hardest_section(reports: _Reports) -> str:
    scores = reports.evaluation.per_class.get("test_novel_clean", {})
    candidates = [
        (label, score)
        for label, score in scores.items()
        if score.support >= _MIN_SUPPORT
    ]
    hardest = sorted(candidates, key=lambda item: item[1].f1)[:_HARDEST]
    rows = [
        [
            label,
            f"{s.support:,}",
            f"{s.precision:.3f}",
            f"{s.recall:.3f}",
            f"{s.f1:.3f}",
        ]
        for label, s in hardest
    ]
    return (
        "## Hardest classes\n\n"
        f"The {_HARDEST} classes with the lowest F1 on novel species (clean), "
        f"among those with at least {_MIN_SUPPORT} test records. *Precision*: "
        "of the records predicted as this class, the share that were; "
        "*recall*: of this class's records, the share found.\n\n"
        + _table(["Class", "Records", "Precision", "Recall", "F1"], rows)
    )


def _limitations_section(config: AppConfig, reports: _Reports) -> str:
    serving = config.router.serving
    known = {
        p.length: p for p in reports.evaluation.fragments.get("test_novel_clean", [])
    }
    shortest = known.get(serving.min_length)
    at_minimum = (
        f" (at {serving.min_length} bases, {_pct(shortest.metrics['accuracy'], 0)} "
        "accurate for novel species)"
        if shortest
        else ""
    )
    return f"""## Limitations

- **Short sequences** are much less accurate{at_minimum}; below
  {serving.min_length} bases, confidence is also too high, so they are declined.
- **Rare groups** have few reference records, so their scores vary more
  (see *Hardest classes*). Groups too rare to train on are never predicted.
- **The reference data reflects what has been sequenced**: GenBank holds far
  more records for some groups and regions than others, and some records
  are misidentified.
- **Only COI** sequences are supported; other markers give meaningless
  results.
- **Confidence** is calibrated on validation data like the test data; on
  sequences unlike the reference data (contaminated, chimeric, or from a
  poorly sampled group) it may be too high."""


def _provenance_section(reports: _Reports) -> str:
    evaluation = reports.evaluation
    rows = [
        ["Training run", f"`{reports.train.parent_run_id}`"],
        ["Evaluation run", f"`{evaluation.run_id}`"],
        ["Test data SHA-256", f"`{evaluation.data_sha256.get('records', '')[:16]}…`"],
        ["Training sample SHA-256", f"`{reports.train.sample_sha256['train'][:16]}…`"],
        ["Evaluated", evaluation.finished_at.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    if reports.calibration is not None:
        rows.insert(1, ["Calibration run", f"`{reports.calibration.run_id}`"])
    return "## Provenance\n\n" + _table(["", ""], rows)


def build_router_card(config: AppConfig) -> CardFiles:
    """Write the router's model card and its two charts.

    Raises:
        CardError: If a report is missing, or the reports describe different
            model versions.
    """
    reports = _read(config)
    card = config.router.card
    card.output_dir.mkdir(parents=True, exist_ok=True)
    fragments_chart = card.output_dir / FRAGMENTS_CHART
    reliability_path = card.output_dir / RELIABILITY_CHART
    fragments_chart.write_text(
        _fragments_chart(reports, config.router.serving.min_length), "utf-8"
    )
    reliability_path.write_text(_reliability_chart(reports), "utf-8")

    sections = [
        f"# {config.title}: router model card",
        "<!-- Generated by `prokopton-ml card` from the pipeline's reports. "
        "Do not edit by hand; edit the notes file or the code instead. -->",
        _model_section(config, reports),
        _use_section(config),
        _data_section(reports),
        _training_section(reports),
        _calibration_section(reports),
        _results_section(reports),
        _fragments_section(reports),
        _reliability_section(reports),
        _ambiguity_section(reports),
        _hardest_section(reports),
        _limitations_section(config, reports),
    ]
    if card.notes is not None:
        if not card.notes.exists():
            msg = f"{card.notes} not found"
            raise CardError(msg)
        notes = card.notes.read_text("utf-8").strip()
        sections.append(f"## Development notes\n\n{notes}")
    sections.append(_provenance_section(reports))
    output = card.output_dir / CARD_NAME
    output.write_text("\n\n".join(sections) + "\n", "utf-8")
    logger.info("Wrote %s, %s and %s", output, fragments_chart, reliability_path)
    return CardFiles(
        card=output,
        fragments_chart=fragments_chart,
        reliability_chart=reliability_path,
        model_version=reports.evaluation.model_version,
    )
