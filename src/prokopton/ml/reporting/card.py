"""Build a classifier's model card from the pipeline's reports.

A model card explains what a model is for, how it was made, how well it
works and where it does not. Cards are generated (never edited by hand) from
the sample, train, calibration and evaluation reports, so they always match
the registered model. The description of the model's task and intended use,
and any extra limitations, come from the app config; hand-written notes can
be included from a Markdown file. Two SVG charts are written beside it.
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

__all__ = ["CardError", "CardFiles", "build_card"]

logger = logging.getLogger(__name__)

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
    # Display names of the test sets
    names: dict[str, str]


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


def _phrase(reports: _Reports, name: str) -> str:
    """A test set's name for use inside a sentence."""
    shown = reports.names[name]
    return "the test set" if shown == "Test set" else shown.lower()


def _main_set(reports: _Reports) -> str:
    """The test set that best shows real use: clean novel species if any."""
    splits = reports.evaluation.splits
    return "test_novel_clean" if "test_novel_clean" in splits else "test_known"


def _read(config: AppConfig, model: str) -> _Reports:
    folder = f"midori2_{config.data.source.release.lower()}"
    model_dir = config.data.paths.models_dir / model
    paths = {
        "sample": config.data.paths.samples_dir / folder / model / sample.REPORT_NAME,
        "train": model_dir / train.REPORT_NAME,
        "evaluation": model_dir / evaluate.REPORT_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        msg = f"{', '.join(missing)} not found; run the pipeline first"
        raise CardError(msg)
    calibration_path = model_dir / calibrate.REPORT_NAME
    evaluation = EvaluationReport.model_validate_json(
        paths["evaluation"].read_text("utf-8")
    )
    names = dict(_SET_NAMES)
    if not any(name.startswith("test_novel") for name in evaluation.splits):
        # Without held-out species, "known species" would be misleading
        names["test_known"] = "Test set"
    reports = _Reports(
        sample=SampleReport.model_validate_json(paths["sample"].read_text("utf-8")),
        train=TrainReport.model_validate_json(paths["train"].read_text("utf-8")),
        calibration=(
            CalibrationReport.model_validate_json(calibration_path.read_text("utf-8"))
            if calibration_path.exists()
            else None
        ),
        evaluation=evaluation,
        names=names,
    )
    made_for = {
        "sample": reports.sample.model,
        "train": reports.train.model,
        "evaluation": reports.evaluation.model,
    }
    if reports.calibration is not None:
        made_for["calibration"] = reports.calibration.model
    wrong = sorted(name for name, found in made_for.items() if found != model)
    if wrong:
        msg = f"The {', '.join(wrong)} report(s) are not for the {model!r} model"
        raise CardError(msg)
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
            reports.names[name],
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
    main = _main_set(reports)
    bins = reports.evaluation.splits[main].calibration
    return reliability_chart(
        f"Reliability: {_phrase(reports, main).removeprefix('the ')}, whole sequences",
        [b.mean_confidence for b in bins],
        [b.accuracy for b in bins],
        [b.records for b in bins],
    )


def _training_data(config: AppConfig, model: str) -> str:
    source = config.data.source
    genes = config.classifier(model).genes
    covered = ", ".join(genes) if len(genes) <= 3 else f"{len(genes)} genes"
    link = f"[doi:{source.doi}](https://doi.org/{source.doi})"
    return f"MIDORI2 {source.release} ({covered}), {link}, {source.licence}"


def _output(reports: _Reports) -> str:
    labels = f"One of {len(reports.sample.labels)} labels"
    reverse = reports.train.rules.reverse_label
    if reverse:
        labels += f" (or *{reverse}* for a sequence given as the other strand)"
    return f"{labels}, with a confidence"


def _model_section(config: AppConfig, model_name: str, reports: _Reports) -> str:
    card = config.classifier(model_name).card
    train_report = reports.train
    evaluation = reports.evaluation
    kind = {"logistic": "Logistic regression", "naive_bayes": "Naive Bayes"}
    model = kind.get(train_report.best_model, train_report.best_model)
    calibration = reports.calibration
    rows = [
        ["Registered model", f"`{evaluation.registered_model}`"],
        ["Version", f"{evaluation.model_version} (alias `@{evaluation.alias}`)"],
        ["Task", card.task],
        ["Input", card.input],
        ["Output", _output(reports)],
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
            _training_data(config, model_name),
        ],
        ["Built with", f"prokopton-ml {evaluation.prokopton_ml_version}"],
    ]
    return "## Model\n\n" + _table(["", ""], rows)


def _use_section(config: AppConfig, model: str) -> str:
    classifier = config.classifier(model)
    serving = classifier.serving
    text = classifier.card.intended_use.strip().format(
        min_length=serving.min_length, short_length=serving.short_length
    )
    return f"## Intended use\n\n{text}" if text else ""


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
    novel = any(name.startswith("test_novel") for name in reports.evaluation.splits)
    splitting = (
        "Records were split by species: *known* species appear in training, "
        "while *novel* species are held out entirely, so their scores show how "
        "the model handles species it has never seen. Novel-species records "
        "with a near-copy (99% identity or more) in training were found with "
        "VSEARCH; *clean* records have none, and are the honest measure."
        if novel
        else "Records were split into training, validation and test sets; "
        "identical sequences always fall in the same set."
    )
    spread = {
        "species": "species",
        "order": "orders",
        "group_order": "groups of organisms, then orders,",
    }[rules.diversity]
    return (
        "## Data\n\n"
        f"{splitting}\n\n"
        f"Training samples were capped at {rules.train_cap:,} records per class "
        f"(validation: {rules.eval_cap:,}), spread across as many {spread} as "
        "possible. "
        f"Classes with under {rules.min_class_records} records were left out "
        f"({len(sample_report.dropped_classes):,} of them), and "
        f"{sample_report.excluded_near_duplicates:,} validation records with a "
        "near-copy in training were removed.\n\n"
        + _table(["Sample", "Records", "Classes", "Species per class"], rows)
    )


def _training_section(reports: _Reports, note: str) -> str:
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
        "compared on the validation samples (never the test sets), and the "
        "best was chosen by the mean macro-F1 across them. "
        f"{note.strip() + ' ' if note.strip() else ''}"
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
        f"T = exp({calibration.a:.4f} {'-' if calibration.b < 0 else '+'} "
        f"{abs(calibration.b):.4f} * ln(length)), "
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
            reports.names.get(name, name),
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
        + (
            f"\n\n{unlabelled:,} test records had no group or order label, and "
            f"{unseen:,} belonged to classes too rare to train on; these were "
            "not scored."
            if unlabelled or unseen
            else ""
        )
    )


def _fragments_section(reports: _Reports, chart: str) -> str:
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
        header += [f"{reports.names[name]}: accuracy", "confidence"]
    fraction = reports.evaluation.rules.fragment_fraction
    return (
        "## How short can you go?\n\n"
        f"A random {_pct(fraction, 0)} of each test set was cut to each length "
        "at a random position, and predicted again.\n\n"
        f"![Accuracy by sequence length]({chart})\n\n" + _table(header, rows)
    )


def _reliability_section(reports: _Reports, chart: str) -> str:
    main = _main_set(reports)
    novel = reports.evaluation.splits[main]
    rows = [
        [_pct(point.threshold, 0), _pct(point.coverage), _pct(point.accuracy)]
        for point in novel.coverage
    ]
    return (
        "## Can the confidence be trusted?\n\n"
        "Each point is a group of predictions with similar confidence; on the "
        "diagonal, confidence matches how often the model is right.\n\n"
        f"![Reliability chart]({chart})\n\n"
        "Accepting only predictions at or above a confidence threshold "
        f"({_phrase(reports, main)}):\n\n"
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
        header += [f"{reports.names[name]}: records", "accuracy"]
    return (
        "## Ambiguous bases\n\n"
        "Scores by the share of a sequence's bases that are ambiguous (such as "
        "`N`).\n\n" + _table(header, rows)
    )


def _hardest_section(reports: _Reports) -> str:
    main = _main_set(reports)
    scores = reports.evaluation.per_class.get(main, {})
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
        f"The {_HARDEST} classes with the lowest F1 on {_phrase(reports, main)}, "
        f"among those with at least {_MIN_SUPPORT} test records. *Precision*: "
        "of the records predicted as this class, the share that were; "
        "*recall*: of this class's records, the share found.\n\n"
        + _table(["Class", "Records", "Precision", "Recall", "F1"], rows)
    )


def _limitations_section(config: AppConfig, model: str, reports: _Reports) -> str:
    classifier = config.classifier(model)
    serving = classifier.serving
    main = _main_set(reports)
    lengths = {p.length: p for p in reports.evaluation.fragments.get(main, [])}
    shortest = lengths.get(serving.min_length)
    at_minimum = (
        f" (at {serving.min_length} bases, {_pct(shortest.metrics['accuracy'], 0)} "
        f"accurate on {_phrase(reports, main)})"
        if shortest
        else ""
    )
    items = [
        f"**Short sequences** are much less accurate{at_minimum}; below "
        f"{serving.min_length} bases they are declined.",
        "**Rare classes** have few reference records, so their scores vary "
        "more (see *Hardest classes*). Classes too rare to train on are never "
        "predicted.",
        "**The reference data reflects what has been sequenced**: GenBank "
        "holds far more records for some groups and regions than others, and "
        "some records are misidentified.",
        *classifier.card.limitations,
        "**Confidence** is calibrated on validation data like the test data; "
        "on sequences unlike the reference data (contaminated, chimeric, or "
        "from a poorly sampled group) it may be too high.",
    ]
    return "## Limitations\n\n" + "\n".join(f"- {item}" for item in items)


def _groups_section(reports: _Reports) -> str:
    splits = reports.evaluation.splits
    names = [n for n in ("test_known", "test_novel_clean") if n in splits]
    if not any(splits[name].by_group for name in names):
        return ""
    rows = [
        [
            reports.names[name],
            group.group,
            f"{group.records:,}",
            _pct(group.accuracy),
            f"{group.macro_f1:.3f}",
        ]
        for name in names
        for group in splits[name].by_group
    ]
    return (
        "## By group of organisms\n\n"
        "Scores for each broad group, to show the model works across them "
        "rather than relying on the groups each class is most common in.\n\n"
        + _table(["Test set", "Group", "Records", "Accuracy", "Macro-F1"], rows)
    )


def _orientation_section(reports: _Reports) -> str:
    fragments = reports.evaluation.fragments
    names = [
        name
        for name, points in fragments.items()
        if any(p.forward_strand_share is not None for p in points)
    ]
    if not names:
        return ""
    points = {name: _fragment_points(fragments[name]) for name in names}
    rows = []
    for index, point in enumerate(points[names[0]]):
        row = [str(point.length) if point.length else "Whole"]
        for name in names:
            share = points[name][index].forward_strand_share
            row.append(_pct(share) if share is not None else "n/a")
        rows.append(row)
    return (
        "## Which strand?\n\n"
        "A sequence may be given as either strand. Each test fragment was also "
        "predicted as its reverse complement (the other strand); this is how "
        "often the original strand got the more confident prediction, so "
        "choosing the more confident strand finds the right one.\n\n"
        + _table(["Length", *[reports.names[name] for name in names]], rows)
    )


def _first_records_sha(evaluation: EvaluationReport) -> str:
    keys = sorted(k for k in evaluation.data_sha256 if k.startswith("records"))
    return evaluation.data_sha256[keys[0]] if keys else ""


def _provenance_section(reports: _Reports) -> str:
    evaluation = reports.evaluation
    rows = [
        ["Training run", f"`{reports.train.parent_run_id}`"],
        ["Evaluation run", f"`{evaluation.run_id}`"],
        ["Test data SHA-256", f"`{_first_records_sha(evaluation)[:16]}…`"],
        ["Training sample SHA-256", f"`{reports.train.sample_sha256['train'][:16]}…`"],
        ["Evaluated", evaluation.finished_at.strftime("%Y-%m-%d %H:%M UTC")],
    ]
    if reports.calibration is not None:
        rows.insert(1, ["Calibration run", f"`{reports.calibration.run_id}`"])
    return "## Provenance\n\n" + _table(["", ""], rows)


def build_card(config: AppConfig, model: str = "router") -> CardFiles:
    """Write a classifier's model card and its two charts.

    The files are named after the model: ``<model>.md``,
    ``<model>-fragments.svg`` and ``<model>-reliability.svg``.

    Raises:
        CardError: If a report is missing, or the reports describe different
            model versions.
    """
    classifier = config.classifier(model)
    reports = _read(config, model)
    card = classifier.card
    card.output_dir.mkdir(parents=True, exist_ok=True)
    fragments_chart = card.output_dir / f"{model}-fragments.svg"
    reliability_path = card.output_dir / f"{model}-reliability.svg"
    fragments_chart.write_text(
        _fragments_chart(reports, classifier.serving.min_length), "utf-8"
    )
    reliability_path.write_text(_reliability_chart(reports), "utf-8")

    sections = [
        f"# {config.title}: {card.title}",
        "<!-- Generated by `prokopton-ml card` from the pipeline's reports. "
        "Do not edit by hand; edit the app config, the notes file or the code "
        "instead. -->",
        _model_section(config, model, reports),
        _use_section(config, model),
        _data_section(reports),
        _training_section(reports, card.training_note),
        _calibration_section(reports),
        _results_section(reports),
        _groups_section(reports),
        _fragments_section(reports, fragments_chart.name),
        _orientation_section(reports),
        _reliability_section(reports, reliability_path.name),
        _ambiguity_section(reports),
        _hardest_section(reports),
        _limitations_section(config, model, reports),
    ]
    if card.notes is not None:
        if not card.notes.exists():
            msg = f"{card.notes} not found"
            raise CardError(msg)
        notes = card.notes.read_text("utf-8").strip()
        sections.append(f"## Development notes\n\n{notes}")
    sections.append(_provenance_section(reports))
    output = card.output_dir / f"{model}.md"
    text = "\n\n".join(section for section in sections if section)
    output.write_text(text + "\n", "utf-8")
    logger.info("Wrote %s, %s and %s", output, fragments_chart, reliability_path)
    return CardFiles(
        card=output,
        fragments_chart=fragments_chart,
        reliability_chart=reliability_path,
        model_version=reports.evaluation.model_version,
    )
