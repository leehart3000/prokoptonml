"""Build a specialist's model card from its reference and evaluation reports.

Like the classifier cards, a specialist card is generated, never edited by
hand. It reads ``_reference_report.json`` and ``_evaluation_report.json``,
and refuses to build unless the thresholds in the app config are the ones
the evaluation chose, so the card always describes what is served. A chart
of the threshold trade-off is written beside it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from prokopton.ml.config import AppConfig, SpecialistConfig
from prokopton.ml.pipeline.reference import REPORT_NAME as REFERENCE_REPORT
from prokopton.ml.pipeline.reference import ReferenceReport, reference_path
from prokopton.ml.pipeline.specialist_evaluate import (
    REPORT_NAME as EVALUATION_REPORT,
)
from prokopton.ml.pipeline.specialist_evaluate import (
    SetScore,
    SpecialistEvaluationReport,
)
from prokopton.ml.reporting.charts import Series, line_chart

__all__ = ["SpecialistCardError", "SpecialistCardFiles", "build_specialist_card"]

logger = logging.getLogger(__name__)

_SETS = {
    "test_known": "Known species",
    "test_novel_clean": "Held-out species, no near-copy in training",
    "test_novel_leaky": "Held-out species with a near-copy in training",
    "test_novel": "Held-out species (all)",
    "test_background": "Other flies (should be kept out)",
}

_OUTCOMES = [
    ("species", "One named species in our reference data matches best, and clearly."),
    (
        "closest_species",
        "As for a species, but other species in our data are as close to it "
        "as this sequence is, so it may be a related species missing from "
        "our data.",
    ),
    (
        "complex",
        "The best matches belong to a known species complex that COI cannot "
        "separate; the members and how to resolve them are given.",
    ),
    ("species_group", "Several named species match equally well."),
    (
        "genus_label_only",
        "The closest records are labelled only to genus in our data.",
    ),
    ("genus", "The closest matches share a genus."),
    ("family", "A member of the group, but not close enough for a genus."),
    ("not_member", "Not a member of the group; the router's answer is kept."),
]


class SpecialistCardError(RuntimeError):
    """Raised when the reports are missing or do not match the config."""


@dataclass(frozen=True, slots=True)
class SpecialistCardFiles:
    """The files written."""

    card: Path
    thresholds_chart: Path


def _count(number: int, noun: str) -> str:
    plural = {"family": "families", "named species": "named species"}
    return f"{number:,} {noun if number == 1 else plural.get(noun, noun + 's')}"


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(row) + " |" for row in rows])


def _load[R: BaseModel](path: Path, model: type[R]) -> R:
    if not path.exists():
        msg = f"{path} not found; run the reference and evaluate stages first"
        raise SpecialistCardError(msg)
    try:
        return model.model_validate_json(path.read_text("utf-8"))
    except ValidationError as error:
        msg = f"{path} is not a valid report: {error}"
        raise SpecialistCardError(msg) from error


def _read(
    config: AppConfig, name: str
) -> tuple[ReferenceReport, SpecialistEvaluationReport]:
    folder = reference_path(config, name).parent
    return (
        _load(folder / REFERENCE_REPORT, ReferenceReport),
        _load(folder / EVALUATION_REPORT, SpecialistEvaluationReport),
    )


def _check_thresholds(
    rules: SpecialistConfig, evaluation: SpecialistEvaluationReport
) -> None:
    chosen = evaluation.chosen
    configured = (rules.species_identity, rules.group_margin, rules.member_identity)
    tuned = (chosen.species_identity, chosen.group_margin, chosen.member_identity)
    if configured != tuned:
        msg = (
            "The app config's thresholds (species_identity, group_margin, "
            f"member_identity) are {configured}, but the evaluation chose "
            f"{tuned}; set the chosen values in the config, or evaluate again"
        )
        raise SpecialistCardError(msg)


def _chart(evaluation: SpecialistEvaluationReport) -> str:
    chosen = evaluation.chosen
    points = sorted(
        (
            p
            for p in evaluation.grid
            if p.thresholds.group_margin == chosen.group_margin
            and p.thresholds.member_identity == chosen.member_identity
        ),
        key=lambda p: p.thresholds.species_identity,
    )
    xs = [p.thresholds.species_identity * 100 for p in points]

    def values(set_name: str, metric: str) -> list[float]:
        return [float(getattr(p.sets[set_name], metric)) for p in points]

    return line_chart(
        "Species threshold trade-off (validation, averaged over species)",
        xs,
        [f"{x:g}%" for x in xs],
        [
            Series(
                "Known species named exactly",
                values("val_known", "exact_by_species"),
            ),
            Series(
                "Held-out species wrongly named",
                values("val_novel", "wrongly_named_by_species"),
            ),
        ],
        x_label="Species identity threshold",
        y_label="Share of species",
        marker_x=chosen.species_identity * 100,
        marker_label="chosen",
    )


def _row(label: str, score: SetScore) -> list[str]:
    return [
        label,
        f"{score.records:,}",
        f"{score.species:,}",
        f"{_pct(score.right)} ({_pct(score.right_by_species)})",
        f"{_pct(score.wrongly_named)} ({_pct(score.wrongly_named_by_species)})",
        f"{_pct(score.exact)} ({_pct(score.exact_by_species)})",
    ]


def _sections(
    rules: SpecialistConfig,
    reference: ReferenceReport,
    evaluation: SpecialistEvaluationReport,
    chart: str,
) -> list[str]:
    card = rules.card
    settings = rules.evaluation
    chosen = evaluation.chosen
    model = (
        "## Model\n\n"
        f"- **Task:** {card.task}\n"
        f"- **Input:** {card.input}\n"
        f"- **Method:** the query is compared with every reference sequence by "
        f"k-mer similarity (k = {rules.k}); the {rules.candidates} closest "
        f"(at most {rules.candidates_per_species} per species) are aligned, "
        "and percent identity decides the outcome. There is no trained model: "
        "the reference set and the thresholds are the model.\n"
        f"- **Thresholds (tuned):** species {_pct(chosen.species_identity)}, "
        f"group margin {_pct(chosen.group_margin)}, membership "
        f"{_pct(chosen.member_identity)}; genus {_pct(rules.genus_identity)}.\n"
        f"- **Data:** MIDORI2 {evaluation.release}, gene {evaluation.gene}."
    )
    outcomes = (
        "## Answers\n\nEvery answer describes our reference data and this "
        "marker, never the organism's scientific status.\n\n"
        + _table(["Outcome", "Meaning"], [[f"`{o}`", t] for o, t in _OUTCOMES])
    )
    complexes = "\n".join(
        f"- **{c.name}** ({len(c.members)} names): {c.resolve} _{c.reference}_"
        for c in rules.complexes
    )
    data = (
        "## Reference data\n\n"
        f"From the training split only: {_count(reference.target_records, 'record')}"
        f" of {rules.target} ({_count(reference.target_named_species, 'named species')}"
        f", up to {rules.per_species} distinct sequences each; "
        f"{_count(reference.duplicates_skipped, 'repeated sequence')} skipped), and "
        f"{_count(reference.background_records, 'record')} of other "
        f"{rules.background} in {_count(reference.background_families, 'family')} "
        "as the background."
    )
    if reference.contradicted_skipped:
        data += (
            f" {_count(reference.contradicted_skipped, 'record')} labelled as one "
            "species but far more like another were left out as likely "
            "mislabelled (listed with the reference set)."
        )
    if reference.species_with_neighbours:
        data += (
            f"\n\n**{reference.species_within_species_identity} of "
            f"{_count(reference.species_with_neighbours, 'named species')}** have "
            "another named species at least "
            f"{_pct(reference.rules.species_identity)} identical in the reference "
            "set: COI does not separate them well. Such species are reported as a "
            "group or a closest match, not alone."
        )
    if rules.complexes:
        data += (
            "\n\nSpecies complexes, reported as the complex with how to resolve "
            f"it:\n\n{complexes}"
        )
    met = (
        "Both limits were met."
        if evaluation.constraints_met
        else (
            "**Not every limit was met** (background kept out: "
            f"{evaluation.background_ok}; known species limit met: "
            f"{evaluation.known_ok}); the best combination was used."
        )
    )
    tuning = (
        "## Tuning\n\n"
        f"Up to {settings.per_species} records per species were drawn from the "
        "validation sets. Each was aligned once, then "
        f"{len(evaluation.grid)} threshold combinations were scored. In order, "
        "the rule prefers: letting in at most "
        f"{_pct(settings.max_background_admitted)} of other flies; wrongly naming "
        f"known species at most {_pct(settings.max_known_wrong)} of the time "
        "(averaged over species); the most known species named exactly minus "
        "held-out species wrongly named; and then the most held-out species "
        f"right. {met}\n\n"
        f"![Species threshold trade-off]({chart})"
    )
    rows = [
        _row(label, evaluation.test[key])
        for key, label in _SETS.items()
        if key in evaluation.test
    ]
    known = evaluation.test.get("test_known")
    mix = ", ".join(f"{o} {n}" for o, n in known.outcomes.items()) if known else ""
    results = (
        "## Test results\n\n"
        "Measured once, with the tuned thresholds. Rates are shares of records "
        "(in brackets: averaged over species). *Right* means right at the "
        "level given (a group or complex is right if it contains the species); "
        "*wrongly named* means a confident wrong species, group or complex (for "
        "other flies: let in as mosquitoes); *exact* means named as exactly "
        "the right species.\n\n"
        + _table(["Set", "Records", "Species", "Right", "Wrongly named", "Exact"], rows)
        + (f"\n\nAnswers for known species: {mix}." if mix else "")
    )
    limitations = "## Limitations\n\n" + "\n".join(
        f"- {item}" for item in card.limitations
    )
    use = "## Intended use\n\n" + card.intended_use.strip()
    sections = [model, use, outcomes, data, tuning, results, limitations]
    if card.notes is not None:
        if not card.notes.exists():
            msg = f"{card.notes} not found"
            raise SpecialistCardError(msg)
        sections.append(
            "## Development notes\n\n" + card.notes.read_text("utf-8").strip()
        )
    sections.append(
        "## Provenance\n\n"
        + _table(
            ["", ""],
            [
                ["Evaluation run", f"`{evaluation.run_id}`"],
                [
                    "Reference set SHA-256",
                    f"`{evaluation.data_sha256['reference'][:16]}…`",
                ],
                ["Split data SHA-256", f"`{evaluation.data_sha256['records'][:16]}…`"],
                ["Evaluated", evaluation.finished_at.strftime("%Y-%m-%d %H:%M UTC")],
                ["prokopton-ml", evaluation.prokopton_ml_version],
            ],
        )
    )
    return sections


def build_specialist_card(config: AppConfig, name: str) -> SpecialistCardFiles:
    """Write a specialist's model card and its threshold chart.

    The files are ``<name>.md`` and ``<name>-thresholds.svg`` in the card's
    output folder.

    Raises:
        SpecialistCardError: If a report is missing or invalid, or the
            config's thresholds are not the ones the evaluation chose.
    """
    rules = config.specialist(name)
    reference, evaluation = _read(config, name)
    _check_thresholds(rules, evaluation)
    card = rules.card
    card.output_dir.mkdir(parents=True, exist_ok=True)
    chart = card.output_dir / f"{name}-thresholds.svg"
    chart.write_text(_chart(evaluation), "utf-8")
    sections = [
        f"# {config.title}: {card.title}",
        "<!-- Generated by `prokopton-ml card --specialist` from the pipeline's "
        "reports. Do not edit by hand; edit the app config, the notes file or "
        "the code instead. -->",
        *_sections(rules, reference, evaluation, chart.name),
    ]
    output = card.output_dir / f"{name}.md"
    output.write_text("\n\n".join(sections) + "\n", "utf-8")
    logger.info("Wrote %s and %s", output, chart)
    return SpecialistCardFiles(card=output, thresholds_chart=chart)
