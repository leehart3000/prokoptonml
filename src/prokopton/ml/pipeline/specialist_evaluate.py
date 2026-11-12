"""Specialist evaluation: tune a specialist's thresholds, then measure it.

A specialist is evaluated on records it has never seen, drawn from the
validation and test splits of its gene:

- **known**: records of target species that are in the reference set;
- **novel_clean** and **novel_leaky**: records of held-out target species,
  split by whether the leakage stage found a near-copy (at least 99%
  identical) in training. A leaky record's nearest training sequence is
  usually a sister species, so naming that sister is hard to avoid;
- **background**: records from the wider group (for example, other flies),
  which should all be kept out.

Up to ``per_species`` records of each species are drawn per set (chosen by
a salted hash), so species with many records do not dominate, and scores
are given both per record and averaged over species. Only target records
with a species name are used, since the right answer must be known; their
names are made consistent in the same way as the reference set's.

Each record is aligned once (in parallel processes). The thresholds in the
grid are then all tried on the saved matches of the validation sets, and
chosen by the rule in ``SpecialistEvaluationConfig``. The test sets are
scored once, with the chosen thresholds. Wrong answers with the chosen
thresholds are listed (up to ``EXAMPLES`` per set), with the closest
reference of the record's own species, to help find their causes. The run
is recorded in MLflow. Outputs
``<models_dir>/specialists/<name>/_evaluation_report.json``.
"""

import heapq
import itertools
import logging
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, SpecialistConfig
from prokopton.ml.data.checksums import sha256_of_file
from prokopton.ml.pipeline.reference import reference_path
from prokopton.ml.pipeline.split import hash_fraction
from prokopton.ml.specialists.identify import (
    NEIGHBOURS_NAME,
    Identification,
    Match,
    Specialist,
    canonical_genus,
    canonical_species,
)
from prokopton.ml.training.tracking import Tracker, specialist_tracker

__all__ = [
    "EXAMPLES",
    "NAMED",
    "REPORT_NAME",
    "GridPoint",
    "SetScore",
    "SpecialistEvaluationError",
    "SpecialistEvaluationReport",
    "Thresholds",
    "WrongAnswer",
    "evaluate_specialist",
    "is_right",
    "score_set",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_evaluation_report.json"
BATCH_SIZE = 100_000
EXAMPLES = 40
"""Wrong answers listed per set."""
CHUNK = 32
"""Queries sent to a worker process at a time."""
PROGRESS_EVERY = 25
"""Log progress after this many chunks."""
NAMED = ("species", "complex", "species_group")
"""Outcomes that name species with confidence."""
SPLITS = {"val": ("val_known", "val_novel"), "test": ("test_known", "test_novel")}
_COLUMNS = ["record_id", "split", "order", "family", "genus", "species"]
_COLUMNS += ["species_named", "ambiguous_fraction", "sequence"]
_TEST_METRICS = (
    *("right", "right_by_species", "wrongly_named"),
    *("wrongly_named_by_species", "exact", "exact_by_species"),
)


class SpecialistEvaluationError(RuntimeError):
    """Raised when the specialist evaluation stage cannot run."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Thresholds(_Report):
    """The thresholds tuned by this stage."""

    species_identity: float
    group_margin: float
    member_identity: float


class SetScore(_Report):
    """How a specialist did on one set of records.

    Rates are shares of records; ``*_by_species`` rates are averaged over
    species, so each species counts equally.
    """

    records: int
    species: int
    right: float
    right_by_species: float
    wrongly_named: float
    """Target records given a wrong species, complex or group with
    confidence (for background sets: records let in as members)."""
    wrongly_named_by_species: float
    exact: float
    """Target records named as exactly the right species."""
    exact_by_species: float
    outcomes: dict[str, int]


class GridPoint(_Report):
    """The validation scores for one combination of thresholds."""

    thresholds: Thresholds
    background_ok: bool
    known_ok: bool
    objective: float
    sets: dict[str, SetScore]


class WrongAnswer(_Report):
    """A wrong answer, with what is needed to find its cause."""

    record_id: str
    species: str | None
    outcome: str
    label: str | None
    identity: float
    best_record_id: str | None
    best_species: str | None
    own_species_identity: float | None
    """The closest aligned reference of the record's own species, if any
    was among the candidates."""


class SpecialistEvaluationReport(_Report):
    """Provenance, tuning and test results of a specialist."""

    app: str
    specialist: str
    release: str
    gene: str
    prokopton_ml_version: str
    started_at: datetime
    finished_at: datetime
    data_sha256: dict[str, str]
    starting: Thresholds
    chosen: Thresholds
    constraints_met: bool
    background_ok: bool
    known_ok: bool
    grid: list[GridPoint]
    validation: dict[str, SetScore]
    test: dict[str, SetScore]
    wrong: dict[str, list[WrongAnswer]]
    output: str
    run_id: str = ""


def is_right(found: Identification, row: dict[str, Any]) -> bool:
    """Whether an answer is right at the level it is given."""
    if row["role"] == "background":
        return found.outcome == "not_member"
    if found.outcome in ("species", "closest_species"):
        return bool(found.label == row["species"])
    if found.outcome in ("complex", "species_group"):
        return bool(row["species"] in found.species)
    if found.outcome == "genus":
        return bool(found.label == row["genus"])
    if found.outcome == "genus_label_only":
        # No genus on the matching record means a family-level answer
        return bool(found.label in (None, row["genus"]))
    return found.outcome == "family"


def _wrongly_named(found: Identification, row: dict[str, Any], right: bool) -> bool:
    if row["role"] == "background":
        return not right
    return found.outcome in NAMED and not right


def score_set(
    found: Sequence[Identification], rows: Sequence[dict[str, Any]]
) -> SetScore:
    """Score a set's answers, per record and averaged over species."""
    per_species: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for answer, row in zip(found, rows, strict=True):
        right = is_right(answer, row)
        exact = answer.outcome == "species" and right
        per_species[row["species"] or row["record_id"]].append(
            (right, _wrongly_named(answer, row, right), exact)
        )
    flags = [flag for group in per_species.values() for flag in group]

    def share(index: int) -> float:
        return sum(f[index] for f in flags) / len(flags) if flags else 0.0

    def by_species(index: int) -> float:
        if not per_species:
            return 0.0
        means = [sum(f[index] for f in g) / len(g) for g in per_species.values()]
        return sum(means) / len(means)

    return SetScore(
        records=len(flags),
        species=len(per_species),
        right=share(0),
        right_by_species=by_species(0),
        wrongly_named=share(1),
        wrongly_named_by_species=by_species(1),
        exact=share(2),
        exact_by_species=by_species(2),
        outcomes=dict(Counter(a.outcome for a in found).most_common()),
    )


def _set_name(row: dict[str, Any], leaky: set[str]) -> str:
    stage = "val" if row["split"].startswith("val") else "test"
    if row["role"] == "background":
        return f"{stage}_background"
    if row["split"].endswith("known"):
        return f"{stage}_known"
    return f"{stage}_novel_{'leaky' if row['record_id'] in leaky else 'clean'}"


def _draw(
    path: Path, rules: SpecialistConfig, leaky: set[str]
) -> dict[str, list[dict[str, Any]]]:
    """Draw up to ``per_species`` records of each species for each set."""
    settings = rules.evaluation
    wanted = {split for splits in SPLITS.values() for split in splits}
    # A max-heap per (set, species) of the lowest priorities seen so far
    heaps: dict[tuple[str, str], list[tuple[float, str, dict[str, Any]]]] = defaultdict(
        list
    )
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=BATCH_SIZE, columns=_COLUMNS
    ):
        for row in batch.to_pylist():
            if row["split"] not in wanted:
                continue
            if row["ambiguous_fraction"] > rules.max_ambiguous_fraction:
                continue
            if row[rules.target_rank] == rules.target:
                if not row["species_named"]:
                    continue
                row["role"] = "target"
            elif row[rules.background_rank] == rules.background:
                row["role"] = "background"
            else:
                continue
            row["genus"] = canonical_genus(row["genus"], rules)
            row["species"] = canonical_species(row["species"], rules)
            name = _set_name(row, leaky)
            heap = heaps[name, row["species"] or row["record_id"]]
            item = (-hash_fraction(row["record_id"], settings.salt), row["record_id"])
            if len(heap) < settings.per_species:
                heapq.heappush(heap, (*item, row))
            elif item > heap[0][:2]:
                heapq.heapreplace(heap, (*item, row))
    by_set: dict[str, list[tuple[float, str, dict[str, Any]]]] = defaultdict(list)
    for (name, _), heap in heaps.items():
        by_set[name].extend(heap)
    return {
        name: [row for _, _, row in sorted(items, reverse=True)][: settings.max_records]
        for name, items in sorted(by_set.items())
    }


# Each worker process loads the specialist once, into this holder
_worker: dict[str, Specialist] = {}


def _start_worker(path: str, rules_json: str) -> None:
    _worker["specialist"] = Specialist.from_parquet(
        Path(path), SpecialistConfig.model_validate_json(rules_json)
    )


def _match_chunk(sequences: list[str]) -> list[list[tuple[int, float, float, int]]]:
    specialist = _worker["specialist"]
    return [specialist.compact(m) for m in specialist.match_many(sequences)]


def _chunks(items: Sequence[str]) -> Iterator[list[str]]:
    for start in range(0, len(items), CHUNK):
        yield list(items[start : start + CHUNK])


def _match_all(
    specialist: Specialist, path: Path, sequences: Sequence[str], workers: int
) -> list[list[Match]]:
    """Align every sequence to its candidates, in ``workers`` processes."""
    rules_json = specialist.rules.model_dump_json()
    found: list[list[tuple[int, float, float, int]]] = []
    if workers == 1:
        _start_worker(str(path), rules_json)
        results: Iterator[list[list[tuple[int, float, float, int]]]] = map(
            _match_chunk, _chunks(sequences)
        )
        found = [m for chunk in results for m in chunk]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_start_worker,
            initargs=(str(path), rules_json),
        ) as pool:
            for number, chunk in enumerate(pool.map(_match_chunk, _chunks(sequences))):
                found.extend(chunk)
                if number % PROGRESS_EVERY == PROGRESS_EVERY - 1:
                    logger.info("Aligned %d of %d", len(found), len(sequences))
    return [specialist.expand(compact) for compact in found]


def _grid(rules: SpecialistConfig) -> list[Thresholds]:
    settings = rules.evaluation
    return [
        Thresholds(species_identity=s, group_margin=g, member_identity=m)
        for s, g, m in itertools.product(
            settings.species_identity_grid,
            settings.group_margin_grid,
            settings.member_identity_grid,
        )
        if m <= rules.genus_identity <= s
    ]


def _with(rules: SpecialistConfig, thresholds: Thresholds) -> SpecialistConfig:
    return rules.model_copy(update=thresholds.model_dump())


def _score(
    specialist: Specialist,
    rules: SpecialistConfig,
    sets: dict[str, list[dict[str, Any]]],
    matches: dict[str, list[list[Match]]],
) -> dict[str, SetScore]:
    scores = {}
    for name, rows in sets.items():
        found = [specialist.decide(m, rules) for m in matches[name]]
        scores[name] = score_set(found, rows)
    # Held-out species as a whole, clean and leaky together
    for stage in SPLITS:
        parts = [f"{stage}_novel_clean", f"{stage}_novel_leaky"]
        rows = [row for part in parts for row in sets.get(part, [])]
        if rows:
            found = [
                specialist.decide(m, rules)
                for part in parts
                for m in matches.get(part, [])
            ]
            scores[f"{stage}_novel"] = score_set(found, rows)
    return scores


def _empty() -> SetScore:
    return score_set([], [])


def _choose(
    rules: SpecialistConfig, scored: list[tuple[Thresholds, dict[str, SetScore]]]
) -> tuple[list[GridPoint], GridPoint]:
    """Score the grid and choose (see ``SpecialistEvaluationConfig``).

    Keeping out the background comes first, then the limit on wrongly
    named known species, then the objective; ties go to the thresholds
    that get the most held-out species right (usually the lowest
    membership threshold that still keeps out the background).
    """
    settings = rules.evaluation
    grid = []
    novel_right = []
    for thresholds, scores in scored:
        known = scores.get("val_known", _empty())
        novel = scores.get("val_novel", _empty())
        background = scores.get("val_background", _empty())
        grid.append(
            GridPoint(
                thresholds=thresholds,
                background_ok=background.wrongly_named
                <= settings.max_background_admitted,
                known_ok=known.wrongly_named_by_species <= settings.max_known_wrong,
                objective=known.exact_by_species - novel.wrongly_named_by_species,
                sets=scores,
            )
        )
        novel_right.append(novel.right_by_species)
    best = max(
        range(len(grid)),
        key=lambda n: (
            grid[n].background_ok,
            grid[n].known_ok,
            grid[n].objective,
            novel_right[n],
        ),
    )
    return grid, grid[best]


def _wrong(
    rules: SpecialistConfig,
    specialist: Specialist,
    sets: dict[str, list[dict[str, Any]]],
    matches: dict[str, list[list[Match]]],
) -> dict[str, list[WrongAnswer]]:
    """List wrong answers with the chosen thresholds, per set."""
    found: dict[str, list[WrongAnswer]] = {}
    for name, rows in sets.items():
        listed: list[WrongAnswer] = []
        for row, aligned in zip(rows, matches[name], strict=True):
            answer = specialist.decide(aligned, rules)
            if is_right(answer, row):
                continue
            own = [
                m.identity
                for m in aligned
                if row["role"] == "target" and m.reference.species == row["species"]
            ]
            best = answer.matches[0].reference if answer.matches else None
            listed.append(
                WrongAnswer(
                    record_id=row["record_id"],
                    species=row["species"],
                    outcome=answer.outcome,
                    label=answer.label,
                    identity=answer.identity,
                    best_record_id=best.record_id if best else None,
                    best_species=best.species if best else None,
                    own_species_identity=max(own) if own else None,
                )
            )
            if len(listed) == EXAMPLES:
                break
        found[name] = listed
    return found


def evaluate_specialist(
    config: AppConfig,
    name: str,
    tracker_factory: Callable[[AppConfig], Tracker] | None = None,
) -> SpecialistEvaluationReport:
    """Tune a specialist's thresholds on validation sets, then test it.

    Raises:
        SpecialistEvaluationError: If the split, leakage or reference data
            is missing.
    """
    started_at = datetime.now(UTC)
    rules = config.specialist(name)
    make_tracker = tracker_factory or (lambda c: specialist_tracker(c, name))
    folder = f"midori2_{config.data.source.release.lower()}"
    records = (
        config.data.paths.splits_dir / folder / f"gene={rules.gene}" / "records.parquet"
    )
    leakage = (
        config.data.paths.leakage_dir
        / folder
        / f"gene={rules.gene}"
        / "near_duplicates.parquet"
    )
    reference = reference_path(config, name)
    missing = [str(p) for p in (records, leakage, reference) if not p.exists()]
    if missing:
        msg = (
            f"{', '.join(missing)} not found; run the split, leakage and "
            "reference stages first"
        )
        raise SpecialistEvaluationError(msg)
    data_sha256 = {
        "records": sha256_of_file(records),
        "near_duplicates": sha256_of_file(leakage),
        "reference": sha256_of_file(reference),
    }
    neighbours = reference.parent / NEIGHBOURS_NAME
    if neighbours.exists():
        data_sha256["neighbours"] = sha256_of_file(neighbours)
    leaky = set(
        pq.read_table(leakage, columns=["record_id"]).column("record_id").to_pylist()
    )

    specialist = Specialist.from_parquet(reference, rules)
    sets = _draw(records, rules, leaky)
    order = [(set_name, row) for set_name, rows in sets.items() for row in rows]
    logger.info(
        "Aligning %d records: %s",
        len(order),
        {set_name: len(rows) for set_name, rows in sets.items()},
    )
    aligned = _match_all(
        specialist,
        reference,
        [row["sequence"] for _, row in order],
        rules.evaluation.workers,
    )
    matches: dict[str, list[list[Match]]] = defaultdict(list)
    for (set_name, _), found in zip(order, aligned, strict=True):
        matches[set_name].append(found)

    validation = {k: v for k, v in sets.items() if k.startswith("val")}
    testing = {k: v for k, v in sets.items() if k.startswith("test")}
    scored = [
        (thresholds, _score(specialist, _with(rules, thresholds), validation, matches))
        for thresholds in _grid(rules)
    ]
    grid, point = _choose(rules, scored)
    chosen = point.thresholds
    constraints_met = point.background_ok and point.known_ok
    if not constraints_met:
        logger.warning(
            "No thresholds met every constraint (background kept out: %s; "
            "known species limit met: %s); chose the best anyway",
            point.background_ok,
            point.known_ok,
        )
    tuned = _with(rules, chosen)
    starting = Thresholds(
        species_identity=rules.species_identity,
        group_margin=rules.group_margin,
        member_identity=rules.member_identity,
    )
    output = reference.parent / REPORT_NAME
    report = SpecialistEvaluationReport(
        app=config.name,
        specialist=name,
        release=config.data.source.release,
        gene=rules.gene,
        prokopton_ml_version=__version__,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        data_sha256=data_sha256,
        starting=starting,
        chosen=chosen,
        constraints_met=constraints_met,
        background_ok=point.background_ok,
        known_ok=point.known_ok,
        grid=grid,
        validation=_score(specialist, tuned, validation, matches),
        test=_score(specialist, tuned, testing, matches),
        wrong=_wrong(tuned, specialist, sets, matches),
        output=str(output),
    )

    tracker = make_tracker(config)
    with tracker.run(f"{name}-evaluation") as run_id:
        report = report.model_copy(update={"run_id": run_id})
        tracker.set_tags(
            {
                "specialist": name,
                "release": report.release,
                **{f"sha256.{k}": v for k, v in data_sha256.items()},
            }
        )
        tracker.log_params(
            {
                **{f"chosen.{k}": v for k, v in chosen.model_dump().items()},
                "constraints_met": constraints_met,
                "grid_points": len(grid),
            }
        )
        tracker.log_metrics(
            {
                f"{set_name}.{metric}": float(getattr(score, metric))
                for set_name, score in report.test.items()
                for metric in _TEST_METRICS
            }
        )
        tracker.log_dict(report.model_dump(mode="json"), "evaluation_report.json")

    output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Chosen for %s: species_identity=%s, group_margin=%s, "
        "member_identity=%s (constraints met: %s). Set these in the app's config.",
        name,
        chosen.species_identity,
        chosen.group_margin,
        chosen.member_identity,
        constraints_met,
    )
    for set_name, score in report.test.items():
        logger.info(
            "%s: %d records, %d species; right %.1f%% (%.1f%% by species), "
            "wrongly named %.1f%% (%.1f%% by species), exact %.1f%%",
            set_name,
            score.records,
            score.species,
            score.right * 100,
            score.right_by_species * 100,
            score.wrongly_named * 100,
            score.wrongly_named_by_species * 100,
            score.exact * 100,
        )
    return report
