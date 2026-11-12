"""Identify a submitted sequence, step by step, explaining each step.

The steps are:

1. **Check the input** (see ``inputs``): one DNA sequence, cleaned.
2. **Find the barcode.** The marker classifier scores the sequence on both
   strands and keeps the strand whose best gene (other than its "reverse"
   class) is most confident. If that gene is not the barcode gene, the
   sequence is declined, saying which gene it looks like. A long sequence
   (such as a whole mitochondrial genome) is first searched for the barcode
   gene with a sliding window, and only that region is used.
3. **Route.** The router predicts the broad group (for animals, the order).
4. **Specialists.** A specialist is consulted when the router gives its
   group at least ``route_threshold`` (Rule B). The specialist checks
   membership itself (Rule A): if the sequence is not a member, the
   router's answer stands.

Every step is recorded in a decision trace, so an answer can be checked.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict

from prokopton.ml.config import AppConfig
from prokopton.ml.features.fragments import reverse_complement
from prokopton.ml.inference.inputs import InputError, Submission, parse_submission
from prokopton.ml.specialists.identify import Identification, Specialist

__all__ = [
    "Answer",
    "Classifier",
    "Identifier",
    "Region",
    "Result",
    "Step",
]

PLANNED_MARKERS = (
    "Support for other markers (plant rbcL, fungal ITS and bacterial 16S) is planned."
)


class Classifier(Protocol):
    """What the identifier needs from a classifier (scikit-learn style)."""

    @property
    def classes_(self) -> np.ndarray: ...

    def predict_proba(self, inputs: Any) -> np.ndarray: ...


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Step(_Model):
    """One step of the decision trace, and how long it took."""

    name: str
    summary: str
    details: dict[str, Any] = {}
    seconds: float = 0.0


class _Trace:
    """The steps so far, each timed from the end of the one before."""

    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.started = self._last = time.perf_counter()

    def add(self, step: Step) -> None:
        now = time.perf_counter()
        self.steps.append(
            step.model_copy(update={"seconds": round(now - self._last, 4)})
        )
        self._last = now

    @property
    def seconds(self) -> float:
        return round(self._last - self.started, 4)


class Region(_Model):
    """Where the barcode gene was found in a long sequence (1-based, inclusive,
    on the strand the sequence was given as)."""

    start: int
    end: int
    strand: Literal["given", "reverse"]
    probability: float


class Answer(_Model):
    """The answer to show."""

    level: str
    """species, complex, species_group, genus, family, or the router's
    group (such as order)."""
    name: str | None
    text: str
    confident: bool
    species: tuple[str, ...] = ()


class Result(_Model):
    """The outcome of identifying one submission."""

    status: Literal["identified", "declined"]
    answer: Answer | None = None
    reason: str | None = None
    """For a declined submission: a short code, such as ``too_short``."""
    message: str | None = None
    length: int = 0
    strand: Literal["given", "reverse"] | None = None
    region: Region | None = None
    notes: tuple[str, ...] = ()
    trace: tuple[Step, ...] = ()
    seconds: float = 0.0
    """Time spent identifying (not loading the models)."""


def _best(
    model: Classifier, sequences: Sequence[str], skip: str | None
) -> list[tuple[str, float]]:
    """The most likely class (other than ``skip``) and its probability."""
    probabilities = model.predict_proba(list(sequences))
    classes = [str(c) for c in model.classes_]
    keep = [i for i, c in enumerate(classes) if c != skip]
    found = []
    for row in probabilities:
        best = max(keep, key=lambda i: row[i])
        found.append((classes[best], float(row[best])))
    return found


class Identifier:
    """Identifies submissions with an app's marker classifier, router and
    specialists (already loaded)."""

    def __init__(
        self,
        config: AppConfig,
        marker: Classifier,
        router: Classifier,
        specialists: Mapping[str, Specialist],
    ) -> None:
        self.config = config
        self.rules = config.inference
        self.marker = marker
        self.router = router
        self.specialists = dict(specialists)
        marker_config = config.marker or config.router
        self.reverse_label = marker_config.training.reverse_label

    def identify(self, text: str) -> Result:
        """Identify one submission (see the module docstring)."""
        trace = _Trace()
        try:
            submission = parse_submission(
                text, self.rules, self.config.router.serving.min_length
            )
        except InputError as error:
            trace.add(Step(name="input", summary=error.message))
            return Result(
                status="declined",
                reason=error.code,
                message=error.message,
                trace=tuple(trace.steps),
                seconds=trace.seconds,
            )
        trace.add(
            Step(
                name="input",
                summary=f"One sequence of {submission.length:,} bases.",
                details={"name": submission.name, "notes": list(submission.notes)},
            )
        )
        return self._identify(submission, trace)

    def _marker(
        self, submission: Submission, trace: _Trace
    ) -> tuple[str, Literal["given", "reverse"], Region | None] | Result:
        """Find the barcode and its strand, or decline."""
        rules = self.rules
        sequence = submission.sequence
        if submission.length > rules.long_input:
            region = self._find_region(sequence)
            if region is None:
                msg = (
                    f"No {rules.barcode_gene} region was found in this "
                    f"{submission.length:,}-base sequence. {PLANNED_MARKERS}"
                )
                trace.add(Step(name="barcode", summary=msg))
                return self._decline("no_barcode", msg, submission, trace)
            given = sequence[region.start - 1 : region.end]
            oriented = given if region.strand == "given" else reverse_complement(given)
            trace.add(
                Step(
                    name="barcode",
                    summary=(
                        f"Found {rules.barcode_gene} at bases {region.start:,} to "
                        f"{region.end:,} ({region.strand} strand), with "
                        f"probability {region.probability:.2f}."
                    ),
                    details=region.model_dump(),
                )
            )
            return oriented, region.strand, region
        (gene, p_given), (reverse_gene, p_reverse) = _best(
            self.marker, [sequence, reverse_complement(sequence)], self.reverse_label
        )
        strand: Literal["given", "reverse"] = (
            "given" if p_given >= p_reverse else "reverse"
        )
        gene, probability = (
            (gene, p_given) if strand == "given" else (reverse_gene, p_reverse)
        )
        details = {"gene": gene, "probability": probability, "strand": strand}
        if gene != rules.barcode_gene:
            msg = (
                f"This looks like the {gene} gene (probability {probability:.2f}), "
                f"not {rules.barcode_gene}, which is the only marker supported "
                f"for now. {PLANNED_MARKERS}"
            )
            trace.add(Step(name="marker", summary=msg, details=details))
            return self._decline("unsupported_marker", msg, submission, trace)
        trace.add(
            Step(
                name="marker",
                summary=(
                    f"{gene} (probability {probability:.2f}), read on the "
                    f"{strand} strand."
                ),
                details=details,
            )
        )
        oriented = sequence if strand == "given" else reverse_complement(sequence)
        return oriented, strand, None

    def _find_region(self, sequence: str) -> Region | None:
        rules = self.rules
        starts = list(range(0, max(len(sequence) - rules.window, 0) + 1, rules.step))
        best: Region | None = None
        for strand in ("given", "reverse"):
            read = sequence if strand == "given" else reverse_complement(sequence)
            windows = [read[s : s + rules.window] for s in starts]
            probabilities = self.marker.predict_proba(windows)
            column = [str(c) for c in self.marker.classes_].index(rules.barcode_gene)
            run: list[int] = []
            runs: list[list[int]] = []
            for number, row in enumerate(probabilities):
                if row[column] >= rules.region_probability:
                    run.append(number)
                elif run:
                    runs.append(run)
                    run = []
            if run:
                runs.append(run)
            for found in runs:
                start = starts[found[0]]
                end = min(starts[found[-1]] + rules.window, len(read))
                probability = float(np.mean([probabilities[n][column] for n in found]))
                if strand == "reverse":
                    # Back to coordinates on the strand as given
                    start, end = len(read) - end, len(read) - start
                region = Region(
                    start=start + 1, end=end, strand=strand, probability=probability
                )
                if best is None or region.end - region.start > best.end - best.start:
                    best = region
        return best

    def _decline(
        self, reason: str, message: str, submission: Submission, trace: _Trace
    ) -> Result:
        return Result(
            status="declined",
            reason=reason,
            message=message,
            length=submission.length,
            notes=submission.notes,
            trace=tuple(trace.steps),
            seconds=trace.seconds,
        )

    def _identify(self, submission: Submission, trace: _Trace) -> Result:
        found = self._marker(submission, trace)
        if isinstance(found, Result):
            return found
        oriented, strand, region = found
        notes = list(submission.notes)
        serving = self.config.router.serving
        if len(oriented) < serving.short_length:
            notes.append(
                f"The sequence is shorter than {serving.short_length} bases, so "
                "answers are less accurate."
            )
        probabilities = self.router.predict_proba([oriented])[0]
        classes = [str(c) for c in self.router.classes_]
        order = np.argsort(probabilities)[::-1]
        top = [(classes[i], float(probabilities[i])) for i in order[:3]]
        label, confidence = top[0]
        confident = confidence >= serving.confident_threshold
        trace.add(
            Step(
                name="router",
                summary=f"{label} (probability {confidence:.2f}).",
                details={"top": [{"label": c, "probability": p} for c, p in top]},
            )
        )
        answer = Answer(
            level="group",
            name=label,
            text=(
                f"Most likely {label} (probability {confidence:.2f})."
                if confident
                else f"Possibly {label}, but the model is unsure (probability "
                f"{confidence:.2f})."
            ),
            confident=confident,
        )
        for name, specialist in self.specialists.items():
            rules = self.config.specialist(name)
            if not rules.router_label or rules.router_label not in classes:
                continue
            share = float(probabilities[classes.index(rules.router_label)])
            if share < self.rules.route_threshold:
                continue
            identification = specialist.identify(oriented)
            trace.add(_specialist_step(name, share, identification))
            if identification.outcome != "not_member":
                answer = _specialist_answer(identification)
                notes.extend(identification.notes)
                break
        return Result(
            status="identified",
            answer=answer,
            length=submission.length,
            strand=strand,
            region=region,
            notes=tuple(notes),
            trace=tuple(trace.steps),
            seconds=trace.seconds,
        )


def _specialist_step(name: str, share: float, found: Identification) -> Step:
    return Step(
        name=f"specialist:{name}",
        summary=(
            f"Consulted (router gave its group {share:.2f}): {found.outcome}"
            + (f", {found.label}" if found.label else "")
            + f", best identity {found.identity:.1%}."
        ),
        details={
            "outcome": found.outcome,
            "label": found.label,
            "identity": found.identity,
            "species": list(found.species),
            "matches": [
                {
                    "record_id": m.reference.record_id,
                    "species": m.reference.species,
                    "identity": m.identity,
                    "coverage": m.coverage,
                    "aligned": m.aligned,
                }
                for m in found.matches
            ],
        },
    )


_LEVELS = {
    "species": "species",
    "closest_species": "species",
    "complex": "complex",
    "species_group": "species_group",
    "genus_label_only": "genus",
    "genus": "genus",
    "family": "family",
}


def _specialist_answer(found: Identification) -> Answer:
    names = ", ".join(found.species)
    texts = {
        "species": f"{found.label} ({found.identity:.1%} identical).",
        "closest_species": (
            f"Closest match: {found.label} ({found.identity:.1%} identical); "
            "possibly a related species."
        ),
        "complex": (
            f"{found.label} (closest: {names}). This marker cannot separate "
            "its members."
        ),
        "species_group": f"One of: {names}.",
        "genus_label_only": (
            f"{found.label or 'A mosquito'}: closest records have no species name."
        ),
        "genus": f"{found.label} ({found.identity:.1%} identical).",
        "family": f"{found.label} ({found.identity:.1%} identical).",
    }
    return Answer(
        level=_LEVELS[found.outcome],
        name=found.label,
        text=texts[found.outcome],
        confident=found.outcome in ("species", "complex", "species_group"),
        species=found.species,
    )
