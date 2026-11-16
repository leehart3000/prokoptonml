"""Serving step kinds for DNA sequences.

1. **Check the input** (``bio.sequence-input``, see ``inputs``): one DNA
   sequence, cleaned.
2. **Find the barcode** (``bio.marker``). The marker classifier scores the
   sequence on both strands and keeps the strand whose best gene (other than
   its "reverse" class) is most confident. If that gene is not the barcode
   gene, the sequence is declined, saying which gene it looks like. A long
   sequence (such as a whole mitochondrial genome) is first searched for the
   barcode gene with a sliding window, and only that region is used.
3. **Route** (``bio.router``). The router predicts the broad group (for
   animals, the order).
4. **Specialists** (``bio.specialist``). A specialist is consulted when the
   router gives its group at least ``route_threshold`` (Rule B). The
   specialist checks membership itself (Rule A): if the sequence is not a
   member, the router's answer stands; if it is, its answer is final.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from prokopton.ml.bio.config import (
    MarkerStepConfig,
    RouterStepConfig,
    SequenceInputStepConfig,
    SpecialistConfig,
    SpecialistStepConfig,
    classifier_settings,
    specialist_settings,
)
from prokopton.ml.bio.features.fragments import reverse_complement
from prokopton.ml.bio.inference.inputs import InputError, Submission, parse_submission
from prokopton.ml.bio.specialists.identify import Identification, Specialist
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.chain import (
    FINAL,
    Answer,
    Decline,
    RequestSpec,
    Result,
    ServingStepKind,
    Step,
    StepRunner,
    Work,
    _Model,
)

__all__ = [
    "MAX_SEQUENCE_CHARACTERS",
    "Classifier",
    "Region",
    "SequenceAnswer",
    "SequenceRequest",
    "SequenceResult",
    "bio_serving_kinds",
    "member_name",
    "specialist_answer",
]


class Classifier(Protocol):
    """What the steps need from a classifier (scikit-learn style)."""

    @property
    def classes_(self) -> np.ndarray: ...

    def predict_proba(self, inputs: Any) -> np.ndarray: ...


class Region(_Model):
    """Where the barcode gene was found in a long sequence (1-based, inclusive,
    on the strand the sequence was given as)."""

    start: int
    end: int
    strand: Literal["given", "reverse"]
    probability: float


MAX_SEQUENCE_CHARACTERS = 30_000
"""Longer request bodies are refused before any work (the input step's own
limit is in bases, after cleaning)."""


class SequenceRequest(BaseModel):
    """One sequence to identify (plain, or one FASTA record)."""

    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(min_length=1, max_length=MAX_SEQUENCE_CHARACTERS)


def _request_model(config: AppConfig, name: str) -> type[BaseModel]:
    rules = _settings(config, name, SequenceInputStepConfig)
    examples: list[Any] = [{"sequence": rules.example}] if rules.example else []
    title = config.serving.endpoint.replace("-", " ").title().replace(" ", "")
    return type(
        f"{title}Request",
        (SequenceRequest,),
        {
            "__doc__": SequenceRequest.__doc__,
            "model_config": ConfigDict(
                extra="forbid", json_schema_extra={"examples": examples}
            ),
        },
    )


def _request_text(request: BaseModel) -> str:
    assert isinstance(request, SequenceRequest)  # Made by _request_model
    return request.sequence


def _log_input(result: Result) -> dict[str, Any]:
    return {
        "length": getattr(result, "length", None),
        "strand": getattr(result, "strand", None),
        "long_input": getattr(result, "region", None) is not None,
    }


def _log_router(result: Result) -> dict[str, Any]:
    router = next((s for s in result.trace if s.name == "router"), None)
    top = router.details["top"][0] if router else {}
    return {
        "router_label": top.get("label"),
        "router_probability": top.get("probability"),
    }


def _log_specialist(result: Result) -> dict[str, Any]:
    found = next((s for s in result.trace if s.name.startswith("specialist:")), None)
    return {
        "specialist_outcome": found.details["outcome"] if found else None,
        "identity": found.details["identity"] if found else None,
    }


class SequenceAnswer(Answer):
    """An answer about a sequence. ``level`` is species, complex,
    species_group, genus, family or group (the router's, such as an order);
    ``species`` names the species behind a species, complex or closest-match
    answer."""

    species: tuple[str, ...] = ()


class SequenceResult(Result):
    """The outcome of identifying one sequence."""

    answer: SequenceAnswer | None = None
    length: int = 0
    strand: Literal["given", "reverse"] | None = None
    region: Region | None = None


def _settings[T](config: AppConfig, name: str, kind: type[T]) -> T:
    settings = config.step_settings(name)
    assert isinstance(settings, kind)  # The step's kind chose its settings type
    return settings


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


def _submission(work: Work) -> Submission:
    found = work.found["submission"]
    assert isinstance(found, Submission)  # Set by the input step
    return found


def _build_input(config: AppConfig, name: str, _: Mapping[str, Any]) -> StepRunner:
    rules = _settings(config, name, SequenceInputStepConfig)
    min_length = classifier_settings(config, rules.model).serving.min_length

    def run(work: Work) -> Decline | None:
        try:
            submission = parse_submission(work.text, rules, min_length)
        except InputError as error:
            work.trace.add(Step(name="input", summary=error.message))
            return Decline(error.code, error.message)
        work.trace.add(
            Step(
                name="input",
                summary=f"One sequence of {submission.length:,} bases.",
                details={"name": submission.name, "notes": list(submission.notes)},
            )
        )
        work.found["submission"] = submission
        work.found["sequence"] = submission.sequence
        work.fields["length"] = submission.length
        work.notes.extend(submission.notes)
        return None

    return run


def _find_region(
    marker: Classifier, rules: MarkerStepConfig, sequence: str
) -> Region | None:
    starts = list(range(0, max(len(sequence) - rules.window, 0) + 1, rules.step))
    best: Region | None = None
    for strand in ("given", "reverse"):
        read = sequence if strand == "given" else reverse_complement(sequence)
        windows = [read[s : s + rules.window] for s in starts]
        probabilities = marker.predict_proba(windows)
        column = [str(c) for c in marker.classes_].index(rules.barcode_gene)
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


def _build_marker(
    config: AppConfig, name: str, models: Mapping[str, Any]
) -> StepRunner:
    rules = _settings(config, name, MarkerStepConfig)
    marker: Classifier = models[rules.model]
    reverse_label = classifier_settings(config, rules.model).training.reverse_label

    def with_note(message: str) -> str:
        note = rules.unsupported_note
        return f"{message} {note}" if note else message

    def run(work: Work) -> Decline | None:
        submission = _submission(work)
        sequence = submission.sequence
        if submission.length > rules.long_input:
            region = _find_region(marker, rules, sequence)
            if region is None:
                msg = with_note(
                    f"No {rules.barcode_gene} region was found in this "
                    f"{submission.length:,}-base sequence."
                )
                work.trace.add(Step(name="barcode", summary=msg))
                return Decline("no_barcode", msg)
            given = sequence[region.start - 1 : region.end]
            oriented = given if region.strand == "given" else reverse_complement(given)
            work.trace.add(
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
            work.found["sequence"] = oriented
            work.fields["strand"] = region.strand
            work.fields["region"] = region
            return None
        (gene, p_given), (reverse_gene, p_reverse) = _best(
            marker, [sequence, reverse_complement(sequence)], reverse_label
        )
        strand: Literal["given", "reverse"] = (
            "given" if p_given >= p_reverse else "reverse"
        )
        gene, probability = (
            (gene, p_given) if strand == "given" else (reverse_gene, p_reverse)
        )
        details = {"gene": gene, "probability": probability, "strand": strand}
        if gene != rules.barcode_gene:
            msg = with_note(
                f"This looks like the {gene} gene (probability {probability:.2f}), "
                f"not {rules.barcode_gene}, which is the only marker supported "
                "for now."
            )
            work.trace.add(Step(name="marker", summary=msg, details=details))
            return Decline("unsupported_marker", msg)
        work.trace.add(
            Step(
                name="marker",
                summary=(
                    f"{gene} (probability {probability:.2f}), read on the "
                    f"{strand} strand."
                ),
                details=details,
            )
        )
        work.found["sequence"] = (
            sequence if strand == "given" else reverse_complement(sequence)
        )
        work.fields["strand"] = strand
        work.fields["region"] = None
        return None

    return run


def _build_router(
    config: AppConfig, name: str, models: Mapping[str, Any]
) -> StepRunner:
    rules = _settings(config, name, RouterStepConfig)
    router: Classifier = models[rules.model]
    serving = classifier_settings(config, rules.model).serving

    def run(work: Work) -> None:
        oriented = work.found["sequence"]
        if len(oriented) < serving.short_length:
            work.notes.append(
                f"The sequence is shorter than {serving.short_length} bases, so "
                "answers are less accurate."
            )
        probabilities = router.predict_proba([oriented])[0]
        classes = [str(c) for c in router.classes_]
        order = np.argsort(probabilities)[::-1]
        top = [(classes[i], float(probabilities[i])) for i in order[:3]]
        label, confidence = top[0]
        confident = confidence >= serving.confident_threshold
        work.trace.add(
            Step(
                name="router",
                summary=f"{label} (probability {confidence:.2f}).",
                details={"top": [{"label": c, "probability": p} for c, p in top]},
            )
        )
        work.answer = SequenceAnswer(
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
        work.found["groups"] = {
            c: float(p) for c, p in zip(classes, probabilities, strict=True)
        }

    return run


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


def member_name(rules: SpecialistConfig) -> str:
    """How answers name a member of a specialist's group with no species or
    genus name."""
    return rules.member_name or f"A member of {rules.target}"


def specialist_answer(found: Identification, member: str) -> SequenceAnswer:
    """The answer for a specialist's identification of a member."""
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
            f"{found.label or member}: closest records have no species name."
        ),
        "genus": f"{found.label} ({found.identity:.1%} identical).",
        "family": f"{found.label} ({found.identity:.1%} identical).",
    }
    return SequenceAnswer(
        level=_LEVELS[found.outcome],
        name=found.label,
        text=texts[found.outcome],
        confident=found.outcome in ("species", "complex", "species_group"),
        species=found.species,
    )


def _build_specialist(
    config: AppConfig, name: str, models: Mapping[str, Any]
) -> StepRunner:
    rules = _settings(config, name, SpecialistStepConfig)
    specialist: Specialist = models[rules.model]
    group = specialist_settings(config, rules.model)
    member = member_name(group)

    def run(work: Work) -> Literal["final"] | None:
        groups: dict[str, float] = work.found.get("groups", {})
        if not group.router_label or group.router_label not in groups:
            return None
        share = groups[group.router_label]
        if share < rules.route_threshold:
            return None
        identification = specialist.identify(work.found["sequence"])
        work.trace.add(_specialist_step(rules.model, share, identification))
        if identification.outcome == "not_member":
            return None
        work.answer = specialist_answer(identification, member)
        work.notes.extend(identification.notes)
        return FINAL

    return run


def bio_serving_kinds() -> list[ServingStepKind]:
    """The serving step kinds for DNA sequences."""
    return [
        ServingStepKind(
            "bio.sequence-input",
            "Check a submission: one DNA sequence, within the length limits.",
            _build_input,
            result=SequenceResult,
            request=RequestSpec(_request_model, _request_text),
            log=_log_input,
        ),
        ServingStepKind(
            "bio.marker",
            "Find the barcode gene and its strand, or decline another marker.",
            _build_marker,
        ),
        ServingStepKind(
            "bio.router",
            "Predict the broad group: the answer unless a specialist gives one.",
            _build_router,
            log=_log_router,
        ),
        ServingStepKind(
            "bio.specialist",
            "Identify within a group, when the router gives it enough weight.",
            _build_specialist,
            log=_log_specialist,
        ),
    ]
