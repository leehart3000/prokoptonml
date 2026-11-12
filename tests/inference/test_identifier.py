"""Tests for prokopton.ml.inference.identify."""

from typing import Any

import pytest

from inference_world import (
    AEGYPTI,
    ALBOPICTUS,
    CYTB,
    MUSCA,
    FakeMarker,
    FakeRouter,
    random_sequence,
)
from prokopton.ml.config import AppConfig
from prokopton.ml.features.fragments import reverse_complement
from prokopton.ml.inference.identify import (
    PLANNED_MARKERS,
    Identifier,
    _specialist_answer,
)
from prokopton.ml.specialists.identify import (
    Identification,
    Outcome,
    Reference,
    Specialist,
)
from specialist_world import config_dict


def reference(
    record_id: str, sequence: str, species: str, role: str = "target"
) -> Reference:
    genus = species.split()[0]
    return Reference(
        record_id=record_id,
        role=role,
        family="Culicidae" if role == "target" else "Muscidae",
        genus=genus,
        species=species,
        species_named=True,
        sequence=sequence,
    )


def config(**specialist: Any) -> AppConfig:
    data = config_dict_with_marker()
    data["specialists"]["culicidae"].update(specialist)
    return AppConfig.model_validate(data)


def config_dict_with_marker() -> dict[str, Any]:
    from pathlib import Path

    data = config_dict(Path("/tmp/unused"))
    data["marker"] = {"label": "gene", "training": {"reverse_label": "reverse"}}
    data["specialists"]["culicidae"]["router_label"] = "Diptera"
    return data


def identifier(router: FakeRouter, app: AppConfig | None = None) -> Identifier:
    app = app or config()
    specialist = Specialist(
        [
            reference("a", AEGYPTI, "Aedes aegypti"),
            reference("b", ALBOPICTUS, "Aedes albopictus"),
            reference("m", MUSCA, "Musca domestica", role="background"),
        ],
        app.specialist("culicidae"),
    )
    return Identifier(app, FakeMarker(), router, {"culicidae": specialist})


DIPTERA = FakeRouter(Diptera=0.95, Lepidoptera=0.05)


def test_identifies_a_mosquito_with_a_trace() -> None:
    result = identifier(DIPTERA).identify(f">sample 1\n{AEGYPTI}")

    assert result.status == "identified"
    assert result.answer is not None
    assert (result.answer.level, result.answer.name) == ("species", "Aedes aegypti")
    assert result.answer.confident
    assert result.answer.text == "Aedes aegypti (100.0% identical)."
    assert result.strand == "given"
    assert [step.name for step in result.trace] == [
        "input",
        "marker",
        "router",
        "specialist:culicidae",
    ]
    assert result.trace[0].details["name"] == "sample 1"
    assert result.trace[3].details["matches"][0]["species"] == "Aedes aegypti"
    assert any("Best match in our reference data" in note for note in result.notes)


def test_reads_the_other_strand() -> None:
    result = identifier(DIPTERA).identify(reverse_complement(AEGYPTI))

    assert result.strand == "reverse"
    assert result.answer is not None
    assert result.answer.name == "Aedes aegypti"


def test_declines_bad_input_and_other_markers() -> None:
    short = identifier(DIPTERA).identify("ACGT" * 10)
    assert (short.status, short.reason) == ("declined", "too_short")
    assert [step.name for step in short.trace] == ["input"]

    other = identifier(DIPTERA).identify(CYTB)
    assert (other.status, other.reason) == ("declined", "unsupported_marker")
    assert other.message is not None
    assert "looks like the Cytb gene" in other.message
    assert PLANNED_MARKERS in other.message


def test_the_router_answer_stands_when_no_specialist_applies() -> None:
    unsure = FakeRouter(Lepidoptera=0.55, Diptera=0.05, Hemiptera=0.40)

    result = identifier(unsure).identify(AEGYPTI)

    assert result.answer is not None
    assert (result.answer.level, result.answer.name) == ("group", "Lepidoptera")
    assert result.answer.text.startswith(
        "Possibly Lepidoptera, but the model is unsure"
    )
    assert not result.answer.confident
    assert [step.name for step in result.trace][-1] == "router"


def test_the_router_answer_stands_when_the_specialist_says_not_a_member() -> None:
    result = identifier(DIPTERA).identify(MUSCA)

    assert result.answer is not None
    assert result.answer.text == "Most likely Diptera (probability 0.95)."
    assert result.trace[-1].details["outcome"] == "not_member"


def test_specialists_without_a_router_label_are_not_consulted() -> None:
    for app in (config(router_label=""), config(router_label="Coleoptera")):
        result = identifier(DIPTERA, app).identify(AEGYPTI)
        assert result.answer is not None
        assert result.answer.level == "group"


def test_short_sequences_get_a_note() -> None:
    result = identifier(DIPTERA).identify(AEGYPTI[:250])

    assert any("shorter than 300 bases" in note for note in result.notes)


def test_finds_the_barcode_in_a_long_sequence() -> None:
    left, right = random_sequence(1200), random_sequence(900)
    genome = left + CYTB + AEGYPTI + right

    result = identifier(DIPTERA).identify(genome)

    assert result.answer is not None
    assert result.answer.name == "Aedes aegypti"
    assert result.region is not None
    assert result.region.strand == "given"
    # The region covers the barcode, and at most a window either side
    barcode_start = len(left) + len(CYTB) + 1
    barcode_end = barcode_start + len(AEGYPTI) - 1
    assert barcode_start - 600 <= result.region.start <= barcode_start
    assert barcode_end <= result.region.end <= barcode_end + 600
    assert result.trace[1].name == "barcode"


def test_finds_a_barcode_given_on_the_other_strand() -> None:
    genome = reverse_complement(random_sequence(1500) + AEGYPTI + random_sequence(700))

    result = identifier(DIPTERA).identify(genome)

    assert result.region is not None
    assert result.region.strand == "reverse"
    assert result.strand == "reverse"
    assert result.answer is not None
    assert result.answer.name == "Aedes aegypti"
    found = genome[result.region.start - 1 : result.region.end]
    assert AEGYPTI[100:500] in reverse_complement(found)


def test_declines_a_long_sequence_without_a_barcode() -> None:
    result = identifier(DIPTERA).identify(random_sequence(3000))

    assert (result.status, result.reason) == ("declined", "no_barcode")
    assert result.message is not None
    assert result.message.startswith("No CO1 region was found in this 3,000-base")


@pytest.mark.parametrize(
    ("outcome", "label", "species", "text", "level", "confident"),
    [
        (
            "closest_species",
            "Aedes x",
            ("Aedes x",),
            "Closest match: Aedes x (99.1% identical); possibly a related species.",
            "species",
            False,
        ),
        (
            "complex",
            "An. gambiae complex",
            ("A", "B"),
            "An. gambiae complex: A, B.",
            "complex",
            True,
        ),
        ("species_group", None, ("A", "B"), "One of: A, B.", "species_group", True),
        (
            "genus_label_only",
            None,
            (),
            "A mosquito: closest records have no species name.",
            "genus",
            False,
        ),
        ("genus", "Aedes", (), "Aedes (99.1% identical).", "genus", False),
        ("family", "Culicidae", (), "Culicidae (99.1% identical).", "family", False),
    ],
)
def test_specialist_answers(
    outcome: Outcome,
    label: str | None,
    species: tuple[str, ...],
    text: str,
    level: str,
    confident: bool,
) -> None:
    found = Identification(
        outcome=outcome, label=label, identity=0.991, species=species
    )

    answer = _specialist_answer(found)

    assert (answer.text, answer.level, answer.confident) == (text, level, confident)


def test_finds_a_barcode_at_the_end_of_a_long_sequence() -> None:
    genome = random_sequence(1500) + AEGYPTI

    result = identifier(DIPTERA).identify(genome)

    assert result.region is not None
    assert result.region.end == len(genome)
