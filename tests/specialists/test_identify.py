"""Tests for prokopton.ml.specialists.identify."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prokopton.ml.config import ComplexConfig, SpecialistConfig
from prokopton.ml.specialists.identify import Reference, Specialist

rng = np.random.default_rng(7)


def random_sequence(length: int = 600) -> str:
    return "".join(rng.choice(list("ACGT"), length))


def mutate(sequence: str, share: float) -> str:
    bases = list(sequence)
    for position in rng.choice(len(bases), int(len(bases) * share), replace=False):
        bases[position] = str(rng.choice([b for b in "ACGT" if b != bases[position]]))
    return "".join(bases)


AEDES = random_sequence()
AEGYPTI = mutate(AEDES, 0.04)
ALBOPICTUS = mutate(AEDES, 0.04)
GAMBIAE = random_sequence()
PLACEHOLDER = random_sequence()
SHARED = random_sequence()
CHAOBORUS = random_sequence()


def ref(
    record_id: str, sequence: str, species: str, genus: str, **fields: object
) -> Reference:
    return Reference(
        record_id=record_id,
        role=str(fields.get("role", "target")),
        family=str(fields.get("family", "Culicidae")),
        genus=genus,
        species=species,
        species_named=bool(fields.get("named", True)),
        sequence=sequence,
    )


REFERENCES = [
    ref("r1", AEGYPTI, "Aedes aegypti", "Aedes"),
    ref("r2", mutate(AEGYPTI, 0.005), "Aedes aegypti", "Aedes"),
    ref("r3", ALBOPICTUS, "Aedes albopictus", "Aedes"),
    ref("r4", GAMBIAE, "Anopheles gambiae", "Anopheles"),
    ref("r5", GAMBIAE, "Anopheles coluzzii", "Anopheles"),
    ref("r6", PLACEHOLDER, "Culex sp. BIOUG001", "Culex", named=False),
    ref("r7", SHARED, "Culex alpha", "Culex"),
    ref("r8", SHARED, "Culex beta", "Culex"),
    ref(
        "r9",
        CHAOBORUS,
        "Chaoborus flavicans",
        "Chaoborus",
        role="background",
        family="Chaoboridae",
    ),
]
RULES = SpecialistConfig(
    target="Culicidae",
    background="Diptera",
    k=6,
    candidates=5,
    complexes=(
        ComplexConfig(
            name="Anopheles gambiae complex",
            members=("Anopheles gambiae", "Anopheles coluzzii", "Anopheles arabiensis"),
            resolve="Use an IGS or SINE200 PCR, or a multilocus panel.",
        ),
    ),
)


@pytest.fixture(scope="module")
def specialist() -> Specialist:
    return Specialist(REFERENCES, RULES)


def test_species(specialist: Specialist) -> None:
    found = specialist.identify(mutate(AEGYPTI, 0.005))

    assert (found.outcome, found.label) == ("species", "Aedes aegypti")
    assert found.identity > 0.98
    assert found.matches[0].reference.species == "Aedes aegypti"


def test_fragment_of_a_species(specialist: Specialist) -> None:
    found = specialist.identify(ALBOPICTUS[100:300])

    assert (found.outcome, found.label) == ("species", "Aedes albopictus")


def test_complex_names_every_member_and_how_to_resolve(specialist: Specialist) -> None:
    found = specialist.identify(GAMBIAE)

    assert (found.outcome, found.label) == ("complex", "Anopheles gambiae complex")
    assert found.species == ("Anopheles coluzzii", "Anopheles gambiae")
    assert any("IGS or SINE200" in note for note in found.notes)


def test_a_single_complex_member_is_still_reported_as_the_complex() -> None:
    lonely = Specialist([REFERENCES[3], REFERENCES[8]], RULES)

    found = lonely.identify(GAMBIAE)

    assert found.outcome == "complex"
    assert found.species == ("Anopheles gambiae",)


def test_species_group_when_named_species_tie(specialist: Specialist) -> None:
    found = specialist.identify(SHARED)

    assert found.outcome == "species_group"
    assert found.species == ("Culex alpha", "Culex beta")


def test_unidentified_specimen(specialist: Specialist) -> None:
    found = specialist.identify(mutate(PLACEHOLDER, 0.005))

    assert (found.outcome, found.label) == ("unidentified_specimen", "Culex")
    assert "unidentified Culex specimen (Culex sp. BIOUG001)" in found.notes[0]


def test_genus(specialist: Specialist) -> None:
    # About 4% from each Aedes species: the genus, but not a species
    found = specialist.identify(AEDES)

    assert (found.outcome, found.label) == ("genus", "Aedes")


def test_family_when_only_distantly_similar() -> None:
    distant = Specialist([REFERENCES[0], REFERENCES[3], REFERENCES[8]], RULES)

    found = distant.identify(mutate(AEGYPTI, 0.07))

    assert (found.outcome, found.label) == ("family", "Culicidae")


def test_not_a_member_when_closest_is_background(specialist: Specialist) -> None:
    found = specialist.identify(mutate(CHAOBORUS, 0.01))

    assert (found.outcome, found.label) == ("not_member", "Chaoboridae")
    assert "outside Culicidae" in found.notes[0]


def test_not_a_member_when_too_dissimilar(specialist: Specialist) -> None:
    found = specialist.identify(mutate(AEGYPTI, 0.15))

    assert found.outcome == "not_member"
    assert "at least 90% identical" in found.notes[0]


def test_not_a_member_when_nothing_covers_the_query(specialist: Specialist) -> None:
    found = specialist.identify(random_sequence(300))

    assert found.outcome == "not_member"
    assert found.identity == 0.0


def test_identify_many_matches_identify(specialist: Specialist) -> None:
    queries = [AEGYPTI, GAMBIAE]

    assert [f.outcome for f in specialist.identify_many(queries)] == [
        "species",
        "complex",
    ]


def test_needs_references() -> None:
    with pytest.raises(ValueError, match="at least one reference"):
        Specialist([], RULES)


def test_loads_a_reference_file(tmp_path: Path) -> None:
    path = tmp_path / "reference.parquet"
    rows = [
        {
            "record_id": r.record_id,
            "role": r.role,
            "order": "Diptera",
            "family": r.family,
            "genus": r.genus,
            "species": r.species,
            "species_named": r.species_named,
            "sequence": r.sequence,
        }
        for r in REFERENCES
    ]
    pq.write_table(pa.Table.from_pylist(rows), path)

    loaded = Specialist.from_parquet(path, RULES)

    assert len(loaded.references) == len(REFERENCES)
    assert loaded.identify(AEGYPTI).label == "Aedes aegypti"
