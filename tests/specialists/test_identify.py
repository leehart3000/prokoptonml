"""Tests for prokopton.ml.specialists.identify."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from prokopton.ml.config import ComplexConfig, SpecialistConfig
from prokopton.ml.specialists.identify import (
    NEIGHBOURS_NAME,
    Neighbour,
    Reference,
    Specialist,
    canonical_genus,
    canonical_species,
    write_neighbours,
)

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
    record_id: str, sequence: str, species: str, genus: str | None, **fields: object
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
    assert found.notes[0].startswith("Best match in our reference data: Aedes")


def test_a_long_query_matches_barcode_length_references(
    specialist: Specialist,
) -> None:
    # A whole gene (here, the barcode with 450 extra bases each side)
    found = specialist.identify(random_sequence(450) + AEGYPTI + random_sequence(450))

    assert (found.outcome, found.label) == ("species", "Aedes aegypti")


def test_species_within_the_group_margin_are_a_group() -> None:
    query = random_sequence()
    references = [
        ref("a", query, "Aedes cinereus", "Aedes"),
        ref("b", mutate(query, 0.005), "Aedes rossicus", "Aedes"),
        REFERENCES[8],
    ]

    found = Specialist(references, RULES).identify(query)

    assert found.outcome == "species_group"
    assert found.species == ("Aedes cinereus", "Aedes rossicus")


def test_closest_species_when_inside_the_barcode_gap() -> None:
    # Two named species 1.2% apart; a query 1.5% from one of them is no
    # closer to it than its neighbour is
    base = random_sequence()
    references = [
        ref("a", base, "Anopheles epiroticus", "Anopheles"),
        ref("b", mutate(base, 0.012), "Anopheles dravidicus", "Anopheles"),
        REFERENCES[8],
    ]
    specialist = Specialist(references, RULES)
    specialist.neighbours = specialist.find_neighbours()

    assert specialist.neighbours["Anopheles epiroticus"].nearest == (
        "Anopheles dravidicus"
    )
    found = specialist.identify(mutate(base, 0.015))
    assert (found.outcome, found.label) == ("closest_species", "Anopheles epiroticus")
    assert "may be a related species missing" in found.notes[-1]
    assert specialist.identify(base).outcome == "species"


def test_candidates_are_capped_per_species() -> None:
    query = random_sequence()
    copies = [
        ref(f"c{i}", mutate(query, 0.001), "Culex delta", "Culex") for i in range(8)
    ]
    other = ref("o", mutate(query, 0.01), "Culex epsilon", "Culex")
    specialist = Specialist([*copies, other, REFERENCES[8]], RULES)

    numbers = specialist.candidates([query], count=4, per_species=2)[0]

    assert [specialist.references[n].species for n in numbers][:3] == [
        "Culex delta",
        "Culex delta",
        "Culex epsilon",
    ]


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


def test_genus_label_only(specialist: Specialist) -> None:
    found = specialist.identify(mutate(PLACEHOLDER, 0.005))

    assert (found.outcome, found.label) == ("genus_label_only", "Culex")
    assert "labelled only to genus (Culex sp. BIOUG001, r6)" in found.notes[0]
    assert "may still belong to a named Culex species" in found.notes[0]


def test_a_record_labelled_only_to_family() -> None:
    family_only = ref("r10", PLACEHOLDER, "Culicidae sp. X", None, named=False)
    found = Specialist([family_only, REFERENCES[8]], RULES).identify(PLACEHOLDER)

    assert (found.outcome, found.label) == ("genus_label_only", None)
    assert "labelled only to family (Culicidae sp. X, r10)" in found.notes[0]


def test_a_named_species_slightly_behind_a_genus_only_record_is_named() -> None:
    # The genus-only record is identical; the named species is 0.5% away,
    # within the group margin, so the species is named, with a note
    query = random_sequence()
    references = [
        ref("u1", query, "Culex sp. BIOUG002", "Culex", named=False),
        ref("n1", mutate(query, 0.005), "Culex gamma", "Culex"),
        REFERENCES[8],
    ]

    found = Specialist(references, RULES).identify(query)

    assert (found.outcome, found.label) == ("species", "Culex gamma")
    assert any("without a species name" in note for note in found.notes)


def test_genus(specialist: Specialist) -> None:
    # About 4% from each Aedes species: the genus, but not a species
    found = specialist.identify(AEDES)

    assert (found.outcome, found.label) == ("genus", "Aedes")
    assert "may be a species missing from our reference data" in found.notes[0]


def test_family_when_only_distantly_similar() -> None:
    distant = Specialist([REFERENCES[0], REFERENCES[3], REFERENCES[8]], RULES)

    found = distant.identify(mutate(AEGYPTI, 0.07))

    assert (found.outcome, found.label) == ("family", "Culicidae")


def test_not_a_member_when_closest_is_background(specialist: Specialist) -> None:
    found = specialist.identify(mutate(CHAOBORUS, 0.01))

    assert (found.outcome, found.label) == ("not_member", "Chaoboridae")
    assert "outside Culicidae" in found.notes[0]


def test_a_background_record_without_a_species_name() -> None:
    unnamed = ref("x", CHAOBORUS, "", "", role="background", family="Chaoboridae")
    found = Specialist([REFERENCES[0], unnamed], RULES).identify(CHAOBORUS)

    assert "a record without a species name" in found.notes[0]


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
    assert loaded.neighbours == {}

    gaps = {"Aedes aegypti": Neighbour("Aedes aegypti", "Aedes albopictus", 0.96)}
    write_neighbours(tmp_path / NEIGHBOURS_NAME, gaps)
    assert Specialist.from_parquet(path, RULES).neighbours == gaps


def test_neighbours_skip_species_without_a_usable_match() -> None:
    # Unrelated species: no alignment covers enough to count
    found = Specialist([REFERENCES[3], REFERENCES[6], REFERENCES[8]], RULES)

    assert found.find_neighbours() == {}


def test_queries_are_compared_in_batches(
    specialist: Specialist, monkeypatch: pytest.MonkeyPatch
) -> None:
    queries = [AEGYPTI, GAMBIAE, SHARED]
    whole = specialist.candidates(queries)
    monkeypatch.setattr("prokopton.ml.specialists.identify.QUERY_BATCH", 2)

    assert specialist.candidates(queries) == whole


GROUPED = RULES.model_copy(
    update={
        "genus_groups": {"Ochlerotatus": "Aedes"},
        "genus_groups_note": "Aedes in the broad sense.",
        "species_synonyms": {"Ochlerotatus berlandi": "Ochlerotatus brelandi"},
    }
)


def test_canonical_names() -> None:
    assert canonical_genus("Ochlerotatus", GROUPED) == "Aedes"
    assert canonical_genus("Culex", GROUPED) == "Culex"
    assert canonical_genus(None, GROUPED) is None
    assert canonical_species("Ochlerotatus berlandi", GROUPED) == "Aedes brelandi"
    assert canonical_species("Ochlerotatus sp. X1", GROUPED) == "Aedes sp. X1"
    assert canonical_species("Ochlerotatus", GROUPED) == "Ochlerotatus"
    assert canonical_species("", GROUPED) == ""


def test_grouped_genera_are_one_genus_with_a_note() -> None:
    # Two references 4% from the query, filed under Aedes and Ochlerotatus:
    # one genus once grouped, so the answer is a genus, not just the family
    query = random_sequence()
    references = [
        ref("a", mutate(query, 0.04), "Aedes alpha", "Aedes"),
        ref("o", mutate(query, 0.04), "Ochlerotatus omega", "Ochlerotatus"),
        REFERENCES[8],
    ]
    specialist = Specialist(references, GROUPED)

    found = specialist.identify(query)

    assert specialist.references[1].species == "Aedes omega"
    assert (found.outcome, found.label) == ("genus", "Aedes")
    assert found.notes[-1] == "Aedes in the broad sense."
    assert Specialist(references, RULES).identify(query).outcome == "family"


def test_grouped_species_and_genus_only_answers_carry_the_note() -> None:
    query = random_sequence()
    named = Specialist(
        [ref("n", query, "Ochlerotatus x", "Ochlerotatus"), REFERENCES[8]], GROUPED
    )
    unnamed = ref("u", query, "Ochlerotatus sp. Y", "Ochlerotatus", named=False)
    placeholder = Specialist([unnamed, REFERENCES[8]], GROUPED)

    assert named.identify(query).notes[-1] == "Aedes in the broad sense."
    found = placeholder.identify(query)
    assert (found.outcome, found.label) == ("genus_label_only", "Aedes")
    assert found.notes[-1] == "Aedes in the broad sense."
    assert Specialist([unnamed, REFERENCES[8]], RULES).identify(query).notes[1:] == ()


def test_a_saved_index_is_used_and_checked(tmp_path: Path) -> None:
    specialist = Specialist(REFERENCES, RULES)
    path = tmp_path / "references.parquet"
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in REFERENCES]), path)
    specialist.save_index(tmp_path / "index.npz")

    loaded = Specialist.from_parquet(path, RULES, index=tmp_path / "index.npz")

    assert (loaded.index != specialist.index).nnz == 0
    assert loaded.identify(AEGYPTI).label == "Aedes aegypti"
    with pytest.raises(ValueError, match="saved index has 9 references"):
        Specialist(REFERENCES[:2], RULES, index=specialist.index)


CURATED = RULES.model_copy(update={"curation_min_support": 5})


def curation_world() -> list[Reference]:
    aegypti, pipiens = random_sequence(), random_sequence()
    references = [
        ref(f"a{i}", mutate(aegypti, 0.002), "Aedes aegypti", "Aedes") for i in range(6)
    ]
    references += [
        ref(f"b{i}", mutate(pipiens, 0.002), "Culex pipiens", "Culex") for i in range(6)
    ]
    # Labelled Cx. pipiens, but an Ae. aegypti sequence
    references.append(ref("x", mutate(aegypti, 0.002), "Culex pipiens", "Culex"))
    return references


def test_finds_references_that_look_like_another_genus() -> None:
    [found] = Specialist(curation_world(), CURATED).contradicted()

    assert (found.record_id, found.species, found.looks_like) == (
        "x",
        "Culex pipiens",
        "Aedes aegypti",
    )
    assert 0.95 < found.similarity <= 1.0
    assert found.own_similarity < 0.6


def test_errors_within_a_genus_are_left_to_the_hand_list() -> None:
    # Labelled Ae. albopictus, but an Ae. aegypti sequence: the same genus
    references = [r for r in curation_world() if r.record_id != "x"]
    albopictus = random_sequence()
    references += [
        ref(f"c{i}", mutate(albopictus, 0.002), "Aedes albopictus", "Aedes")
        for i in range(6)
    ]
    references.append(
        ref("y", mutate(references[0].sequence, 0.002), "Aedes albopictus", "Aedes")
    )

    assert Specialist(references, CURATED).contradicted() == []


def test_species_with_few_references_and_unnamed_neighbours_are_not_judged() -> None:
    # Cx. pipiens has 7 references, so it needs more than 7 to be judged
    few = CURATED.model_copy(update={"curation_min_support": 7})
    assert Specialist(curation_world(), few).contradicted() == []

    # A record unlike its own species, whose closest references have no
    # species name, is kept
    base, other = random_sequence(), random_sequence()
    references = [ref("y", base, "Culex gamma", "Culex")]
    references += [
        ref(f"g{i}", mutate(other, 0.005), "Culex gamma", "Culex") for i in range(6)
    ]
    references += [
        ref(f"p{i}", mutate(base, 0.005), "Culex sp. P", "Culex", named=False)
        for i in range(5)
    ]
    assert Specialist(references, CURATED).contradicted() == []
