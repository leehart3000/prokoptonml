"""Identify a sequence within a specialist's group, explaining the answer.

A query is compared with every reference sequence by k-mer similarity (fast,
using the same features as the classifiers); the ``candidates`` most similar
(at most ``candidates_per_species`` from any one species) are aligned to it,
and their percent identity decides the outcome.

Every answer describes our reference data and this marker, never the
organism's scientific status: a species missing from our data, or one this
marker cannot separate from its relatives, may still be a well-known
species. The outcomes are:

- **not_member**: the closest reference is outside the group (background),
  or nothing in the group is at least ``member_identity`` identical. The
  site then keeps the router's answer.
- **species**: one named species in our reference data is at least
  ``species_identity`` identical, no other named species is within
  ``group_margin`` of it, and the query is closer to it than any other
  named species in our data is (its barcode gap).
- **closest_species**: as for a species, but other species in our data are
  as close to that species as this sequence is, so it may be a related
  species missing from our data.
- **complex**: the best named species belong to a configured complex of
  species that this marker cannot tell apart; all are named, with how to
  resolve them. A single member is still reported as the complex.
- **species_group**: several named species match equally well (within
  ``group_margin``), and they are not a known complex.
- **genus_label_only**: the closest records in our data are labelled only
  to genus (such as "Aedes sp. BIOUG..."), and clearly closer than any
  named species. The organism may well belong to a named species.
- **genus**: the closest records share a genus and are at least
  ``genus_identity`` identical.
- **family**: within the group, but not close enough for a genus.

Every outcome comes with the aligned matches, so the decision can be shown
and checked.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from prokopton.ml.config import ComplexConfig, SpecialistConfig
from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.specialists.identity import align

__all__ = [
    "NEIGHBOURS_NAME",
    "Identification",
    "Match",
    "Neighbour",
    "Outcome",
    "Reference",
    "Specialist",
    "read_neighbours",
    "write_neighbours",
]

NEIGHBOURS_NAME = "neighbours.json"
QUERY_BATCH = 256
"""Queries compared with the reference by k-mers at a time (bounds memory)."""
POOL_FACTOR = 10
"""How many k-mer candidates to consider for each one aligned."""

Outcome = Literal[
    "species",
    "closest_species",
    "complex",
    "species_group",
    "genus_label_only",
    "genus",
    "family",
    "not_member",
]


@dataclass(frozen=True, slots=True)
class Reference:
    """One reference sequence and what is known about it."""

    record_id: str
    role: str
    family: str | None
    genus: str | None
    species: str | None
    species_named: bool
    sequence: str


@dataclass(frozen=True, slots=True)
class Match:
    """A reference aligned to the query."""

    reference: Reference
    identity: float
    coverage: float
    aligned: int


@dataclass(frozen=True, slots=True)
class Neighbour:
    """The nearest other named species to a species, in our reference data."""

    species: str
    nearest: str
    identity: float


@dataclass(frozen=True, slots=True)
class Identification:
    """A specialist's answer, with the evidence for it."""

    outcome: Outcome
    label: str | None
    identity: float
    species: tuple[str, ...] = ()
    matches: tuple[Match, ...] = ()
    notes: tuple[str, ...] = field(default=())


def _key(reference: Reference) -> str:
    return reference.species or reference.record_id


class Specialist:
    """Matches queries against a reference set (see the module docstring)."""

    def __init__(
        self,
        references: Sequence[Reference],
        rules: SpecialistConfig,
        neighbours: Mapping[str, Neighbour] | None = None,
    ) -> None:
        if not references:
            msg = "A specialist needs at least one reference sequence"
            raise ValueError(msg)
        self.references = list(references)
        self.rules = rules
        self.neighbours = dict(neighbours or {})
        self.vectorizer = KmerVectorizer(k=rules.k).fit()
        self.index: sparse.csr_matrix = normalize(
            self.vectorizer.transform([r.sequence for r in self.references])
        ).T.tocsr()
        self._keys = [_key(r) for r in self.references]

    @classmethod
    def from_parquet(cls, path: Path, rules: SpecialistConfig) -> "Specialist":
        """Load a reference set written by the reference stage.

        The barcode-gap table (``neighbours.json``) is loaded too, if it is
        next to the reference set.
        """
        rows = pq.read_table(path).to_pylist()
        neighbours_path = path.parent / NEIGHBOURS_NAME
        neighbours = (
            read_neighbours(neighbours_path) if neighbours_path.exists() else None
        )
        return cls(
            [
                Reference(
                    record_id=row["record_id"],
                    role=row["role"],
                    family=row["family"],
                    genus=row["genus"],
                    species=row["species"],
                    species_named=bool(row["species_named"]),
                    sequence=row["sequence"],
                )
                for row in rows
            ],
            rules,
            neighbours,
        )

    def candidates(
        self,
        sequences: Sequence[str],
        *,
        count: int | None = None,
        per_species: int | None = None,
        eligible: np.ndarray | None = None,
        exclude: Sequence[str | None] | None = None,
    ) -> list[list[int]]:
        """Return, for each query, the reference numbers most similar by k-mers.

        Args:
            sequences: The queries.
            count: How many to return per query (default ``candidates``).
            per_species: The most from any one species (default
                ``candidates_per_species``).
            eligible: If given, only references marked True are returned.
            exclude: If given, a species to leave out for each query.
        """
        count = count or self.rules.candidates
        per_species = per_species or self.rules.candidates_per_species
        pool = min(count * POOL_FACTOR, len(self.references))
        found: list[list[int]] = []
        for start in range(0, len(sequences), QUERY_BATCH):
            batch = list(sequences[start : start + QUERY_BATCH])
            queries = normalize(self.vectorizer.transform(batch))
            similarity = np.asarray((queries @ self.index).todense())
            if eligible is not None:
                similarity[:, ~eligible] = -1.0
            best = np.argpartition(-similarity, pool - 1, axis=1)[:, :pool]
            order = np.argsort(-np.take_along_axis(similarity, best, axis=1), axis=1)
            ranked = np.take_along_axis(best, order, axis=1)
            for row, numbers in enumerate(ranked):
                skip = exclude[start + row] if exclude is not None else None
                taken: Counter[str] = Counter()
                chosen: list[int] = []
                for number in numbers.tolist():
                    if similarity[row, number] < 0:
                        break
                    key = self._keys[number]
                    if key == skip or taken[key] >= per_species:
                        continue
                    taken[key] += 1
                    chosen.append(number)
                    if len(chosen) == count:
                        break
                found.append(chosen)
        return found

    def _align(self, sequence: str, numbers: Sequence[int]) -> list[Match]:
        matches = []
        for number in numbers:
            reference = self.references[number]
            alignment = align(sequence, reference.sequence)
            matches.append(
                Match(
                    reference=reference,
                    identity=alignment.identity,
                    coverage=alignment.coverage,
                    aligned=alignment.aligned,
                )
            )
        return matches

    def _usable(self, matches: Sequence[Match]) -> list[Match]:
        return sorted(
            (
                m
                for m in matches
                if m.coverage >= self.rules.min_coverage
                and m.aligned >= self.rules.min_aligned
            ),
            key=lambda m: -m.identity,
        )

    def identify(self, sequence: str) -> Identification:
        """Identify one sequence (already the right strand)."""
        return self.identify_many([sequence])[0]

    def identify_many(self, sequences: Sequence[str]) -> list[Identification]:
        """Identify several sequences; the k-mer step is done in batches."""
        return [
            self.decide(self._align(sequence, numbers))
            for sequence, numbers in zip(
                sequences, self.candidates(sequences), strict=True
            )
        ]

    def find_neighbours(self, candidates: int = 5) -> dict[str, Neighbour]:
        """Find each named target species' nearest other named species.

        Up to ``neighbour_references`` records of each species are compared
        with the closest references of ``candidates`` other named species.
        This is the species' barcode gap in our reference data.
        """
        eligible = np.array(
            [r.role == "target" and r.species_named for r in self.references]
        )
        chosen: list[int] = []
        taken: Counter[str] = Counter()
        for number in np.flatnonzero(eligible).tolist():
            key = self._keys[number]
            if taken[key] < self.rules.neighbour_references:
                taken[key] += 1
                chosen.append(number)
        sequences = [self.references[n].sequence for n in chosen]
        species = [self._keys[n] for n in chosen]
        found: dict[str, Neighbour] = {}
        for name, sequence, numbers in zip(
            species,
            sequences,
            self.candidates(
                sequences,
                count=candidates,
                per_species=1,
                eligible=eligible,
                exclude=species,
            ),
            strict=True,
        ):
            usable = self._usable(self._align(sequence, numbers))
            if not usable:
                continue
            best = usable[0]
            if name not in found or best.identity > found[name].identity:
                found[name] = Neighbour(
                    species=name,
                    nearest=_key(best.reference),
                    identity=best.identity,
                )
        return dict(sorted(found.items()))

    def _complex_of(self, species: str) -> ComplexConfig | None:
        return next((c for c in self.rules.complexes if species in c.members), None)

    def decide(self, matches: Sequence[Match]) -> Identification:
        """Turn aligned matches into an outcome (see the module docstring)."""
        rules = self.rules
        usable = self._usable(matches)
        if not usable:
            return Identification(
                outcome="not_member",
                label=None,
                identity=0.0,
                notes=(
                    "No sequence in our reference data overlaps enough of "
                    "this sequence to compare.",
                ),
            )
        best = usable[0]
        shown = tuple(usable[:5])
        if best.reference.role != "target":
            return Identification(
                outcome="not_member",
                label=best.reference.family,
                identity=best.identity,
                matches=shown,
                notes=(
                    f"The closest match in our reference data is outside "
                    f"{rules.target}: "
                    f"{best.reference.species or 'a record without a species name'}"
                    f" ({best.reference.family}), {best.identity:.1%} identical.",
                ),
            )
        if best.identity < rules.member_identity:
            return Identification(
                outcome="not_member",
                label=None,
                identity=best.identity,
                matches=shown,
                notes=(
                    f"Nothing in {rules.target} in our reference data is at "
                    f"least {rules.member_identity:.0%} identical.",
                ),
            )
        target = [m for m in usable if m.reference.role == "target"]
        named: dict[str, Match] = {}
        for m in target:
            if m.reference.species_named and m.reference.species:
                named.setdefault(m.reference.species, m)
        top = next(iter(named.values()), None)
        unnamed = next((m for m in target if not m.reference.species_named), None)
        if best.identity >= rules.species_identity:
            if (
                top is None
                or top.identity < rules.species_identity
                or best.identity - top.identity > rules.group_margin
            ):
                return self._genus_label_only(best, shown)
            return self._named(top, named, unnamed, shown)
        genera = sorted(
            {
                m.reference.genus
                for m in target
                if m.reference.genus
                and m.identity >= best.identity - rules.group_margin
            }
        )
        missing = (
            "This may be a species missing from our reference data, or one "
            "this marker does not separate well."
        )
        if best.identity >= rules.genus_identity and len(genera) == 1:
            return Identification(
                outcome="genus",
                label=genera[0],
                identity=best.identity,
                matches=shown,
                notes=(
                    f"The closest matches in our reference data are {genera[0]} "
                    f"({best.identity:.1%} identical). {missing}",
                ),
            )
        return Identification(
            outcome="family",
            label=rules.target,
            identity=best.identity,
            matches=shown,
            notes=(
                f"The closest matches in our reference data are in "
                f"{rules.target} ({best.identity:.1%} identical), but not "
                f"close enough to name a genus. {missing}",
            ),
        )

    def _genus_label_only(
        self, best: Match, shown: tuple[Match, ...]
    ) -> Identification:
        level = best.reference.genus or self.rules.target
        rank = "genus" if best.reference.genus else "family"
        return Identification(
            outcome="genus_label_only",
            label=best.reference.genus,
            identity=best.identity,
            matches=shown,
            notes=(
                f"The closest match is a reference record labelled only to "
                f"{rank} ({best.reference.species or level}, "
                f"{best.reference.record_id}), {best.identity:.1%} identical. "
                f"Our reference data has no species name for it; the organism "
                f"may still belong to a named {level} species.",
            ),
        )

    def _named(
        self,
        top: Match,
        named: Mapping[str, Match],
        unnamed: Match | None,
        shown: tuple[Match, ...],
    ) -> Identification:
        rules = self.rules
        floor = max(rules.species_identity, top.identity - rules.group_margin)
        close = sorted(name for name, m in named.items() if m.identity >= floor)
        notes = []
        if unnamed is not None and unnamed.identity > top.identity:
            notes.append(
                f"A reference record without a species name "
                f"({unnamed.reference.species}) matches slightly better "
                f"({unnamed.identity:.1%})."
            )
        group = self._complex_of(close[0])
        if group is not None and all(name in group.members for name in close):
            notes.append(
                f"Members of the {group.name} cannot be told apart by this "
                f"marker. {group.resolve}"
            )
            return Identification(
                outcome="complex",
                label=group.name,
                identity=top.identity,
                species=tuple(close),
                matches=shown,
                notes=tuple(notes),
            )
        if len(close) > 1:
            notes.append(
                "Several species in our reference data match equally well. "
                "This marker does not separate them here."
            )
            return Identification(
                outcome="species_group",
                label=None,
                identity=top.identity,
                species=tuple(close),
                matches=shown,
                notes=tuple(notes),
            )
        name = close[0]
        neighbour = self.neighbours.get(name)
        if neighbour is not None and neighbour.identity >= top.identity:
            notes.append(
                f"The closest match in our reference data is {name} "
                f"({top.identity:.1%} identical), but {neighbour.nearest} is "
                f"{neighbour.identity:.1%} identical to {name}, so this may be "
                f"a related species missing from our reference data."
            )
            return Identification(
                outcome="closest_species",
                label=name,
                identity=top.identity,
                species=(name,),
                matches=shown,
                notes=tuple(notes),
            )
        notes.insert(
            0,
            f"Best match in our reference data: {name}, {top.identity:.1%} identical.",
        )
        return Identification(
            outcome="species",
            label=name,
            identity=top.identity,
            species=(name,),
            matches=shown,
            notes=tuple(notes),
        )


def write_neighbours(path: Path, neighbours: Mapping[str, Neighbour]) -> None:
    """Save a barcode-gap table as JSON."""
    rows = [asdict(n) for n in neighbours.values()]
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


def read_neighbours(path: Path) -> dict[str, Neighbour]:
    """Load a barcode-gap table saved by ``write_neighbours``."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["species"]: Neighbour(**row) for row in rows}
