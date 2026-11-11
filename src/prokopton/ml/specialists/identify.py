"""Identify a sequence within a specialist's group, explaining the answer.

A query is compared with every reference sequence by k-mer similarity (fast,
using the same features as the classifiers); the ``candidates`` most similar
are aligned to it, and their percent identity decides the outcome:

- **not_member**: the closest reference is outside the group (background),
  or nothing in the group is at least ``member_identity`` identical. The
  site then keeps the router's answer.
- **species**: one named species is at least ``species_identity`` identical
  (and no other named species is within ``tie_margin`` of it).
- **complex**: the best named species belongs to a configured complex of
  species that the marker cannot tell apart; all are named, with how to
  resolve them.
- **species_group**: several named species tie, and they are not a known
  complex.
- **unidentified_specimen**: the closest records are near-identical but
  carry placeholder names (such as "Aedes sp. BIOUG..."), so only the genus
  is known; the match is reported as useful information.
- **genus**: the closest records share a genus and are at least
  ``genus_identity`` identical.
- **family**: within the group, but not close enough for a genus.

Every outcome comes with the aligned matches, so the decision can be shown
and checked.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.preprocessing import normalize

from prokopton.ml.config import ComplexConfig, SpecialistConfig
from prokopton.ml.features.kmers import KmerVectorizer
from prokopton.ml.specialists.identity import align

__all__ = ["Identification", "Match", "Outcome", "Reference", "Specialist"]

Outcome = Literal[
    "species",
    "complex",
    "species_group",
    "unidentified_specimen",
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
class Identification:
    """A specialist's answer, with the evidence for it."""

    outcome: Outcome
    label: str | None
    identity: float
    species: tuple[str, ...] = ()
    matches: tuple[Match, ...] = ()
    notes: tuple[str, ...] = field(default=())


class Specialist:
    """Matches queries against a reference set (see the module docstring)."""

    def __init__(
        self, references: Sequence[Reference], rules: SpecialistConfig
    ) -> None:
        if not references:
            msg = "A specialist needs at least one reference sequence"
            raise ValueError(msg)
        self.references = list(references)
        self.rules = rules
        self.vectorizer = KmerVectorizer(k=rules.k).fit()
        self.index: sparse.csr_matrix = normalize(
            self.vectorizer.transform([r.sequence for r in self.references])
        ).T.tocsr()

    @classmethod
    def from_parquet(cls, path: Path, rules: SpecialistConfig) -> "Specialist":
        """Load a reference set written by the reference stage."""
        rows = pq.read_table(path).to_pylist()
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
        )

    def candidates(self, sequences: Sequence[str]) -> np.ndarray:
        """Return, for each query, the reference numbers most similar by k-mers."""
        queries = normalize(self.vectorizer.transform(list(sequences)))
        similarity = np.asarray((queries @ self.index).todense())
        top = min(self.rules.candidates, similarity.shape[1])
        best = np.argpartition(-similarity, top - 1, axis=1)[:, :top]
        order = np.argsort(-np.take_along_axis(similarity, best, axis=1), axis=1)
        ranked: np.ndarray = np.take_along_axis(best, order, axis=1)
        return ranked

    def identify(self, sequence: str) -> Identification:
        """Identify one sequence (already the right strand)."""
        return self.identify_many([sequence])[0]

    def identify_many(self, sequences: Sequence[str]) -> list[Identification]:
        """Identify several sequences; the k-mer step is done for all at once."""
        found = []
        for sequence, numbers in zip(
            sequences, self.candidates(sequences), strict=True
        ):
            matches = []
            for number in numbers:
                reference = self.references[int(number)]
                alignment = align(sequence, reference.sequence)
                matches.append(
                    Match(
                        reference=reference,
                        identity=alignment.identity,
                        coverage=alignment.coverage,
                        aligned=alignment.aligned,
                    )
                )
            found.append(self.decide(matches))
        return found

    def _complex_of(self, species: str) -> ComplexConfig | None:
        return next((c for c in self.rules.complexes if species in c.members), None)

    def decide(self, matches: Sequence[Match]) -> Identification:
        """Turn aligned matches into an outcome (see the module docstring)."""
        rules = self.rules
        usable = sorted(
            (m for m in matches if m.coverage >= rules.min_coverage and m.aligned),
            key=lambda m: -m.identity,
        )
        if not usable:
            return Identification(
                outcome="not_member",
                label=None,
                identity=0.0,
                notes=("No reference sequence covers enough of this sequence.",),
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
                    f"The closest reference is outside {rules.target}: "
                    f"{best.reference.species or 'an unnamed specimen'} "
                    f"({best.reference.family}), {best.identity:.1%} identical.",
                ),
            )
        if best.identity < rules.member_identity:
            return Identification(
                outcome="not_member",
                label=None,
                identity=best.identity,
                matches=shown,
                notes=(
                    f"Nothing in {rules.target} is at least "
                    f"{rules.member_identity:.0%} identical.",
                ),
            )
        tied = [
            m
            for m in usable
            if m.reference.role == "target"
            and m.identity >= best.identity - rules.tie_margin
        ]
        named = sorted(
            {
                m.reference.species
                for m in tied
                if m.reference.species_named and m.reference.species
            }
        )
        unnamed = [m for m in tied if not m.reference.species_named]
        genera = sorted({m.reference.genus for m in tied if m.reference.genus})
        notes = []
        if unnamed and best.identity >= rules.species_identity:
            specimen = unnamed[0]
            notes.append(
                f"{specimen.identity:.1%} identical to an unidentified "
                f"{specimen.reference.genus or rules.target} specimen "
                f"({specimen.reference.species})."
            )
        if best.identity >= rules.species_identity and named:
            group = self._complex_of(named[0])
            if group is not None and all(name in group.members for name in named):
                notes.append(
                    f"Members of the {group.name} cannot be told apart by this "
                    f"marker. {group.resolve}"
                )
                return Identification(
                    outcome="complex",
                    label=group.name,
                    identity=best.identity,
                    species=tuple(named),
                    matches=shown,
                    notes=tuple(notes),
                )
            if len(named) == 1:
                return Identification(
                    outcome="species",
                    label=named[0],
                    identity=best.identity,
                    species=tuple(named),
                    matches=shown,
                    notes=tuple(notes),
                )
            notes.append(
                "Several species are equally close; this marker cannot tell "
                "them apart here."
            )
            return Identification(
                outcome="species_group",
                label=None,
                identity=best.identity,
                species=tuple(named),
                matches=shown,
                notes=tuple(notes),
            )
        if best.identity >= rules.species_identity:
            return Identification(
                outcome="unidentified_specimen",
                label=best.reference.genus,
                identity=best.identity,
                matches=shown,
                notes=tuple(notes),
            )
        if best.identity >= rules.genus_identity and len(genera) == 1:
            return Identification(
                outcome="genus",
                label=genera[0],
                identity=best.identity,
                matches=shown,
            )
        return Identification(
            outcome="family",
            label=rules.target,
            identity=best.identity,
            matches=shown,
        )
