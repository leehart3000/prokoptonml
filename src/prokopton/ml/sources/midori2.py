"""Parse MIDORI2 reference database records in SINTAX format.

MIDORI2 (https://www.reference-midori.info/) is a curated reference database of
eukaryotic mitochondrial gene sequences built from GenBank, released under
CC BY 4.0. Its SINTAX headers look like::

    MH535936.1.<1.>890;tax=k:Eukaryota_2759,p:Arthropoda_6656,...,s:Name_123;

Each taxon is written as ``<name>_<NCBI taxid>``. When a rank is unknown,
MIDORI2 fills it with a placeholder that starts with the rank name, such as
``phylum_class_order_family_genus_Amoebozoa sp._1892891``; these are returned
as missing (``None``). Note that the ``k:`` rank is the superkingdom (for
example Eukaryota), not the kingdom.
"""

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from prokopton.ml.data.fasta import FastaRecord

__all__ = [
    "RANKS",
    "Midori2HeaderError",
    "Midori2Record",
    "Taxon",
    "parse_midori2_records",
    "parse_sintax_header",
]

RANKS: tuple[str, ...] = (
    "superkingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
)
"""Taxonomic ranks in MIDORI2 SINTAX headers, from broadest to narrowest."""

_PREFIX_TO_RANK = dict(zip("kpcofgs", RANKS, strict=True))
_TAX_FIELD_SPLIT = re.compile(r",(?=[kpcofgs]:)")
_TAXON_WITH_ID = re.compile(r"^(?P<name>.+)_(?P<taxid>\d+)$")
_RECORD_ID = re.compile(
    r"^(?P<accession>[^.]+\.\d+)\.(?P<start>[<>]?\d+)\.(?P<end>[<>]?\d+)$"
)


class Midori2HeaderError(ValueError):
    """Raised when a MIDORI2 SINTAX header cannot be parsed."""


@dataclass(frozen=True, slots=True)
class Taxon:
    """A named taxon with its NCBI taxonomy ID."""

    name: str
    taxid: int | None


@dataclass(frozen=True, slots=True)
class Midori2Record:
    """A parsed MIDORI2 record.

    Attributes:
        record_id: The full identifier, e.g. ``MH535936.1.<1.>890``.
        accession: The GenBank accession and version, e.g. ``MH535936.1``.
        start: Start position in the GenBank record; ``<`` marks a partial start.
        end: End position in the GenBank record; ``>`` marks a partial end.
        taxonomy: Taxon for each rank in ``RANKS``, or ``None`` if unknown.
        sequence: The nucleotide (or amino acid) sequence.
    """

    record_id: str
    accession: str
    start: str | None
    end: str | None
    taxonomy: dict[str, Taxon | None]
    sequence: str


def _parse_taxon(rank: str, value: str) -> Taxon | None:
    """Parse one ``<name>_<taxid>`` value, returning ``None`` for placeholders."""
    match = _TAXON_WITH_ID.match(value)
    name, taxid = (match["name"], int(match["taxid"])) if match else (value, None)
    if not name or name.startswith(f"{rank}_"):
        return None
    return Taxon(name=name, taxid=taxid)


def parse_sintax_header(header: str) -> tuple[str, dict[str, Taxon | None]]:
    """Split a SINTAX header into its record ID and taxonomy.

    Returns:
        The record ID and a mapping from each rank in ``RANKS`` to its taxon,
        or ``None`` where the rank is unknown or missing.

    Raises:
        Midori2HeaderError: If the header has no ``;tax=`` section or contains
            an unknown rank prefix.
    """
    record_id, separator, tax_section = header.partition(";tax=")
    if not separator or not record_id:
        msg = f"Not a SINTAX header: {header!r}"
        raise Midori2HeaderError(msg)

    taxonomy: dict[str, Taxon | None] = dict.fromkeys(RANKS)
    for field in _TAX_FIELD_SPLIT.split(tax_section.rstrip(";")):
        prefix, colon, value = field.partition(":")
        rank = _PREFIX_TO_RANK.get(prefix)
        if not colon or rank is None:
            msg = f"Unknown taxonomy field {field!r} in header {header!r}"
            raise Midori2HeaderError(msg)
        taxonomy[rank] = _parse_taxon(rank, value)
    return record_id, taxonomy


def parse_midori2_records(records: Iterable[FastaRecord]) -> Iterator[Midori2Record]:
    """Convert FASTA records with MIDORI2 SINTAX headers into ``Midori2Record``."""
    for record in records:
        record_id, taxonomy = parse_sintax_header(record.header)
        match = _RECORD_ID.match(record_id)
        yield Midori2Record(
            record_id=record_id,
            accession=match["accession"] if match else record_id,
            start=match["start"] if match else None,
            end=match["end"] if match else None,
            taxonomy=taxonomy,
            sequence=record.sequence.upper(),
        )
