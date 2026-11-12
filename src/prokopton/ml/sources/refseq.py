"""NCBI RefSeq mitogenomes: curated COI sequences to check labels against.

RefSeq records (accessions starting ``NC_``) are curated by NCBI, so each
mitogenome's COI gene is a trustworthy example of its species. They are
fetched from NCBI's public E-utilities: the search lists the mitogenomes of
a taxon, the summaries give each one's organism, and the coding sequences
give its COI gene (named COX1, COI or similar in the annotation).

Only organisms with a plain two-word species name are kept (a subspecies is
read as its species); forms such as "Anopheles albitarsis F" or
"Chagasia sp. 1" are listed as skipped. The anchors are written as FASTA,
with a TSV listing every mitogenome found and what was done with it, so the
file can be reviewed and committed.
"""

import json
import logging
import re
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from prokopton.ml.data.fasta import parse_fasta

__all__ = [
    "COX1_NAMES",
    "Anchor",
    "AnchorError",
    "binomial",
    "fetch_anchors",
    "read_anchors",
]

logger = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
COX1_NAMES = frozenset({"COX1", "COI", "CO1", "COXI", "COX-1", "CYTOCHROME OXIDASE I"})
"""Gene names used for COI in mitogenome annotations (compared in capitals)."""
BATCH = 100
_GENE = re.compile(r"\[gene=([^\]]+)\]")
_ACCESSION = re.compile(r"^lcl\|(.+?)_cds_")

Fetch = Callable[[str], bytes]


class AnchorError(ValueError):
    """Raised when a specialist has no anchors configured."""


@dataclass(frozen=True, slots=True)
class Anchor:
    """The COI gene of one RefSeq mitogenome."""

    accession: str
    species: str
    sequence: str


def _urlopen(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:
        data: bytes = response.read()
    return data


def _url(tool: str, **params: str | int) -> str:
    query = urllib.parse.urlencode({**params, "tool": "prokopton-ml"})
    return f"{EUTILS}{tool}.fcgi?{query}"


def binomial(organism: str) -> str | None:
    """The species name of an organism, or None if it is not a plain species.

    "Culex pipiens pipiens" gives "Culex pipiens"; "Anopheles albitarsis F",
    "Anopheles nr. costai" and "Chagasia sp. 1 SJ-2020" give None.
    """
    words = organism.split()
    if len(words) < 2 or not words[0][:1].isupper():
        return None
    if not re.fullmatch(r"[a-z]+", words[1]):
        return None
    if len(words) > 2 and not re.fullmatch(r"[a-z]+", words[2]):
        return None
    return f"{words[0]} {words[1]}"


def fetch_anchors(taxon: str, output: Path, fetch: Fetch = _urlopen) -> list[Anchor]:
    """Fetch the COI genes of a taxon's RefSeq mitogenomes into ``output``.

    Writes ``output`` (FASTA, headers ``>accession species``) and
    ``output`` with the suffix ``.tsv`` (every mitogenome, and whether it
    was used).
    """
    term = f"{taxon}[Organism] AND refseq[filter] AND mitochondrion[filter]"
    term += " AND complete genome[Title]"
    found = json.loads(
        fetch(_url("esearch", db="nuccore", term=term, retmax=10_000, retmode="json"))
    )
    ids = found["esearchresult"]["idlist"]
    organisms: dict[str, str] = {}
    sequences: dict[str, str] = {}
    for start in range(0, len(ids), BATCH):
        batch = ",".join(ids[start : start + BATCH])
        result = json.loads(
            fetch(_url("esummary", db="nuccore", id=batch, retmode="json"))
        )["result"]
        for uid in result["uids"]:
            organisms[result[uid]["accessionversion"]] = result[uid]["organism"]
        text = fetch(
            _url(
                "efetch", db="nuccore", id=batch, rettype="fasta_cds_na", retmode="text"
            )
        ).decode("utf-8")
        for record in parse_fasta(text.splitlines()):
            gene = _GENE.search(record.header)
            source = _ACCESSION.match(record.header)
            if gene and source and gene.group(1).upper() in COX1_NAMES:
                sequences.setdefault(source.group(1), record.sequence.upper())

    anchors: list[Anchor] = []
    rows = ["accession\torganism\tspecies\tcox1_length\tused"]
    for accession, organism in sorted(organisms.items()):
        species = binomial(organism)
        sequence = sequences.get(accession, "")
        if species is None:
            used = "no: not a plain species name"
        elif not sequence:
            used = "no: no COI gene in the annotation"
        else:
            used = "yes"
            anchors.append(Anchor(accession, species, sequence))
        rows.append(
            f"{accession}\t{organism}\t{species or ''}\t{len(sequence)}\t{used}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(f">{a.accession} {a.species}\n{a.sequence}\n" for a in anchors), "utf-8"
    )
    output.with_suffix(".tsv").write_text("\n".join(rows) + "\n", "utf-8")
    logger.info(
        "%d of %d RefSeq mitogenomes of %s give a COI anchor -> %s",
        len(anchors),
        len(organisms),
        taxon,
        output,
    )
    return anchors


def read_anchors(path: Path) -> list[Anchor]:
    """Read anchors written by ``fetch_anchors``."""
    anchors = []
    for record in parse_fasta(path.read_text("utf-8").splitlines()):
        accession, _, species = record.header.partition(" ")
        anchors.append(Anchor(accession, species, record.sequence))
    return anchors
