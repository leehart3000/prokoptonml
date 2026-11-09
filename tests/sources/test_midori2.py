"""Tests for prokopton.ml.sources.midori2."""

import pytest

from prokopton.ml.data.fasta import FastaRecord
from prokopton.ml.sources.midori2 import (
    RANKS,
    Midori2HeaderError,
    Taxon,
    parse_midori2_records,
    parse_sintax_header,
)

# A real header from MIDORI2 GB273, with unknown ranks filled by placeholders
PLACEHOLDER_HEADER = (
    "MH535936.1.<1.>890;tax=k:Eukaryota_2759,"
    "p:phylum_class_order_family_genus_Amoebozoa sp._1892891,"
    "c:class_order_family_genus_Amoebozoa sp._1892891,"
    "o:order_family_genus_Amoebozoa sp._1892891,"
    "f:family_genus_Amoebozoa sp._1892891,"
    "g:genus_Amoebozoa sp._1892891,"
    "s:Amoebozoa sp._1892891;"
)

FULL_HEADER = (
    "AB123456.1.1.658;tax=k:Eukaryota_2759,p:Arthropoda_6656,c:Insecta_50557,"
    "o:Diptera_7147,f:Culicidae_7157,g:Anopheles_7164,s:Anopheles gambiae_7165;"
)


def test_parses_full_taxonomy() -> None:
    record_id, taxonomy = parse_sintax_header(FULL_HEADER)

    assert record_id == "AB123456.1.1.658"
    assert list(taxonomy) == list(RANKS)
    assert taxonomy["order"] == Taxon(name="Diptera", taxid=7147)
    assert taxonomy["species"] == Taxon(name="Anopheles gambiae", taxid=7165)


def test_placeholder_ranks_are_missing() -> None:
    _, taxonomy = parse_sintax_header(PLACEHOLDER_HEADER)

    assert taxonomy["superkingdom"] == Taxon(name="Eukaryota", taxid=2759)
    for rank in ("phylum", "class", "order", "family", "genus"):
        assert taxonomy[rank] is None
    assert taxonomy["species"] == Taxon(name="Amoebozoa sp.", taxid=1892891)


def test_taxon_without_taxid() -> None:
    _, taxonomy = parse_sintax_header("id1;tax=k:Eukaryota,s:Unnamed thing;")

    assert taxonomy["superkingdom"] == Taxon(name="Eukaryota", taxid=None)
    assert taxonomy["species"] == Taxon(name="Unnamed thing", taxid=None)
    assert taxonomy["phylum"] is None


def test_names_containing_commas_are_kept_whole() -> None:
    _, taxonomy = parse_sintax_header("id1;tax=k:Eukaryota_2759,s:Odd, name_42;")

    assert taxonomy["species"] == Taxon(name="Odd, name", taxid=42)


@pytest.mark.parametrize(
    "header",
    ["AB123456.1", ";tax=k:Eukaryota_2759;", "id1;tax=x:Unknown_1;"],
)
def test_invalid_headers_raise(header: str) -> None:
    with pytest.raises(Midori2HeaderError):
        parse_sintax_header(header)


def test_parse_records_extracts_accession_and_coordinates() -> None:
    fasta = [
        FastaRecord(header=PLACEHOLDER_HEADER, sequence="acgt"),
        FastaRecord(header="custom-id;tax=k:Eukaryota_2759;", sequence="GGCC"),
    ]

    first, second = parse_midori2_records(fasta)

    assert first.record_id == "MH535936.1.<1.>890"
    assert first.accession == "MH535936.1"
    assert (first.start, first.end) == ("<1", ">890")
    assert first.sequence == "ACGT"
    assert second.accession == "custom-id"
    assert (second.start, second.end) == (None, None)
