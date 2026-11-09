"""Tests for prokopton.ml.data.fasta."""

import pytest

from prokopton.ml.data.fasta import FastaFormatError, FastaRecord, parse_fasta


def test_parses_multiline_records() -> None:
    lines = [">seq1 description\n", "ACGT\n", "TTGA\n", ">seq2\n", "GGCC\n"]

    records = list(parse_fasta(lines))

    assert records == [
        FastaRecord(header="seq1 description", sequence="ACGTTTGA"),
        FastaRecord(header="seq2", sequence="GGCC"),
    ]


def test_keeps_last_line_of_last_record_without_trailing_newline() -> None:
    records = list(parse_fasta([">seq1\n", "ACGT\n", "TTGA"]))

    assert records == [FastaRecord(header="seq1", sequence="ACGTTTGA")]


def test_ignores_blank_lines() -> None:
    records = list(parse_fasta(["\n", ">seq1\n", "\n", "AC\n", "  \n", "GT\n"]))

    assert records == [FastaRecord(header="seq1", sequence="ACGT")]


def test_allows_records_with_empty_sequences() -> None:
    records = list(parse_fasta([">empty\n", ">seq2\n", "AC\n"]))

    assert records == [
        FastaRecord(header="empty", sequence=""),
        FastaRecord(header="seq2", sequence="AC"),
    ]


def test_empty_input_yields_nothing() -> None:
    assert list(parse_fasta([])) == []


def test_sequence_before_header_raises() -> None:
    with pytest.raises(FastaFormatError, match="before the first FASTA header"):
        list(parse_fasta(["ACGT\n", ">seq1\n"]))


def test_empty_header_raises() -> None:
    with pytest.raises(FastaFormatError, match="Empty FASTA header on line 1"):
        list(parse_fasta([">\n", "ACGT\n"]))
