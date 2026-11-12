"""Tests for prokopton.ml.inference.inputs."""

import pytest

from prokopton.ml.config import InferenceConfig
from prokopton.ml.inference.inputs import InputError, parse_submission

RULES = InferenceConfig(max_length=1000, max_ambiguous_fraction=0.1)
BASES = "ACGT" * 50  # 200 bases


def parse(text: str) -> tuple[str, str, tuple[str, ...]]:
    found = parse_submission(text, RULES, min_length=100)
    return found.name, found.sequence, found.notes


def code(text: str) -> str:
    with pytest.raises(InputError) as raised:
        parse_submission(text, RULES, min_length=100)
    return raised.value.code


def test_plain_and_fasta() -> None:
    assert parse(BASES.lower()) == ("", BASES, ())
    assert parse(f">my sample\n{BASES[:100]}\n{BASES[100:]}\n") == (
        "my sample",
        BASES,
        (),
    )
    assert parse_submission(BASES, RULES, min_length=100).length == 200


def test_genbank_style_numbers_and_spaces_are_ignored() -> None:
    text = f"1 {BASES[:60]}\n61 {BASES[60:]}"

    assert parse(text)[1] == BASES


def test_gaps_and_rna_are_cleaned_with_notes() -> None:
    _, sequence, notes = parse(BASES[:100] + "--.." + BASES[100:])
    assert sequence == BASES
    assert notes == ("Alignment gaps ('-' or '.') were removed.",)

    _, sequence, notes = parse(BASES.replace("T", "U"))
    assert sequence == BASES
    assert notes == ("This looked like RNA, so U was read as T.",)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("   ", "empty"),
        (f"@read1\n{BASES}\n+\n{'I' * 200}", "raw_reads"),
        (f">a\n{BASES}\n>b\n{BASES}", "multiple_sequences"),
        (f"note\n>a\n{BASES}", "invalid_format"),
        ("MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP" * 3, "protein"),
        (BASES + "XZ", "invalid_characters"),
        (BASES[:99], "too_short"),
        ("A" * 1001, "too_long"),
        (BASES[:150] + "N" * 50, "too_ambiguous"),
    ],
)
def test_declined_with_a_reason(text: str, expected: str) -> None:
    assert code(text) == expected


def test_messages_say_what_to_do() -> None:
    with pytest.raises(InputError, match="one sequence at a time"):
        parse_submission(f">a\n{BASES}\n>b\n{BASES}", RULES, min_length=100)
    with pytest.raises(InputError, match="'X', 'Z'"):
        parse_submission(BASES + "XZ", RULES, min_length=100)
