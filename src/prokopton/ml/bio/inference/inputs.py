"""Check a submitted sequence and turn it into clean DNA, or explain why not.

People paste sequences in many forms: plain text, a FASTA record, a GenBank
"ORIGIN" block with numbers and spaces, RNA with U for T, or an alignment
with gaps. These are cleaned, with a note for each change. Things the site
cannot identify are declined with a reason a person can act on: raw reads
(FASTQ), several sequences at once, protein sequences, characters that are
not DNA, too many ambiguous bases, or a sequence that is too short or long.
"""

import re
from dataclasses import dataclass, field

from prokopton.ml.config import InferenceConfig

__all__ = ["DNA", "InputError", "Submission", "parse_submission"]

DNA = frozenset("ACGTRYSWKMBDHVN")
"""Bases and IUPAC ambiguity codes."""
_AMBIGUOUS = frozenset("RYSWKMBDHVN")
# Letters found in proteins but not in DNA (or its ambiguity codes)
_PROTEIN_ONLY = frozenset("EFILPQ")
_IGNORED = re.compile(r"[\s0-9]")


class InputError(ValueError):
    """A submission that cannot be identified, with a reason to show."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Submission:
    """A cleaned DNA sequence, ready to identify."""

    name: str
    sequence: str
    notes: tuple[str, ...] = field(default=())

    @property
    def length(self) -> int:
        return len(self.sequence)


def _records(text: str) -> tuple[str, str]:
    """Split FASTA text into (name, body); plain text has no name."""
    lines = text.strip().splitlines()
    headers = [line for line in lines if line.startswith(">")]
    if len(headers) > 1:
        msg = f"Found {len(headers)} sequences. Please submit one sequence at a time."
        raise InputError("multiple_sequences", msg)
    if headers:
        if not lines[0].startswith(">"):
            msg = "Text was found before the FASTA header line ('>')."
            raise InputError("invalid_format", msg)
        return lines[0][1:].strip(), "".join(lines[1:])
    return "", "".join(lines)


def parse_submission(text: str, rules: InferenceConfig, min_length: int) -> Submission:
    """Clean a submission, or raise ``InputError`` explaining the problem.

    Args:
        text: What the person pasted or uploaded.
        rules: The app's inference settings.
        min_length: The shortest sequence accepted (the router's minimum).
    """
    stripped = text.strip()
    if not stripped:
        raise InputError("empty", "Please paste or upload a DNA sequence.")
    lines = stripped.splitlines()
    if lines[0].startswith("@") and any(line.startswith("+") for line in lines):
        msg = (
            "This looks like raw sequencing reads (FASTQ). The site identifies "
            "one assembled barcode sequence; please assemble or choose a "
            "consensus sequence first."
        )
        raise InputError("raw_reads", msg)
    name, body = _records(stripped)
    sequence = _IGNORED.sub("", body).upper()
    notes = []
    if "-" in sequence or "." in sequence:
        sequence = sequence.replace("-", "").replace(".", "")
        notes.append("Alignment gaps ('-' or '.') were removed.")
    if "U" in sequence and "T" not in sequence:
        sequence = sequence.replace("U", "T")
        notes.append("This looked like RNA, so U was read as T.")
    other = sorted(set(sequence) - DNA)
    if other:
        if set(other) & _PROTEIN_ONLY and len(set(sequence) & _PROTEIN_ONLY) >= 3:
            msg = (
                "This looks like a protein sequence. The site identifies DNA "
                "barcodes only."
            )
            raise InputError("protein", msg)
        shown = ", ".join(repr(c) for c in other[:5])
        msg = f"The sequence contains characters that are not DNA: {shown}."
        raise InputError("invalid_characters", msg)
    if len(sequence) < min_length:
        msg = (
            f"The sequence has {len(sequence):,} bases; at least {min_length:,} "
            "are needed for a reliable answer."
        )
        raise InputError("too_short", msg)
    if len(sequence) > rules.max_length:
        msg = (
            f"The sequence has {len(sequence):,} bases; the most accepted is "
            f"{rules.max_length:,} (about a whole mitochondrial genome)."
        )
        raise InputError("too_long", msg)
    ambiguous = sum(base in _AMBIGUOUS for base in sequence) / len(sequence)
    if ambiguous > rules.max_ambiguous_fraction:
        msg = (
            f"{ambiguous:.0%} of the bases are ambiguous (such as N); at most "
            f"{rules.max_ambiguous_fraction:.0%} are accepted."
        )
        raise InputError("too_ambiguous", msg)
    return Submission(name=name, sequence=sequence, notes=tuple(notes))
