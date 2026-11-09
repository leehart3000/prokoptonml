"""Read FASTA records from a stream of text lines.

FASTA is a plain-text format for biological sequences. Each record starts with
a header line beginning with ``>``, followed by one or more lines of sequence.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

__all__ = ["FastaFormatError", "FastaRecord", "parse_fasta"]


class FastaFormatError(ValueError):
    """Raised when input does not follow the FASTA format."""


@dataclass(frozen=True, slots=True)
class FastaRecord:
    """A single FASTA record.

    Attributes:
        header: The header line, without the leading ``>``.
        sequence: The full sequence, with line breaks removed.
    """

    header: str
    sequence: str


def parse_fasta(lines: Iterable[str]) -> Iterator[FastaRecord]:
    """Yield FASTA records from an iterable of text lines.

    Records are yielded one at a time, so large files can be processed without
    loading them into memory. Blank lines are ignored. Sequence lines are joined
    without separators, and the final line of the final record is always kept.

    Raises:
        FastaFormatError: If sequence data appears before the first header, or
            a header is empty.
    """
    header: str | None = None
    parts: list[str] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield FastaRecord(header=header, sequence="".join(parts))
            header = line[1:].strip()
            if not header:
                msg = f"Empty FASTA header on line {line_number}"
                raise FastaFormatError(msg)
            parts = []
        elif header is None:
            msg = f"Sequence data before the first FASTA header on line {line_number}"
            raise FastaFormatError(msg)
        else:
            parts.append(line)
    if header is not None:
        yield FastaRecord(header=header, sequence="".join(parts))
