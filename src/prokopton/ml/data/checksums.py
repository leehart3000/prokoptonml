"""Verify files against published SHA-256 checksums, including files inside zips.

A SHA-256 checksum is a fingerprint of a file's exact bytes. Comparing it with
the value published by the data provider proves that a download is complete and
unaltered.
"""

import gzip
import hashlib
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = [
    "ChecksumMismatchError",
    "parse_sha256_manifest",
    "read_gzip_lines_from_zip",
    "sha256_of_zip_member",
    "verify_zip_member",
]

_CHUNK_SIZE = 1024 * 1024
_SHA256_HEX_LENGTH = 64


class ChecksumMismatchError(ValueError):
    """Raised when a file's SHA-256 checksum does not match the expected value."""


def parse_sha256_manifest(lines: Iterable[str]) -> dict[str, str]:
    """Parse ``sha256sum``-style lines (``<hash>  <path>``) into a dictionary.

    Paths are normalised by removing a leading ``./``. Blank lines are ignored.

    Raises:
        ValueError: If a line is not in the expected format.
    """
    manifest: dict[str, str] = {}
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        digest, separator, path = line.partition("  ")
        if not separator or len(digest) != _SHA256_HEX_LENGTH:
            msg = f"Invalid checksum manifest line {line_number}: {line!r}"
            raise ValueError(msg)
        manifest[path.removeprefix("./")] = digest.lower()
    return manifest


def sha256_of_zip_member(zip_path: Path, member: str) -> str:
    """Return the SHA-256 hex digest of a file stored inside a zip archive.

    The member is streamed in chunks, so it is never fully held in memory.
    """
    digest = hashlib.sha256()
    with zipfile.ZipFile(zip_path) as archive, archive.open(member) as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_zip_member(zip_path: Path, member: str, expected_sha256: str) -> None:
    """Check that a zip member's SHA-256 matches the expected value.

    Raises:
        ChecksumMismatchError: If the checksums differ.
    """
    actual = sha256_of_zip_member(zip_path, member)
    if actual != expected_sha256.lower():
        msg = (
            f"Checksum mismatch for {member} in {zip_path}: "
            f"expected {expected_sha256}, got {actual}"
        )
        raise ChecksumMismatchError(msg)


def read_gzip_lines_from_zip(zip_path: Path, member: str) -> Iterator[str]:
    """Yield the text lines of a gzip-compressed file stored inside a zip archive.

    Decompression happens on the fly, so nothing is extracted to disk.
    """
    with (
        zipfile.ZipFile(zip_path) as archive,
        archive.open(member) as raw,
        gzip.open(raw, mode="rt", encoding="utf-8") as text,
    ):
        yield from text
