"""Tests for prokopton.ml.data.checksums."""

import gzip
import hashlib
import zipfile
from pathlib import Path

import pytest

from prokopton.ml.data.checksums import (
    ChecksumMismatchError,
    parse_sha256_manifest,
    read_gzip_lines_from_zip,
    sha256_of_zip_member,
    verify_zip_member,
)

MEMBER = "FORMAT/uniq/example.fasta.gz"
CONTENT = b">seq1\nACGT\n>seq2\nGGCC\n"


@pytest.fixture
def zip_with_gzip_member(tmp_path: Path) -> tuple[Path, str]:
    """Create a zip containing one gzip file; return its path and the gzip SHA-256."""
    compressed = gzip.compress(CONTENT)
    zip_path = tmp_path / "archive.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(MEMBER, compressed)
    return zip_path, hashlib.sha256(compressed).hexdigest()


def test_parse_manifest_normalises_paths() -> None:
    digest = "a" * 64
    lines = [f"{digest}  ./FORMAT/uniq/example.fasta.gz\n", "\n"]

    assert parse_sha256_manifest(lines) == {MEMBER: digest}


def test_parse_manifest_lowercases_digests() -> None:
    manifest = parse_sha256_manifest([f"{'A' * 64}  file.txt"])

    assert manifest == {"file.txt": "a" * 64}


@pytest.mark.parametrize("line", ["not-a-checksum", f"{'a' * 63}  file.txt"])
def test_parse_manifest_rejects_invalid_lines(line: str) -> None:
    with pytest.raises(ValueError, match="Invalid checksum manifest line 1"):
        parse_sha256_manifest([line])


def test_sha256_of_zip_member(zip_with_gzip_member: tuple[Path, str]) -> None:
    zip_path, expected = zip_with_gzip_member

    assert sha256_of_zip_member(zip_path, MEMBER) == expected


def test_verify_zip_member_accepts_matching_checksum(
    zip_with_gzip_member: tuple[Path, str],
) -> None:
    zip_path, expected = zip_with_gzip_member

    verify_zip_member(zip_path, MEMBER, expected.upper())


def test_verify_zip_member_rejects_wrong_checksum(
    zip_with_gzip_member: tuple[Path, str],
) -> None:
    zip_path, _ = zip_with_gzip_member

    with pytest.raises(ChecksumMismatchError, match="Checksum mismatch"):
        verify_zip_member(zip_path, MEMBER, "0" * 64)


def test_read_gzip_lines_from_zip(zip_with_gzip_member: tuple[Path, str]) -> None:
    zip_path, _ = zip_with_gzip_member

    lines = list(read_gzip_lines_from_zip(zip_path, MEMBER))

    assert lines == [">seq1\n", "ACGT\n", ">seq2\n", "GGCC\n"]
