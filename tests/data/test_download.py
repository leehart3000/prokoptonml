"""Tests for prokopton.ml.data.download."""

from pathlib import Path

import pytest

from prokopton.ml.data.download import DownloadError, download_file


def test_downloads_file_url(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("hello", encoding="utf-8")
    destination = tmp_path / "nested" / "copy.txt"

    assert download_file(source.as_uri(), destination) is True
    assert destination.read_text(encoding="utf-8") == "hello"
    assert not destination.with_name("copy.txt.part").exists()


def test_skips_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "existing.txt"
    destination.write_text("keep me", encoding="utf-8")

    assert download_file((tmp_path / "missing").as_uri(), destination) is False
    assert destination.read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("url", ["http://example.org/x", "ftp://example.org/x"])
def test_rejects_disallowed_schemes(tmp_path: Path, url: str) -> None:
    with pytest.raises(DownloadError, match="Refusing to download"):
        download_file(url, tmp_path / "x")


def test_failed_download_leaves_no_files(tmp_path: Path) -> None:
    destination = tmp_path / "out.txt"

    with pytest.raises(DownloadError, match="Failed to download"):
        download_file((tmp_path / "missing.txt").as_uri(), destination)

    assert list(tmp_path.iterdir()) == []
