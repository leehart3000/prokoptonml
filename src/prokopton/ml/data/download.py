"""Download files safely, and only when they are not already present."""

import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

__all__ = ["ALLOWED_SCHEMES", "DownloadError", "download_file"]

ALLOWED_SCHEMES = frozenset({"https", "file"})
"""URL schemes that may be downloaded: HTTPS, plus local files for testing."""

_CHUNK_SIZE = 1024 * 1024


class DownloadError(RuntimeError):
    """Raised when a file cannot be downloaded."""


def download_file(url: str, destination: Path, *, timeout: float = 60.0) -> bool:
    """Download ``url`` to ``destination``, unless the file already exists.

    The file is first written to ``<destination>.part`` and only renamed once
    complete, so an interrupted download never leaves a truncated file behind.

    Returns:
        ``True`` if the file was downloaded, ``False`` if it already existed.

    Raises:
        DownloadError: If the URL scheme is not allowed or the download fails.
    """
    scheme = urlparse(url).scheme
    if scheme not in ALLOWED_SCHEMES:
        msg = f"Refusing to download from scheme {scheme!r}: {url}"
        raise DownloadError(msg)
    if destination.exists():
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,
            partial.open("wb") as output,
        ):
            shutil.copyfileobj(response, output, length=_CHUNK_SIZE)
    except OSError as error:
        partial.unlink(missing_ok=True)
        msg = f"Failed to download {url}: {error}"
        raise DownloadError(msg) from error
    partial.replace(destination)
    return True
