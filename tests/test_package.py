"""Basic tests for the prokopton.ml package."""

from importlib.metadata import version

import prokopton.ml


def test_version_matches_package_metadata() -> None:
    assert prokopton.ml.__version__ == version("prokopton-ml")
