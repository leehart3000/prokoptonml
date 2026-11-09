"""Basic tests for the prokoptonml package."""

from importlib.metadata import version

import prokoptonml


def test_version_matches_package_metadata() -> None:
    assert prokoptonml.__version__ == version("prokoptonml")
