"""Tests that prokopton.ml works alongside the prokopton core package."""

from pathlib import Path

import prokopton.core
from prokopton.core.config import ConfigModel, load_config

import prokopton.ml


class ExampleConfig(ConfigModel):
    name: str


def test_core_and_ml_share_the_prokopton_namespace() -> None:
    assert prokopton.core.__name__ == "prokopton.core"
    assert prokopton.ml.__name__ == "prokopton.ml"


def test_core_config_loader_is_usable(tmp_path: Path) -> None:
    path = tmp_path / "app.toml"
    path.write_text('name = "dnabarcode"\n', encoding="utf-8")

    assert load_config(path, ExampleConfig).name == "dnabarcode"
