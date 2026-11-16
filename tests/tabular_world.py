"""A small made-up table for the tabular tests: three kinds of thing, told
apart by size and colour, with a few values missing."""

import random
from pathlib import Path
from typing import Any

HEADER = "size,weight,colour,kind\n"


def write_table(path: Path, rows: int = 90) -> Path:
    """Three kinds, told apart by size and colour; a few values missing."""
    pick = random.Random(1)
    lines = [HEADER]
    for row in range(rows):
        kind = ("small", "medium", "large")[row % 3]
        size = {"small": 1.0, "medium": 5.0, "large": 9.0}[kind] + pick.random()
        colour = {"small": "red", "medium": "green", "large": "blue"}[kind]
        weight = "NA" if row % 10 == 0 else f"{size * 2:.2f}"
        colour = "" if row % 15 == 0 else colour
        lines.append(f"{size:.2f},{weight},{colour},{kind}\n")
    lines.append("3.0,6.0,red,NA\n")  # No target: left out
    path.write_text("".join(lines), "utf-8")
    return path


def app_data(tmp_path: Path, **model: Any) -> dict[str, Any]:
    table = {
        "path": str(write_table(tmp_path / "things.csv")),
        "target": "kind",
        "numeric": ["size", "weight"],
        "categorical": ["colour"],
    }
    return {
        "schema": 4,
        "name": "things",
        "title": "Things",
        "data": {"release": "2026-1"},
        "paths": {"models_dir": str(tmp_path / "models")},
        "models": {
            "kind": {
                "kind": "core.tabular-classifier",
                "table": table,
                "training": {"c_values": [0.01, 1.0]},
                **model,
            }
        },
        "pipeline": {"stages": ["split", "train", "evaluate", "card"]},
        "stages": {
            name: {"kind": f"core.tabular-{name}", "model": "kind"}
            for name in ("split", "train", "evaluate", "card")
        },
    }
