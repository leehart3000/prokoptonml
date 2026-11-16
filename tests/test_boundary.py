"""The boundary between the generic core and sequence-specific code.

Sequence-specific code lives in ``prokopton.ml.bio``. Every other module is
listed here as core, split (with both kinds of code, to be divided) or an
old import path kept for saved models. Core modules may not import ``bio``,
except for the imports listed in ``ALLOWED``, which may only shrink. A new
module outside ``bio`` must be added to one of the lists.
"""

import ast
from pathlib import Path

import prokopton.ml

ROOT = Path(prokopton.ml.__file__).parent

CORE = frozenset(
    {
        "",
        "__main__",
        "checks",
        "cli",
        "data",
        "data.checksums",
        "data.download",
        "evaluation",
        "inference",
        "inference.bundle",
        "inference.chain",
        "inference.load",
        "inference.models",
        "migrate",
        "release",
        "pipeline",
        "reporting",
        "reporting.charts",
        "serving",
        "serving.app",
        "stages",
        "training",
        "training.calibration",
        "training.metrics",
        "training.tracking",
    }
)
"""Generic modules (packages' ``__init__`` files are named by their package)."""

OLD_PATHS = frozenset({"features", "features.kmers"})
"""Old import paths, kept so that models saved before 0.2.0 still load."""

SPLIT = frozenset(
    {
        "config",
        "evaluation.scores",
        "inference.identify",
        "pipeline.calibrate",
        "pipeline.evaluate",
        "pipeline.sample",
        "pipeline.split",
        "pipeline.train",
        "reporting.card",
        "training.router",
    }
)
"""Modules with both kinds of code, to be divided."""

ALLOWED = frozenset(
    {
        ("cli", "bio.stages"),
    }
)
"""Core-to-bio imports that exist today, to be removed as code moves."""


def _modules() -> dict[str, Path]:
    found = {}
    for path in ROOT.rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def _imports(path: Path) -> set[str]:
    """The ``prokopton.ml`` modules a file imports (relative to the package)."""
    prefix = "prokopton.ml"
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == prefix:
                found.update(alias.name for alias in node.names)
            elif node.module.startswith(prefix + "."):
                base = node.module.removeprefix(prefix + ".")
                found.add(base)
                found.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(
                alias.name.removeprefix(prefix + ".")
                for alias in node.names
                if alias.name.startswith(prefix + ".")
            )
    return found


def _core_to_bio() -> set[tuple[str, str]]:
    modules = _modules()
    bio = {name for name in modules if name == "bio" or name.startswith("bio.")}
    return {
        (name, imported)
        for name in CORE
        for imported in _imports(modules[name])
        if imported in bio
    }


def test_every_module_is_classified() -> None:
    assert not (CORE & SPLIT or CORE & OLD_PATHS or SPLIT & OLD_PATHS)
    outside_bio = {
        name for name in _modules() if name != "bio" and not name.startswith("bio.")
    }
    assert outside_bio == CORE | SPLIT | OLD_PATHS


def test_core_imports_no_new_bio_modules() -> None:
    assert _core_to_bio() - ALLOWED == set()


def test_allowed_imports_are_all_still_needed() -> None:
    # When an allowed import goes, remove it from ALLOWED, so it can't return
    assert ALLOWED - _core_to_bio() == set()
