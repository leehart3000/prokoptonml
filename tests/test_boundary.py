"""The boundary between the generic core and sequence-specific code.

Sequence-specific code is moving to ``prokopton.ml.bio``. Until it has moved,
each module is listed here as core, bio or split (to be divided), and core
modules may not import bio modules, except for the imports listed in
``ALLOWED``, which are today's and may only shrink. A new module must be
added to one of the lists.
"""

import ast
from pathlib import Path

import prokopton.ml

ROOT = Path(prokopton.ml.__file__).parent

CORE = frozenset(
    {
        "",
        "__main__",
        "cli",
        "data",
        "data.checksums",
        "data.download",
        "evaluation",
        "features",
        "inference",
        "inference.bundle",
        "inference.load",
        "pipeline",
        "reporting",
        "reporting.charts",
        "serving",
        "serving.app",
        "sources",
        "specialists",
        "training",
        "training.calibration",
        "training.metrics",
        "training.tracking",
    }
)
"""Generic modules (packages' ``__init__`` files are named by their package)."""

BIO = frozenset(
    {
        "data.fasta",
        "data.taxonomy",
        "features.fragments",
        "features.kmers",
        "inference.inputs",
        "pipeline.ingest",
        "pipeline.leakage",
        "pipeline.reference",
        "pipeline.specialist_evaluate",
        "pipeline.validate",
        "reporting.specialist_card",
        "sources.midori2",
        "sources.refseq",
        "specialists.identify",
        "specialists.identity",
    }
)
"""Sequence-specific modules, to move to ``prokopton.ml.bio``."""

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
        ("cli", "data.taxonomy"),
        ("cli", "pipeline.ingest"),
        ("cli", "pipeline.leakage"),
        ("cli", "pipeline.reference"),
        ("cli", "pipeline.specialist_evaluate"),
        ("cli", "pipeline.validate"),
        ("cli", "reporting.specialist_card"),
        ("cli", "sources.refseq"),
        ("inference.bundle", "pipeline.reference"),
        ("inference.bundle", "specialists.identify"),
        ("inference.load", "pipeline.reference"),
        ("inference.load", "specialists.identify"),
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
    return {
        (name, imported)
        for name in CORE
        for imported in _imports(modules[name])
        if imported in BIO
    }


def test_every_module_is_classified() -> None:
    assert not (CORE & BIO or CORE & SPLIT or BIO & SPLIT)
    assert set(_modules()) == CORE | BIO | SPLIT


def test_core_imports_no_new_bio_modules() -> None:
    assert _core_to_bio() - ALLOWED == set()


def test_allowed_imports_are_all_still_needed() -> None:
    # When an allowed import goes, remove it from ALLOWED, so it can't return
    assert ALLOWED - _core_to_bio() == set()
