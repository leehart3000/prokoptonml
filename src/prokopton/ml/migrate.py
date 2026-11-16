"""Rewrite a schema 3 config file as schema 4, keeping its comments.

The rewrite is done on the text, so every comment and the order of the file
stay as they are:

- ``release`` moves from ``[data.source]`` to a new ``[data]`` table, and
  ``models_dir`` from ``[data.paths]`` to a new ``[paths]`` table, both
  before the first table;
- each section's schema 3 tables are renamed as its settings declare (for
  sequence apps, ``[data.source]`` becomes ``[bio.source]``, ``[validation]``
  becomes ``[bio.validation]``, and so on), with their sub-tables;
- ``schema = 3`` becomes ``schema = 4``.

The result is checked before it is returned: it must give exactly the same
settings as the original. Schemas 1 and 2 are no longer read (see
``prokopton.ml.config``).
"""

import re
import tomllib

from prokopton.ml.config import (
    CURRENT_SCHEMA,
    AppConfig,
    old_schema_message,
    schema_of,
)
from prokopton.ml.settings import declared

__all__ = ["CURRENT_SCHEMA", "MigrationError", "migrate_text", "schema_of"]

_HEADER = re.compile(
    r"^(?P<open>\[\[?)\s*(?P<path>[A-Za-z0-9_.-]+)\s*(?P<close>\]\]?)(?P<rest>.*)$"
)


class MigrationError(ValueError):
    """Raised when a config cannot be rewritten safely."""


_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)\s*=")


_MOVED_KEYS = {"data.source": "release", "data.paths": "models_dir"}
"""Keys that leave their schema 3 table for a new core table."""


def _renamed(path: str, tables: dict[str, str]) -> str:
    for old, new in tables.items():
        if path == old or path.startswith(old + "."):
            return new + path[len(old) :]
    return path


def _to_schema_4(text: str) -> str:
    """Rewrite a schema 3 file's text as schema 4 (see the module docstring)."""
    tables: dict[str, str] = {}
    for section in declared().sections.values():
        tables.update(section.schema_3_tables)
    lines = text.splitlines()
    output: list[str] = []
    taken: dict[str, str] = {}
    table = ""
    first_table: int | None = None
    for line in lines:
        stripped = line.strip()
        if re.match(r"^schema\s*=\s*3\s*(#.*)?$", stripped):
            output.append("schema = 4")
            continue
        if match := _HEADER.match(stripped):
            table = match["path"]
            if first_table is None:
                first_table = len(output)
                while first_table > 0 and output[first_table - 1].startswith("#"):
                    first_table -= 1  # Comments just above it belong to it
            path = _renamed(table, tables)
            output.append(f"{match['open']}{path}{match['close']}{match['rest']}")
            continue
        key = _KEY.match(stripped)
        if key and _MOVED_KEYS.get(table) == key["key"]:
            taken[table] = stripped
            continue
        output.append(line)
    added: list[str] = []
    if "data.source" in taken:
        added += ["[data]", taken["data.source"], ""]
    if "data.paths" in taken:
        added += ["[paths]", taken["data.paths"], ""]
    at = len(output) if first_table is None else first_table
    output[at:at] = added
    return "\n".join(output).rstrip("\n") + "\n"


def migrate_text(text: str) -> str:
    """Rewrite a schema 3 config file's text as schema 4.

    Raises:
        MigrationError: If the file is already schema 4, is schema 1 or 2 (no
            longer read), or if the rewritten file would not give exactly the
            same settings.
    """
    try:
        original = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        msg = f"Not a valid TOML file: {error}"
        raise MigrationError(msg) from error
    if schema_of(original) >= CURRENT_SCHEMA:
        msg = f"This config is already schema {CURRENT_SCHEMA}"
        raise MigrationError(msg)
    if schema_of(original) < 3:
        raise MigrationError(old_schema_message(schema_of(original)))
    try:
        before = AppConfig.model_validate(original)
    except ValueError as error:
        msg = f"Not a valid config: {error}"
        raise MigrationError(msg) from error
    result = _to_schema_4(text)
    try:
        after = AppConfig.model_validate(tomllib.loads(result))
    except (tomllib.TOMLDecodeError, ValueError) as error:
        msg = f"The rewritten config is not valid: {error}"
        raise MigrationError(msg) from error
    if before != after:
        msg = "The rewritten config would not give the same settings"
        raise MigrationError(msg)
    return result
