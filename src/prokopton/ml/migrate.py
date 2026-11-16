"""Rewrite a schema 1 config file as schema 2, keeping its comments.

The rewrite is done on the text, so every comment and the order of the file
stay as they are:

- ``schema = 2`` is added before the first setting;
- ``[router]``, ``[marker]`` and ``[specialists.<name>]`` tables (and their
  sub-tables) become ``[models.<name>]``, each with its ``kind``;
- ``[pipeline]`` and one ``[stages.<name>]`` table per stage are added at the
  end: the stages a schema 1 app ran (see ``config.default_stages``).

The result is checked before it is returned: read as schema 2, it must give
exactly the same settings as the original read as schema 1.
"""

import re
import tomllib
from typing import Any

from prokopton.ml.config import AppConfig, convert_schema_1

__all__ = ["MigrationError", "is_schema_1", "migrate_text"]

_HEADER = re.compile(
    r"^(?P<open>\[\[?)\s*(?P<path>[A-Za-z0-9_.-]+)\s*(?P<close>\]\]?)(?P<rest>.*)$"
)
_KINDS = {"classifier": "bio.kmer-classifier", "specialist": "bio.specialist"}


class MigrationError(ValueError):
    """Raised when a config cannot be rewritten safely."""


def is_schema_1(data: dict[str, Any]) -> bool:
    """Whether config data (as read from TOML) is schema 1."""
    return not ({"schema", "schema_version"} & set(data))


def _model_of(path: str) -> tuple[str, str, str] | None:
    """For a table path in a schema 1 file: (model name, kind, rest of path)."""
    parts = path.split(".")
    if parts[0] in ("router", "marker"):
        return parts[0], "classifier", ".".join(parts[1:])
    if parts[0] == "specialists" and len(parts) > 1:
        return parts[1], "specialist", ".".join(parts[2:])
    return None


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pipeline_text(converted: dict[str, Any]) -> str:
    stages = converted["stages"]
    lines = [
        "",
        "# The pipeline: its stages, in the order they run (prokopton-ml stages).",
        "# Added by prokopton-ml migrate-config.",
        "[pipeline]",
        "stages = [",
        *(f"  {_quote(name)}," for name in converted["pipeline"]["stages"]),
        "]",
    ]
    for name, stage in stages.items():
        lines += ["", f"[stages.{name}]", f"kind = {_quote(stage['kind'])}"]
        if "model" in stage:
            lines.append(f"model = {_quote(stage['model'])}")
    return "\n".join(lines) + "\n"


def migrate_text(text: str) -> str:
    """Rewrite a schema 1 config file's text as schema 2.

    Raises:
        MigrationError: If the file is already schema 2, or if the rewritten
            file would not give exactly the same settings.
    """
    try:
        original = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        msg = f"Not a valid TOML file: {error}"
        raise MigrationError(msg) from error
    if not is_schema_1(original):
        msg = "This config is already schema 2"
        raise MigrationError(msg)
    try:
        before = AppConfig.model_validate(original)
    except ValueError as error:
        msg = f"Not a valid config: {error}"
        raise MigrationError(msg) from error
    lines = text.splitlines()
    own_table = set()
    for line in lines:
        match = _HEADER.match(line.strip())
        found = _model_of(match["path"]) if match else None
        if match and found and not found[2] and match["open"] == "[":
            own_table.add(found[0])
    output: list[str] = []
    with_kind: set[str] = set()
    schema_added = False
    for line in lines:
        stripped = line.strip()
        if not schema_added and stripped and not stripped.startswith("#"):
            output += ["schema = 2", ""]
            schema_added = True
        match = _HEADER.match(stripped)
        found = _model_of(match["path"]) if match else None
        if match is None or found is None:
            output.append(line)
            continue
        name, kind, rest = found
        is_own = not rest and match["open"] == "["
        if name not in own_table and name not in with_kind:
            # Only sub-tables in the file: open the model's own table first
            output += [f"[models.{name}]", f"kind = {_quote(_KINDS[kind])}", ""]
            with_kind.add(name)
        path = f"models.{name}" + (f".{rest}" if rest else "")
        output.append(f"{match['open']}{path}{match['close']}{match['rest']}")
        if is_own:
            output.append(f"kind = {_quote(_KINDS[kind])}")
            with_kind.add(name)
    converted = convert_schema_1(original)
    for name in converted["models"]:
        if name not in with_kind:
            # A model with no table of its own in the file (the default router)
            kind = converted["models"][name]["kind"]
            output += ["", f"[models.{name}]", f"kind = {_quote(kind)}"]
    result = "\n".join(output).rstrip("\n") + "\n" + _pipeline_text(converted)
    try:
        after = AppConfig.model_validate(tomllib.loads(result))
    except (tomllib.TOMLDecodeError, ValueError) as error:
        msg = f"The rewritten config is not valid: {error}"
        raise MigrationError(msg) from error
    if before != after:
        msg = "The rewritten config would not give the same settings"
        raise MigrationError(msg)
    return result
