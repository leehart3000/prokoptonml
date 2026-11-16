"""Rewrite an older config file as schema 3, keeping its comments.

The rewrite is done on the text, so every comment and the order of the file
stay as they are. From schema 1:

- ``[router]``, ``[marker]`` and ``[specialists.<name>]`` tables (and their
  sub-tables) become ``[models.<name>]``, each with its ``kind``;
- ``[pipeline]`` and one ``[stages.<name>]`` table per stage are added at the
  end: the stages a schema 1 app ran (see ``config.default_stages``).

From schema 2 (and so from 1):

- ``[inference]`` is replaced by ``[serving]`` and one
  ``[serving.steps.<name>]`` table per serving step, at the end; each setting
  moves to its step with the comments above it (``route_threshold`` to every
  specialist step).

``schema = 3`` is set before the first setting. The result is checked before
it is returned: it must give exactly the same settings as the original.
"""

import re
import tomllib
from typing import Any

from prokopton.ml.config import AppConfig, convert_schema_1, convert_schema_2

__all__ = [
    "CURRENT_SCHEMA",
    "MigrationError",
    "is_schema_1",
    "migrate_text",
    "schema_of",
]

CURRENT_SCHEMA = 3

_HEADER = re.compile(
    r"^(?P<open>\[\[?)\s*(?P<path>[A-Za-z0-9_.-]+)\s*(?P<close>\]\]?)(?P<rest>.*)$"
)
_KINDS = {"classifier": "bio.kmer-classifier", "specialist": "bio.specialist"}


class MigrationError(ValueError):
    """Raised when a config cannot be rewritten safely."""


def schema_of(data: dict[str, Any]) -> int:
    """The schema of config data (as read from TOML): 1 if it doesn't say."""
    found = data.get("schema", data.get("schema_version", 1))
    return found if isinstance(found, int) else 1


def is_schema_1(data: dict[str, Any]) -> bool:
    """Whether config data (as read from TOML) is schema 1."""
    return schema_of(data) == 1


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


def _to_schema_2(text: str, original: dict[str, Any]) -> str:
    """Rewrite a schema 1 file's text as schema 2."""
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
    return "\n".join(output).rstrip("\n") + "\n" + _pipeline_text(converted)


_STEP_OF_KEY = {
    **dict.fromkeys(("max_length", "max_ambiguous_fraction"), "input"),
    **dict.fromkeys(
        (
            "barcode_gene",
            "long_input",
            "window",
            "step",
            "region_probability",
            "unsupported_note",
        ),
        "marker",
    ),
    "route_threshold": "specialist",
}
_KEY = re.compile(r"^(?P<key>[A-Za-z0-9_-]+)\s*=")


def _to_schema_3(text: str, original: dict[str, Any]) -> str:
    """Rewrite a schema 2 file's text as schema 3: its ``[inference]``
    settings move, with the comments above each, into the serving steps."""
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if re.match(r"^schema\s*=\s*2\s*(#.*)?$", line.strip()):
            lines[number] = "schema = 3"
            break
    start = next(
        (
            number
            for number, line in enumerate(lines)
            if (match := _HEADER.match(line.strip()))
            and match["open"] == "["
            and match["path"] == "inference"
        ),
        None,
    )
    moved: dict[str, list[str]] = {}
    heading: list[str] = []
    if start is not None:
        end = next(
            (
                n
                for n in range(start + 1, len(lines))
                if _HEADER.match(lines[n].strip())
            ),
            len(lines),
        )
        while end - 1 > start and lines[end - 1].strip().startswith("#"):
            end -= 1  # Comments just above the next table belong to it
        above = start
        while above > 0 and lines[above - 1].strip().startswith("#"):
            above -= 1
        heading = lines[above:start]
        pending: list[str] = []
        for line in lines[start + 1 : end]:
            stripped = line.strip()
            if stripped.startswith("#"):
                pending.append(line)
            elif match := _KEY.match(stripped):
                moved[match["key"]] = [*pending, line]
                pending = []
        heading += pending  # comments left at the end of the table
        lines = lines[:above] + lines[end:]
    converted = convert_schema_2(
        convert_schema_1(original) if is_schema_1(original) else original
    )
    serving = converted["serving"]
    output = [
        "",
        *heading,
        "# How an answer is made: the serving steps, in order.",
        "# Added by prokopton-ml migrate-config.",
        "[serving]",
        "# POST /identify, as before; the description is shown in the API docs",
        f"endpoint = {_quote(serving['endpoint'])}",
        f"description = {_quote(serving['description'])}",
        "chain = [" + ", ".join(_quote(name) for name in serving["chain"]) + "]",
    ]
    for name, step in serving["steps"].items():
        output += ["", f"[serving.steps.{name}]", f"kind = {_quote(step['kind'])}"]
        output.append(f"model = {_quote(step['model'])}")
        role = "specialist" if step["kind"] == "bio.specialist" else name
        for key, entry in moved.items():
            if _STEP_OF_KEY.get(key) == role:
                output += entry
    result = "\n".join(lines).rstrip("\n") + "\n" + "\n".join(output) + "\n"
    return re.sub(r"\n{3,}", "\n\n", result)


def migrate_text(text: str) -> str:
    """Rewrite a schema 1 or 2 config file's text as schema 3.

    Raises:
        MigrationError: If the file is already schema 3, or if the rewritten
            file would not give exactly the same settings.
    """
    try:
        original = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        msg = f"Not a valid TOML file: {error}"
        raise MigrationError(msg) from error
    if schema_of(original) >= CURRENT_SCHEMA:
        msg = f"This config is already schema {CURRENT_SCHEMA}"
        raise MigrationError(msg)
    try:
        before = AppConfig.model_validate(original)
    except ValueError as error:
        msg = f"Not a valid config: {error}"
        raise MigrationError(msg) from error
    if is_schema_1(original):
        text = _to_schema_2(text, original)
    result = _to_schema_3(text, original)
    try:
        after = AppConfig.model_validate(tomllib.loads(result))
    except (tomllib.TOMLDecodeError, ValueError) as error:
        msg = f"The rewritten config is not valid: {error}"
        raise MigrationError(msg) from error
    if before != after:
        msg = "The rewritten config would not give the same settings"
        raise MigrationError(msg)
    return result
