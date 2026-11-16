"""Settings that packages bring: for their model kinds, serving step kinds and
app-wide sections.

An app's config (``prokopton.ml.config.AppConfig``) holds only what every app
has. A model's table (``[models.<name>]``) and a serving step's table
(``[serving.steps.<name>]``) are checked against the settings declared for
their ``kind``, and each other top-level table (such as ``[bio]``) against
the section of that name, when the config is read. Settings are declared by
installed packages, through an entry point in the ``prokopton.ml.settings``
group: a function that returns a list of ``Settings``. The platform's own
sequence settings are declared the same way (``prokopton.ml.bio.config``).
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import entry_points
from typing import Any, Literal

from pydantic import BaseModel

__all__ = [
    "ENTRY_POINT_GROUP",
    "Declared",
    "Settings",
    "declared",
    "entry_point_settings",
]

ENTRY_POINT_GROUP = "prokopton.ml.settings"


@dataclass(frozen=True)
class Settings:
    """The settings of one model kind, serving step kind or section."""

    of: Literal["model", "step", "section"]
    name: str
    """The kind (``bio.specialist``), or the section's table (``bio``)."""
    model: type[BaseModel]
    """Checks the table (with its ``kind``, for a model or a step)."""
    needs_model: str | None = None
    """For a serving step: the kind of model it uses, if it must be one."""
    schema_3_tables: Mapping[str, str] = field(default_factory=dict)
    """For a section: the schema 3 tables that move into it in schema 4, such
    as ``validation`` to ``bio.validation``."""


class Declared:
    """Every declared setting, by what it is for and its name."""

    def __init__(self, found: Iterable[Settings]) -> None:
        self.models: dict[str, Settings] = {}
        self.steps: dict[str, Settings] = {}
        self.sections: dict[str, Settings] = {}
        groups = {"model": self.models, "step": self.steps, "section": self.sections}
        for settings in found:
            group = groups[settings.of]
            if settings.name in group:
                msg = f"Settings for {settings.of} {settings.name!r} are declared twice"
                raise ValueError(msg)
            group[settings.name] = settings


def entry_point_settings() -> list[Settings]:
    """The settings declared by installed packages' entry points."""
    found: list[Settings] = []
    for point in entry_points(group=ENTRY_POINT_GROUP):
        provide: Callable[[], Any] = point.load()
        found.extend(provide())
    return found


@cache
def declared() -> Declared:
    """Every installed package's settings (read once)."""
    return Declared(entry_point_settings())
