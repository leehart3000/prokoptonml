"""Identify a submission with an app's serving steps (see ``chain``).

``Identifier`` builds the app's serving chain from its loaded models. The
steps and model kinds for DNA sequences are in ``prokopton.ml.bio``; other
packages can add serving step kinds and model kinds through the
``prokopton.ml.serving_steps`` and ``prokopton.ml.model_kinds`` entry point
groups.
"""

from collections.abc import Mapping
from typing import Any

from prokopton.ml.bio.models import bio_model_kinds
from prokopton.ml.bio.serving import (
    Classifier,
    Region,
    SequenceResult,
    bio_serving_kinds,
    member_name,
)
from prokopton.ml.bio.specialists.identify import Specialist
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.chain import (
    Answer,
    Chain,
    Result,
    ServingKinds,
    Step,
    serving_entry_point_kinds,
)
from prokopton.ml.inference.models import ModelKinds, model_entry_point_kinds

__all__ = [
    "Answer",
    "Classifier",
    "Identifier",
    "Region",
    "Result",
    "SequenceResult",
    "Step",
    "member_name",
    "model_kinds",
    "serving_kinds",
]


def model_kinds() -> ModelKinds:
    """Every model kind available: the platform's and installed packages'."""
    return ModelKinds([*bio_model_kinds(), *model_entry_point_kinds()])


def serving_kinds() -> ServingKinds:
    """Every serving step kind available: the platform's and installed
    packages'."""
    return ServingKinds([*bio_serving_kinds(), *serving_entry_point_kinds()])


class Identifier(Chain):
    """An app's serving chain, with its models (already loaded, by name)."""

    def __init__(self, config: AppConfig, models: Mapping[str, Any]) -> None:
        self.models = dict(models)
        super().__init__(config, serving_kinds(), models)

    @property
    def specialists(self) -> dict[str, Specialist]:
        """The loaded specialists, by name."""
        return {n: m for n, m in self.models.items() if isinstance(m, Specialist)}

    def identify(self, text: str) -> SequenceResult:
        """Identify one submission."""
        result = super().identify(text)
        assert isinstance(result, SequenceResult)  # The chain starts with an input step
        return result
