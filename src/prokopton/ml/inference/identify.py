"""Identify a submission with an app's serving steps (see ``chain``).

``Identifier`` builds the app's serving chain from its loaded models. Every
serving step kind and model kind comes from an installed package's entry
points (the ``prokopton.ml.serving_steps`` and ``prokopton.ml.model_kinds``
groups), including the platform's own kinds for DNA sequences in
``prokopton.ml.bio``.
"""

from collections.abc import Mapping
from typing import Any

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
    "Identifier",
    "Result",
    "Step",
    "model_kinds",
    "serving_kinds",
]


def model_kinds() -> ModelKinds:
    """Every model kind available, from installed packages' entry points (the
    platform's own ``bio`` kinds among them)."""
    return ModelKinds(model_entry_point_kinds())


def serving_kinds() -> ServingKinds:
    """Every serving step kind available, from installed packages' entry
    points (the platform's own ``bio`` kinds among them)."""
    return ServingKinds(serving_entry_point_kinds())


class Identifier(Chain):
    """An app's serving chain, with its models (already loaded, by name)."""

    def __init__(self, config: AppConfig, models: Mapping[str, Any]) -> None:
        self.models = dict(models)
        super().__init__(config, serving_kinds(), models)
