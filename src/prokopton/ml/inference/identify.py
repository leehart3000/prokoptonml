"""Identify a submission with an app's serving steps (see ``chain``).

``Identifier`` builds the app's serving chain from its loaded models. The
steps for DNA sequences are in ``prokopton.ml.bio.serving``; other packages
can add serving step kinds through the ``prokopton.ml.serving_steps`` entry
point group.
"""

from collections.abc import Mapping
from typing import Any

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

__all__ = [
    "Answer",
    "Classifier",
    "Identifier",
    "Region",
    "Result",
    "SequenceResult",
    "Step",
    "member_name",
    "serving_kinds",
]


def serving_kinds() -> ServingKinds:
    """Every serving step kind available: the platform's and installed
    packages'."""
    return ServingKinds([*bio_serving_kinds(), *serving_entry_point_kinds()])


class Identifier(Chain):
    """An app's serving chain, with its marker classifier, router and
    specialists (already loaded)."""

    def __init__(
        self,
        config: AppConfig,
        marker: Classifier,
        router: Classifier,
        specialists: Mapping[str, Specialist],
    ) -> None:
        self.marker = marker
        self.router = router
        self.specialists = dict(specialists)
        models: dict[str, Any] = {"marker": marker, "router": router, **specialists}
        super().__init__(config, serving_kinds(), models)

    def identify(self, text: str) -> SequenceResult:
        """Identify one submission."""
        result = super().identify(text)
        assert isinstance(result, SequenceResult)  # The chain starts with an input step
        return result
