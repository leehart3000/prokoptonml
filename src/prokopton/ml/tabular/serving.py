"""Serving steps for a tabular classifier, answering one record at a time.

1. **Take the record** (``core.record-input``). The request is a JSON object
   with the model's input columns, each a number or a category, or missing
   (``null`` or left out). A record with no values at all is declined.
2. **Classify** (``core.classifier``). The answer is the most probable class:
   "Most likely …" at the step's ``threshold`` or above, "Possibly …" below
   it, with the top probabilities in the trace.
"""

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from prokopton.ml.config import AppConfig
from prokopton.ml.inference.chain import (
    FINAL,
    Answer,
    Decline,
    RequestSpec,
    ServingStepKind,
    Step,
    StepRunner,
    Work,
)
from prokopton.ml.tabular.config import (
    ClassifierStepConfig,
    RecordInputStepConfig,
    TableConfig,
    table_classifier,
)

__all__ = ["tabular_serving_kinds"]


def _table(config: AppConfig, model: str) -> TableConfig:
    return table_classifier(config, model).table


def _settings[T](config: AppConfig, name: str, kind: type[T]) -> T:
    settings = config.step_settings(name)
    assert isinstance(settings, kind)  # The step's kind chose its settings type
    return settings


def _request_model(config: AppConfig, name: str) -> type[BaseModel]:
    rules = _settings(config, name, RecordInputStepConfig)
    table = _table(config, rules.model)
    fields: dict[str, Any] = {}
    for number, column in enumerate(table.columns):
        kind: Any = float | None if column in table.numeric else str | None
        fields[f"column_{number}"] = (kind, Field(default=None, alias=column))
    title = config.serving.endpoint.replace("-", " ").title().replace(" ", "")
    examples: list[Any] = [rules.example] if rules.example else []
    request: type[BaseModel] = create_model(
        f"{title}Request",
        __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": examples}),
        __doc__=f"One record: {', '.join(table.columns)} (any may be missing).",
        **fields,
    )
    return request


def _request_text(request: BaseModel) -> str:
    return json.dumps(request.model_dump(by_alias=True))


def _build_input(config: AppConfig, name: str, models: Mapping[str, Any]) -> StepRunner:
    rules = _settings(config, name, RecordInputStepConfig)
    table = _table(config, rules.model)

    def run(work: Work) -> Decline | None:
        values = json.loads(work.text)
        record = {column: values.get(column) for column in table.columns}
        given = sum(value is not None for value in record.values())
        work.trace.add(
            Step(
                name=name,
                summary=f"One record with {given} of {len(record)} values.",
                details={"missing": [c for c, v in record.items() if v is None]},
            )
        )
        if given == 0:
            return Decline("empty_record", "The record has no values.")
        work.found["record"] = record
        return None

    return run


def _build_classifier(
    config: AppConfig, name: str, models: Mapping[str, Any]
) -> StepRunner:
    rules = _settings(config, name, ClassifierStepConfig)
    model = models[rules.model]
    table = _table(config, rules.model)

    def run(work: Work) -> Literal["final"]:
        record = work.found["record"]
        row = {c: record[c] if record[c] is not None else np.nan for c in table.columns}
        frame = pd.DataFrame([row], columns=list(table.columns))
        for column in table.numeric:
            frame[column] = frame[column].astype(float)
        probabilities = model.predict_proba(frame)[0]
        classes = [str(c) for c in model.classes_]
        order = np.argsort(probabilities)[::-1]
        top = [(classes[i], float(probabilities[i])) for i in order[:3]]
        label, probability = top[0]
        confident = probability >= rules.threshold
        work.trace.add(
            Step(
                name=name,
                summary=f"{label} (probability {probability:.2f}).",
                details={"top": [{"label": c, "probability": p} for c, p in top]},
            )
        )
        work.answer = Answer(
            level="class",
            name=label,
            text=(
                f"Most likely {label} (probability {probability:.2f})."
                if confident
                else f"Possibly {label}, but the model is unsure (probability "
                f"{probability:.2f})."
            ),
            confident=confident,
        )
        return FINAL

    return run


def tabular_serving_kinds() -> list[ServingStepKind]:
    """The tabular serving step kinds (the ``prokopton.ml.serving_steps``
    entry point)."""
    return [
        ServingStepKind(
            "core.record-input",
            "Take one record of the model's columns, any of them missing.",
            _build_input,
            request=RequestSpec(_request_model, _request_text),
        ),
        ServingStepKind(
            "core.classifier",
            "Predict the record's class, with its probability.",
            _build_classifier,
        ),
    ]
