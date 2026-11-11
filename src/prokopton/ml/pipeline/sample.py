"""Sample stage: draw balanced training and validation samples for a classifier.

Each classifier predicts a **label** for each sequence. For the router
(``label = "group_order"``) it is the order for animals (Metazoa), or the
group for other eukaryotes (Fungi, Viridiplantae, Other eukaryotes); animal
records without an order, and records with no known group, are left out. For
the marker classifier (``label = "gene"``) it is the gene a record came from.
Records are drawn from the splits of each of the classifier's ``genes``.

Classes are very unbalanced, so each is capped (``train_cap`` for training,
``eval_cap`` for validation). Within a class, records are chosen to cover as
many species (or orders) as possible: first one record from every species,
then a second, and so on, with ties broken by a salted hash so the choice is
reproducible. Classes with fewer than ``min_class_records`` eligible training
records are dropped, from validation too, and reported.

The data is streamed twice per split (to choose, then to write), so memory
stays small. Outputs ``<samples_dir>/midori2_<release>/<model>/<split>.parquet``
and ``_sample_report.json``.
"""

import logging
from array import array
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, ClassifierSampleConfig
from prokopton.ml.pipeline.split import hash_fraction

__all__ = [
    "REPORT_NAME",
    "SAMPLE_SCHEMA",
    "SampleReport",
    "SampleStageError",
    "SplitSampleReport",
    "label_for",
    "sample",
    "select_capped",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_sample_report.json"
BATCH_SIZE = 100_000
LABEL_GROUPS = ("Fungi", "Viridiplantae", "Other eukaryotes")
_COLUMNS = [
    "record_id",
    "split",
    "group",
    "order",
    "family",
    "genus",
    "species",
    "species_named",
    "length",
    "ambiguous_fraction",
    "sequence",
]

SAMPLE_SCHEMA = pa.schema(
    [
        ("record_id", pa.string()),
        ("split", pa.string()),
        ("label", pa.string()),
        ("group", pa.string()),
        ("order", pa.string()),
        ("family", pa.string()),
        ("genus", pa.string()),
        ("species", pa.string()),
        ("species_named", pa.bool_()),
        ("length", pa.int32()),
        ("ambiguous_fraction", pa.float64()),
        ("sequence", pa.string()),
    ]
)


class SampleStageError(RuntimeError):
    """Raised when the sample stage cannot run."""


def label_for(group: str | None, order: str | None) -> str | None:
    """Return the router label: the order for animals, the group otherwise."""
    if group == "Metazoa":
        return order
    if group in LABEL_GROUPS:
        return group
    return None


def select_capped(
    classes: np.ndarray,
    species: np.ndarray,
    priority: np.ndarray,
    cap: int,
    outer: np.ndarray | None = None,
) -> np.ndarray:
    """Choose up to ``cap`` records per class, covering as many species as possible.

    Args:
        classes: Class id of each record.
        species: Species id of each record (or another diversity key, such
            as the order).
        priority: Tie-breaking number for each record (lower is chosen first).
        cap: Maximum records per class.
        outer: Optional broader key (such as the group of organisms): if
            given, each class first takes turns between these, and only
            within each between species.

    Returns:
        A boolean mask of the chosen records.
    """
    count = classes.size
    if count == 0:
        return np.zeros(0, dtype=bool)
    rank = _rank_within(classes, species, np.zeros(count, dtype=np.int64), priority)
    if outer is not None:
        # Within each (class, outer) group, take turns between species
        rank = _rank_within(classes, outer, rank, priority)
    # Within each class, take rank 0 first (of every species or group), then 1
    by_class = np.lexsort((priority, rank, classes))
    sorted_classes = classes[by_class]
    new_class = np.ones(count, dtype=bool)
    new_class[1:] = sorted_classes[1:] != sorted_classes[:-1]
    class_start = np.maximum.accumulate(np.where(new_class, np.arange(count), 0))
    chosen = np.zeros(count, dtype=bool)
    chosen[by_class[(np.arange(count) - class_start) < cap]] = True
    return chosen


def _rank_within(
    classes: np.ndarray, keys: np.ndarray, rank: np.ndarray, priority: np.ndarray
) -> np.ndarray:
    """Number each record within its (class, key) group, by (rank, priority)."""
    count = classes.size
    order = np.lexsort((priority, rank, keys, classes))
    sorted_classes, sorted_keys = classes[order], keys[order]
    new_group = np.ones(count, dtype=bool)
    new_group[1:] = (sorted_classes[1:] != sorted_classes[:-1]) | (
        sorted_keys[1:] != sorted_keys[:-1]
    )
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(count), 0))
    found = np.empty(count, dtype=np.int64)
    found[order] = np.arange(count) - group_start
    return found


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SplitSampleReport(_Report):
    """Sample sizes for one split."""

    eligible_records: int
    sampled_records: int
    classes: int
    classes_below_cap: int
    mean_species_per_class: float
    output: str


class SampleReport(_Report):
    """Provenance and sizes of a sampling run."""

    app: str
    model: str = "router"
    release: str
    # The genes sampled, comma-separated
    gene: str
    prokopton_ml_version: str
    rules: ClassifierSampleConfig
    started_at: datetime
    finished_at: datetime
    labels: list[str]
    dropped_classes: dict[str, int]
    excluded_near_duplicates: int
    splits: dict[str, SplitSampleReport]


def _eligible_rows(
    inputs: list[tuple[str, Path]],
    split: str,
    rules: ClassifierSampleConfig,
    excluded: set[str],
    label_kind: str,
) -> Iterator[dict[str, Any]]:
    """Yield the rows of ``split`` that can be sampled, with their label.

    Each row also gets a ``diversity`` key: its species, or its order (its
    group if it has none), depending on ``rules.diversity``.
    """
    for gene, path in inputs:
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=BATCH_SIZE, columns=_COLUMNS
        ):
            for row in batch.to_pylist():
                if row["split"] != split or row["record_id"] in excluded:
                    continue
                if row["ambiguous_fraction"] > rules.max_ambiguous_fraction:
                    continue
                label = (
                    gene
                    if label_kind == "gene"
                    else label_for(row["group"], row["order"])
                )
                if label is None:
                    continue
                row["label"] = label
                row["diversity"] = (
                    row["species"]
                    if rules.diversity == "species"
                    else row["order"] or row["group"]
                ) or ""
                row["outer"] = row["group"] or ""
                yield row


def _sample_split(
    inputs: list[tuple[str, Path]],
    label_kind: str,
    split: str,
    rules: ClassifierSampleConfig,
    cap: int,
    excluded: set[str],
    keep_label: Callable[[str, int], bool],
    output: Path,
) -> tuple[SplitSampleReport, dict[str, int]]:
    """Choose and write one split's sample.

    Args:
        keep_label: Given a label and its number of eligible records, whether
            to keep that class.

    Returns:
        The report, and the number of eligible records for every label.
    """
    label_ids: dict[str, int] = {}
    species_ids: dict[str, int] = {}
    classes = array("q")
    species = array("q")
    outer = array("q")
    outer_ids: dict[str, int] = {}
    priority = array("d")
    for row in _eligible_rows(inputs, split, rules, excluded, label_kind):
        classes.append(label_ids.setdefault(row["label"], len(label_ids)))
        species.append(species_ids.setdefault(row["diversity"], len(species_ids)))
        outer.append(outer_ids.setdefault(row["outer"], len(outer_ids)))
        priority.append(hash_fraction(row["record_id"], rules.salt))
    labels_by_id = dict(enumerate(label_ids))
    class_array = np.frombuffer(classes, dtype=np.int64)
    eligible = {label: 0 for label in label_ids}
    for class_id, total in zip(
        *np.unique(class_array, return_counts=True), strict=True
    ):
        eligible[labels_by_id[int(class_id)]] = int(total)
    species_ids.clear()

    keep_class = np.array(
        [keep_label(labels_by_id[i], eligible[labels_by_id[i]]) for i in labels_by_id],
        dtype=bool,
    )
    chosen = select_capped(
        class_array,
        np.frombuffer(species, dtype=np.int64),
        np.frombuffer(priority, dtype=np.float64),
        cap,
        np.frombuffer(outer, dtype=np.int64)
        if rules.diversity == "group_order"
        else None,
    )
    if keep_class.size:
        chosen &= keep_class[class_array]

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f"{output.name}.part")
    sampled: Counter[str] = Counter()
    species_seen: dict[str, set[str]] = {}
    with pq.ParquetWriter(partial, SAMPLE_SCHEMA, compression="zstd") as writer:
        rows: list[dict[str, Any]] = []
        rows_again = _eligible_rows(inputs, split, rules, excluded, label_kind)
        for position, row in enumerate(rows_again):
            if not chosen[position]:
                continue
            rows.append({name: row[name] for name in SAMPLE_SCHEMA.names})
            sampled[row["label"]] += 1
            species_seen.setdefault(row["label"], set()).add(row["species"] or "")
            if len(rows) >= BATCH_SIZE:
                writer.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA))
                rows.clear()
        writer.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA))
    partial.replace(output)

    report = SplitSampleReport(
        eligible_records=class_array.size,
        sampled_records=sum(sampled.values()),
        classes=len(sampled),
        classes_below_cap=sum(1 for count in sampled.values() if count < cap),
        mean_species_per_class=round(
            sum(len(names) for names in species_seen.values()) / len(species_seen), 1
        )
        if species_seen
        else 0.0,
        output=str(output),
    )
    return report, eligible


def sample(config: AppConfig, model: str = "router") -> SampleReport:
    """Draw a classifier's training and validation samples and write them.

    Args:
        config: The app's configuration.
        model: Which classifier to sample for (``router`` or ``marker``).

    Raises:
        SampleStageError: If the split (or, when needed, leakage) data is missing.
    """
    started_at = datetime.now(UTC)
    classifier = config.classifier(model)
    rules = classifier.sample
    folder = f"midori2_{config.data.source.release.lower()}"
    inputs = [
        (
            gene,
            config.data.paths.splits_dir / folder / f"gene={gene}" / "records.parquet",
        )
        for gene in classifier.genes
    ]
    for _gene, split_path in inputs:
        if not split_path.exists():
            msg = f"{split_path} not found; run the split stage first"
            raise SampleStageError(msg)

    excluded: set[str] = set()
    if rules.exclude_near_duplicates:
        # Near-copies are only searched for in the genes the leakage stage covers
        for gene in [g for g in classifier.genes if g in config.leakage.genes]:
            leakage_path = (
                config.data.paths.leakage_dir
                / folder
                / f"gene={gene}"
                / "near_duplicates.parquet"
            )
            if not leakage_path.exists():
                msg = f"{leakage_path} not found; run the leakage stage first"
                raise SampleStageError(msg)
            excluded |= set(
                pq.read_table(leakage_path, columns=["record_id"])
                .column("record_id")
                .to_pylist()
            )

    output_root = config.data.paths.samples_dir / folder / model
    splits: dict[str, SplitSampleReport] = {}
    splits["train"], train_eligible = _sample_split(
        inputs,
        classifier.label,
        "train",
        rules,
        rules.train_cap,
        excluded,
        lambda _label, count: count >= rules.min_class_records,
        output_root / "train.parquet",
    )
    labels = sorted(
        label
        for label, count in train_eligible.items()
        if count >= rules.min_class_records
    )
    dropped = {
        label: count
        for label, count in sorted(train_eligible.items())
        if count < rules.min_class_records
    }
    known = set(labels)
    for split_name in rules.eval_splits:
        splits[split_name], _ = _sample_split(
            inputs,
            classifier.label,
            split_name,
            rules,
            rules.eval_cap,
            excluded,
            lambda label, _count: label in known,
            output_root / f"{split_name}.parquet",
        )
    for split_name, report in splits.items():
        logger.info(
            "Sampled %s: %d records in %d classes -> %s",
            split_name,
            report.sampled_records,
            report.classes,
            report.output,
        )

    result = SampleReport(
        app=config.name,
        model=model,
        release=config.data.source.release,
        gene=",".join(classifier.genes),
        prokopton_ml_version=__version__,
        rules=rules,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        labels=labels,
        dropped_classes=dropped,
        excluded_near_duplicates=len(excluded),
        splits=splits,
    )
    report_path = output_root / REPORT_NAME
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return result
