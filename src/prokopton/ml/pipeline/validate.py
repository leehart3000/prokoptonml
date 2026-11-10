"""Validation stage: check data quality, apply rules and add derived columns.

For each gene written by the ingestion stage, this stage:

- adds ``group`` (Metazoa, Fungi, Viridiplantae, Other eukaryotes, or
  "No phylum"), ``ambiguous_fraction``, ``species_named`` (whether the species
  is a proper Latin name rather than a placeholder such as
  "Chironomidae sp. BIOUG02146_D07") and ``species_sharing`` (how many *named*
  species share the identical sequence: 0 if none is named, 1 if unique);
- removes records that break a rule, recording which rule;
- writes the kept records to
  ``<processed_dir>/midori2_<release>/gene=<gene>/records.parquet`` and a
  ``_validation_report.json`` with counts for every rule and distribution.

Rules are checked in this order, and each removed record is counted once,
under the first rule it breaks: invalid characters, too short, too long, too
ambiguous, order conflict (identical sequences labelled with different orders;
all copies are removed). Species-level sharing is kept, because close relatives
can genuinely share COI haplotypes.
"""

import logging
import math
from datetime import UTC, datetime
from itertools import pairwise

import pandas as pd
from pydantic import BaseModel, ConfigDict

from prokopton.ml import __version__
from prokopton.ml.config import AppConfig, ValidationConfig
from prokopton.ml.data.taxonomy import NO_PHYLUM, UNMAPPED, load_phylum_groups

__all__ = [
    "REASONS",
    "REPORT_NAME",
    "DataValidationError",
    "GeneValidationReport",
    "SharedSequenceExample",
    "ValidationReport",
    "validate",
    "validate_gene",
]

logger = logging.getLogger(__name__)

REPORT_NAME = "_validation_report.json"
REASONS = (
    "invalid_characters",
    "too_short",
    "too_long",
    "too_ambiguous",
    "order_conflict",
)
"""Rules, in the order they are checked."""

_INVALID = r"[^ACGTRYSWKMBDHVN]"  # anything that is not an IUPAC nucleotide code
_AMBIGUOUS = r"[^ACGT]"
# A binomial ("Anopheles gambiae"), optionally with a subspecies epithet
_NAMED_SPECIES = r"^[A-Z][a-z]+ [a-z][a-z-]+(?: [a-z][a-z-]+)?$"
_MAX_EXAMPLE_SPECIES = 20
_LENGTH_EDGES = (0, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, math.inf)


class DataValidationError(RuntimeError):
    """Raised when the validation stage cannot run."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SharedSequenceExample(_Report):
    """An identical sequence found under more than one named species."""

    species_count: int
    species: list[str]
    records: int
    length: int
    accessions: list[str]


class GeneValidationReport(_Report):
    """Validation results for one gene."""

    input_records: int
    kept_records: int
    removed: dict[str, int]
    length_histogram: dict[str, int]
    ambiguity_above: dict[str, int]
    groups: dict[str, int]
    named_species_records: int
    named_species: int
    order_conflict_sequences: int
    order_conflict_records: int
    order_conflict_short_records: int
    shared_species_sequences: int
    shared_species_records: int
    top_shared_examples: list[SharedSequenceExample]
    output: str | None = None


class ValidationReport(_Report):
    """Provenance and results of a validation run."""

    app: str
    release: str
    prokopton_ml_version: str
    rules: ValidationConfig
    started_at: datetime
    finished_at: datetime
    genes: dict[str, GeneValidationReport]


def _as_bool(series: pd.Series) -> pd.Series:
    """Convert a possibly nullable boolean series to plain booleans (NA -> False)."""
    return series.fillna(value=False).astype(bool)


def _length_label(low: float, high: float) -> str:
    return f">={low:g}" if math.isinf(high) else f"{low:g}-{high - 1:g}"


def _length_histogram(lengths: pd.Series) -> dict[str, int]:
    labels = [_length_label(low, high) for low, high in pairwise(_LENGTH_EDGES)]
    binned = pd.cut(lengths, bins=list(_LENGTH_EDGES), right=False, labels=labels)
    counts = binned.value_counts(sort=False)
    return {str(label): int(counts[label]) for label in labels}


def _top_shared(shared: pd.DataFrame, limit: int) -> list[SharedSequenceExample]:
    """Summarise the sequences shared by the most species."""
    if limit == 0 or shared.empty:
        return []
    named = shared[shared["species_named"]]
    summary = named.groupby("sequence", sort=False).agg(
        species=("species", lambda s: sorted(set(s.dropna()))),
        records=("species", "size"),
        length=("length", "first"),
        accessions=("accession", lambda s: sorted(s)[:3]),
    )
    summary["n_species"] = summary["species"].map(len)
    summary = summary.sort_values(["n_species", "records"], ascending=False).head(limit)
    return [
        SharedSequenceExample(
            species_count=len(row.species),
            species=list(row.species)[:_MAX_EXAMPLE_SPECIES],
            records=int(row.records),
            length=int(row.length),
            accessions=list(row.accessions),
        )
        for row in summary.itertuples()
    ]


def validate_gene(
    records: pd.DataFrame, groups: dict[int, str], rules: ValidationConfig
) -> tuple[pd.DataFrame, GeneValidationReport]:
    """Validate one gene's records.

    Returns:
        The kept records (with derived columns added) and a report.
    """
    df = records.copy()
    group = df["phylum_taxid"].map(groups)
    group = group.where(df["phylum_taxid"].notna(), NO_PHYLUM).fillna(UNMAPPED)
    df["group"] = group.astype("string")
    df["ambiguous_fraction"] = (
        df["sequence"].str.count(_AMBIGUOUS) / df["length"].clip(lower=1)
    ).astype("float64")
    df["species_named"] = _as_bool(df["species"].str.match(_NAMED_SPECIES))
    df["named_species"] = df["species"].where(df["species_named"])
    by_sequence = df.groupby("sequence", sort=False)
    df["species_sharing"] = (
        by_sequence["named_species"].transform("nunique").astype("int64")
    )
    n_orders = by_sequence["order"].transform("nunique")
    df = df.drop(columns="named_species")

    checks = {
        "invalid_characters": _as_bool(df["sequence"].str.contains(_INVALID)),
        "too_short": _as_bool(df["length"] < rules.min_length),
        "too_long": _as_bool(df["length"] > rules.max_length),
        "too_ambiguous": df["ambiguous_fraction"] > rules.max_ambiguous_fraction,
        "order_conflict": _as_bool(n_orders > 1),
    }
    reason = pd.Series(None, index=df.index, dtype="object")
    for name in REASONS:
        reason = reason.mask(reason.isna() & checks[name], name)
    keep = reason.isna()
    kept = df[keep].reset_index(drop=True)

    conflict = checks["order_conflict"]
    shared = kept[kept["species_sharing"] > 1]
    report = GeneValidationReport(
        input_records=len(df),
        kept_records=len(kept),
        removed={name: int((reason == name).sum()) for name in REASONS},
        length_histogram=_length_histogram(df["length"]),
        ambiguity_above={
            f">{threshold:.0%}": int((df["ambiguous_fraction"] > threshold).sum())
            for threshold in rules.ambiguity_report_thresholds
        },
        groups={str(k): int(v) for k, v in kept["group"].value_counts().items()},
        named_species_records=int(kept["species_named"].sum()),
        named_species=int(kept.loc[kept["species_named"], "species"].nunique()),
        order_conflict_sequences=int(df.loc[conflict, "sequence"].nunique()),
        order_conflict_records=int(conflict.sum()),
        order_conflict_short_records=int(
            (conflict & (df["length"] < rules.short_length)).sum()
        ),
        shared_species_sequences=int(shared["sequence"].nunique()),
        shared_species_records=len(shared),
        top_shared_examples=_top_shared(shared, rules.top_shared_examples),
    )
    return kept, report


def validate(config: AppConfig) -> ValidationReport:
    """Validate every gene ingested for an application and write the results.

    Raises:
        DataValidationError: If a gene has not been ingested yet.
        ReferenceDataError: If the phylum reference table is missing or invalid.
    """
    started_at = datetime.now(UTC)
    source = config.data.source
    folder = f"midori2_{source.release.lower()}"
    input_root = config.data.paths.interim_dir / folder
    output_root = config.data.paths.processed_dir / folder
    groups = load_phylum_groups(config.validation.phylum_reference)

    genes: dict[str, GeneValidationReport] = {}
    for gene in source.genes:
        input_path = input_root / f"gene={gene}" / "records.parquet"
        if not input_path.exists():
            msg = f"{input_path} not found; run the ingest stage first"
            raise DataValidationError(msg)
        records = pd.read_parquet(input_path, dtype_backend="numpy_nullable")
        kept, report = validate_gene(records, groups, config.validation)

        output = output_root / f"gene={gene}" / "records.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f"{output.name}.part")
        kept.to_parquet(partial, index=False, compression="zstd")
        partial.replace(output)
        genes[gene] = report.model_copy(update={"output": str(output)})
        logger.info(
            "Validated %s: kept %d of %d records -> %s",
            gene,
            report.kept_records,
            report.input_records,
            output,
        )

    result = ValidationReport(
        app=config.name,
        release=source.release,
        prokopton_ml_version=__version__,
        rules=config.validation,
        started_at=started_at,
        finished_at=datetime.now(UTC),
        genes=genes,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / REPORT_NAME
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", report_path)
    return result
