"""Marker classifier errors: which groups and genes, and which strand?

The marker classifier's evaluation showed lower accuracy for fungi, and that
the more confident strand is the right one only about 98% of the time for
whole sequences. This script looks closer, on a sample of ``test_known``
records (up to ``--per-cell`` records for each gene and group, chosen at
random with a fixed seed):

1. accuracy for each gene within each group of organisms;
2. for each group other than animals, which genes are mistaken for which;
3. when the reverse complement is the more confident strand, whether it is
   predicted as the record's own gene (suggesting the record is stored as
   the other strand) or as another gene (a model error).

It loads the registered marker model (``@candidate``) from MLflow.

Usage:
    pixi run python analysis/04_marker_errors.py apps/dnabarcode/dnabarcode.toml \\
        data/analysis/marker_errors [--per-cell 3000]
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from prokopton.core.config import load_config

from prokopton.ml.cli import ENV_PREFIX
from prokopton.ml.config import AppConfig
from prokopton.ml.features.fragments import reverse_complement
from prokopton.ml.training.router import predict_proba
from prokopton.ml.training.tracking import mlflow_tracker

COLUMNS = ["record_id", "split", "group", "sequence"]


def sample_records(
    config: AppConfig, per_cell: int, seed: int
) -> list[tuple[str, str, str]]:
    """Return (gene, group, sequence) for up to ``per_cell`` test records per cell."""
    marker = config.classifier("marker")
    folder = f"midori2_{config.data.source.release.lower()}"
    rng = random.Random(seed)
    chosen: dict[tuple[str, str], list[str]] = defaultdict(list)
    seen: Counter[tuple[str, str]] = Counter()
    for gene in marker.genes:
        path = (
            config.data.paths.splits_dir / folder / f"gene={gene}" / "records.parquet"
        )
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=100_000, columns=COLUMNS
        ):
            for row in batch.to_pylist():
                if row["split"] != "test_known":
                    continue
                cell = (gene, row["group"] or "No group")
                seen[cell] += 1
                # Reservoir sampling: every record has the same chance
                if len(chosen[cell]) < per_cell:
                    chosen[cell].append(row["sequence"])
                else:
                    slot = rng.randrange(seen[cell])
                    if slot < per_cell:
                        chosen[cell][slot] = row["sequence"]
    return [
        (gene, group, sequence)
        for (gene, group), sequences in sorted(chosen.items())
        for sequence in sequences
    ]


def table(header: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--per-cell", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=273)
    args = parser.parse_args()

    config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
    training = config.classifier("marker").training
    model, version = mlflow_tracker(config, "marker").load_model(
        training.registered_model, training.alias
    )
    records = sample_records(config, args.per_cell, args.seed)
    genes = np.array([r[0] for r in records], dtype=object)
    groups = np.array([r[1] for r in records], dtype=object)
    sequences = [r[2] for r in records]
    classes = np.array([str(c) for c in model.classes_], dtype=object)

    forward = predict_proba(model, sequences, 8)
    reverse = predict_proba(model, [reverse_complement(s) for s in sequences], 8)
    # Compare the best real class on each strand, not a "reverse" class
    real = classes != (training.reverse_label or "")
    predicted = classes[forward.argmax(axis=1)]
    reverse_predicted = classes[real][reverse[:, real].argmax(axis=1)]
    reverse_wins = reverse[:, real].max(axis=1) > forward[:, real].max(axis=1)
    correct = predicted == genes

    out = [f"# Marker classifier errors (dnabarcode-marker v{version})", ""]
    out += [
        f"{len(records):,} test records, up to {args.per_cell:,} per gene and group.",
        "",
    ]

    group_names = sorted(set(groups.tolist()), key=lambda g: -int((groups == g).sum()))
    out += ["## Accuracy by gene and group (forward strand)", ""]
    rows = []
    for gene in sorted(set(genes.tolist())):
        row: list[Any] = [gene]
        for group in group_names:
            mask = (genes == gene) & (groups == group)
            row.append(
                f"{correct[mask].mean():.1%} ({mask.sum():,})" if mask.any() else ""
            )
        rows.append(row)
    out += [table(["Gene", *group_names], rows), ""]

    for group in [g for g in group_names if g != "Metazoa"]:
        mask = (groups == group) & ~correct
        pairs = Counter(
            zip(genes[mask].tolist(), predicted[mask].tolist(), strict=True)
        )
        out += [f"## {group}: most common mistakes", ""]
        out += [
            table(
                ["True gene", "Predicted as", "Records"],
                [[a, b, n] for (a, b), n in pairs.most_common(15)],
            ),
            "",
        ]

    out += ["## When the other strand is more confident", ""]
    rows = []
    for gene in sorted(set(genes.tolist())):
        mask = genes == gene
        wins = mask & reverse_wins
        same = wins & (reverse_predicted == genes)
        rows.append(
            [
                gene,
                f"{reverse_wins[mask].mean():.1%}",
                f"{same.sum() / max(wins.sum(), 1):.0%}",
                f"{np.median(reverse[:, real].max(axis=1)[wins]):.2f}"
                if wins.any()
                else "",
            ]
        )
    out += [
        "Other strand more confident: share of records. Of those, predicted as "
        "its own gene: the share where the other strand is called the record's "
        "own gene (the record is probably stored as the other strand).",
        "",
        table(
            [
                "Gene",
                "Other strand more confident",
                "Of those, predicted as its own gene",
                "Median confidence",
            ],
            rows,
        ),
        "",
    ]
    rows = [
        [group, f"{reverse_wins[groups == group].mean():.1%}"] for group in group_names
    ]
    out += [table(["Group", "Other strand more confident"], rows), ""]

    args.output.mkdir(parents=True, exist_ok=True)
    report = args.output / "report.md"
    report.write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
