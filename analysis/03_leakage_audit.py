"""Leakage audit: do novel-set sequences have near-identical matches in training?

Samples sequences from one split (by default ``test_novel``), searches them
against the ``train`` split with VSEARCH, and reports how many have a
near-identical training match. Training sequences are searched in chunks, so
memory stays low; each chunk's temporary FASTA file is deleted after use.

VSEARCH comes from the bioconda channel and is run through ``pixi exec``, so
the project's own environment is not changed.

Usage:
    pixi run python analysis/03_leakage_audit.py <split_records.parquet> <output_dir> \\
        [--sample 5000] [--split test_novel] [--threads 4] \\
        [--min-identity 0.97] [--maxaccepts 8] [--maxrejects 64]

The defaults are thorough. Faster settings for "is there any >=99% match?"
are ``--min-identity 0.99 --maxaccepts 1 --maxrejects 8``.

Example (a quick trial first):
    pixi run python analysis/03_leakage_audit.py \\
        data/splits/dnabarcode/midori2_gb273/gene=CO1/records.parquet \\
        data/analysis/leakage_trial --sample 500
"""

import argparse
import csv
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

VSEARCH = [
    "pixi",
    "exec",
    "--channel",
    "conda-forge",
    "--channel",
    "bioconda",
    "--spec",
    "vsearch>=2.28,<3",
    "vsearch",
]
BATCH_SIZE = 100_000
MIN_IDENTITY = 0.97
LEAK_IDENTITY = 99.0
RANKS = ("genus", "family", "order")


def label(record_id: str) -> str:
    """Make a FASTA label without characters that could confuse tools."""
    return record_id.replace("<", "lt").replace(">", "gt").replace(" ", "_")


def sample_queries(
    path: Path, split: str, size: int, seed: int
) -> list[dict[str, Any]]:
    """Reservoir-sample ``size`` records of ``split``, streaming the file in batches."""
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    seen = 0
    columns = ["record_id", "split", "sequence", "length", *RANKS]
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=BATCH_SIZE, columns=columns
    ):
        for row in batch.to_pylist():
            if row["split"] != split:
                continue
            seen += 1
            if len(sample) < size:
                sample.append(row)
            else:
                index = rng.randrange(seen)
                if index < size:
                    sample[index] = row
    return sample


def run_vsearch(
    queries: Path, database: Path, output: Path, args: argparse.Namespace
) -> None:
    command = [
        *VSEARCH,
        "--usearch_global",
        str(queries),
        "--db",
        str(database),
        "--id",
        str(args.min_identity),
        "--maxaccepts",
        str(args.maxaccepts),
        "--maxrejects",
        str(args.maxrejects),
        "--userout",
        str(output),
        "--userfields",
        "query+target+id",
        "--threads",
        str(args.threads),
        "--quiet",
    ]
    subprocess.run(command, check=True)


def search_training_chunks(
    path: Path, queries: Path, work: Path, args: argparse.Namespace
) -> dict[str, tuple[float, str]]:
    """Return ``{query label: (best identity, target label)}`` across all chunks."""
    best: dict[str, tuple[float, str]] = {}
    chunk: list[tuple[str, str]] = []
    number = 0

    def flush() -> None:
        nonlocal number
        if not chunk:
            return
        number += 1
        database = work / f"train_chunk_{number}.fasta"
        hits = work / f"hits_{number}.tsv"
        with database.open("w", encoding="utf-8") as file:
            file.writelines(f">{name}\n{sequence}\n" for name, sequence in chunk)
        started = time.monotonic()
        run_vsearch(queries, database, hits, args)
        with hits.open(encoding="utf-8") as file:
            for query, target, identity in csv.reader(file, delimiter="\t"):
                value = float(identity)
                if query not in best or value > best[query][0]:
                    best[query] = (value, target)
        database.unlink()
        hits.unlink()
        print(
            f"Chunk {number}: {len(chunk):,} training sequences searched in "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )
        chunk.clear()

    columns = ["record_id", "split", "sequence"]
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=BATCH_SIZE, columns=columns
    ):
        for row in batch.to_pylist():
            if row["split"] == "train":
                chunk.append((label(row["record_id"]), row["sequence"]))
                if len(chunk) >= args.chunk_size:
                    flush()
    flush()
    return best


def target_taxonomy(path: Path, targets: set[str]) -> dict[str, dict[str, Any]]:
    """Look up genus, family and order for the matched training records."""
    found: dict[str, dict[str, Any]] = {}
    columns = ["record_id", *RANKS]
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=BATCH_SIZE, columns=columns
    ):
        for row in batch.to_pylist():
            name = label(row["record_id"])
            if name in targets:
                found[name] = row
    return found


def relation(query: dict[str, Any], target: dict[str, Any]) -> str:
    """Describe how closely related a query and its training match are."""
    for rank in RANKS:
        if query[rank] is not None and query[rank] == target[rank]:
            return f"same {rank}"
    if query["order"] is None or target["order"] is None:
        return "order unknown"
    return "different order"


def length_bin(length: int) -> str:
    if length < 300:
        return "<300"
    if length < 500:
        return "300-499"
    return ">=500"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("records", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--split", default="test_novel")
    parser.add_argument("--sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=273)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--min-identity", type=float, default=MIN_IDENTITY)
    parser.add_argument("--maxaccepts", type=int, default=8)
    parser.add_argument("--maxrejects", type=int, default=64)
    args = parser.parse_args()

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = sample_queries(args.records, args.split, args.sample, args.seed)
    queries = {label(row["record_id"]): row for row in sample}
    query_fasta = args.output_dir / "queries.fasta"
    with query_fasta.open("w", encoding="utf-8") as file:
        file.writelines(
            f">{name}\n{row['sequence']}\n" for name, row in queries.items()
        )
    print(f"Sampled {len(queries):,} {args.split} sequences", flush=True)

    best = search_training_chunks(args.records, query_fasta, args.output_dir, args)
    taxonomy = target_taxonomy(args.records, {target for _, target in best.values()})

    bands: Counter[str] = Counter()
    relations: Counter[str] = Counter()
    by_length: dict[str, Counter[str]] = {}
    with (args.output_dir / "matches.tsv").open("w", encoding="utf-8") as file:
        file.write("query\tlength\tbest_identity\ttarget\trelation\n")
        for name, row in queries.items():
            identity, target = best.get(name, (0.0, ""))
            band = (
                ">=99%"
                if identity >= LEAK_IDENTITY
                else "97-99%"
                if identity >= MIN_IDENTITY * 100
                else "<97% or none"
            )
            bands[band] += 1
            by_length.setdefault(length_bin(row["length"]), Counter())[band] += 1
            kind = ""
            if band == ">=99%":
                kind = relation(row, taxonomy[target])
                relations[kind] += 1
            file.write(f"{name}\t{row['length']}\t{identity}\t{target}\t{kind}\n")

    total = len(queries)
    summary = {
        "split": args.split,
        "sample": total,
        "seed": args.seed,
        "min_identity_searched": args.min_identity,
        "maxaccepts": args.maxaccepts,
        "maxrejects": args.maxrejects,
        "bands": {band: bands[band] for band in (">=99%", "97-99%", "<97% or none")},
        "share_at_least_99": round(bands[">=99%"] / total, 4) if total else None,
        "relation_of_99_matches": dict(relations.most_common()),
        "by_length": {key: dict(value) for key, value in sorted(by_length.items())},
        "seconds": round(time.monotonic() - started),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    query_fasta.unlink()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
