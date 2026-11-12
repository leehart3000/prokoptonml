"""Mislabel candidates: which reference records look like another species?

Lists named records of a specialist's serving reference set that are nearly
identical to a record of another named species (k-mer similarity at least
``--other``), but unlike most records of their own species (median
similarity at most ``--own``). For each pair of species it also gives how
similar the two species usually are (the median over up to 30 records of
each): a low value means the species are clearly different in general, so
a near-identical record is probably mislabelled; a high value means they
share sequences (sister or cryptic species), so it probably is not.

The output is for review by a person; records judged to be errors go into
the app config's ``excluded`` list, with a reason.

Usage:
    pixi run python analysis/06_mislabel_candidates.py \\
        apps/dnabarcode/dnabarcode.toml culicidae out.tsv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from prokopton.core.config import load_config

from prokopton.ml.cli import ENV_PREFIX
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.reference import serving_reference_path
from prokopton.ml.specialists.identify import QUERY_BATCH, Specialist

SAMPLE = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("specialist")
    parser.add_argument("output", type=Path)
    parser.add_argument("--other", type=float, default=0.95)
    parser.add_argument("--own", type=float, default=0.6)
    args = parser.parse_args()
    config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
    rules = config.specialist(args.specialist)
    specialist = Specialist.from_parquet(
        serving_reference_path(config, args.specialist), rules
    )
    references = specialist.references
    species = np.array([r.species or "" for r in references], dtype=object)
    named = np.array(
        [r.role == "target" and r.species_named and bool(r.species) for r in references]
    )
    members: dict[str, list[int]] = defaultdict(list)
    for number in np.flatnonzero(named).tolist():
        members[species[number]].append(number)
    columns = specialist.index.T.tocsr()

    found = []
    checked = [
        n for n in np.flatnonzero(named).tolist() if len(members[species[n]]) > 5
    ]
    for start in range(0, len(checked), QUERY_BATCH):
        numbers = checked[start : start + QUERY_BATCH]
        similarity = np.asarray((columns[numbers] @ specialist.index).todense())
        for row, number in enumerate(numbers):
            same = species == species[number]
            same[number] = False
            own = float(np.median(similarity[row, same]))
            if own > args.own:
                continue
            other = named & (species != species[number])
            masked = np.where(other, similarity[row], -1.0)
            best = int(np.argmax(masked))
            if masked[best] >= args.other:
                found.append((number, best, float(masked[best]), own))

    def pair_median(a: str, b: str) -> float:
        left, right = members[a][:SAMPLE], members[b][:SAMPLE]
        cross = (columns[left] @ specialist.index[:, right]).toarray()
        return float(np.median(cross))

    pairs: dict[tuple[str, str], list[tuple[int, float, float]]] = defaultdict(list)
    for number, best, similar, own in found:
        pairs[str(species[number]), str(species[best])].append((number, similar, own))
    typical = {pair: pair_median(*pair) for pair in pairs}

    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "species",
                "looks_like",
                "pair_median",
                "record_id",
                "similarity",
                "own_median",
            ]
        )
        for pair in sorted(pairs, key=lambda p: typical[p]):
            for number, similar, own in pairs[pair]:
                writer.writerow(
                    [
                        *pair,
                        f"{typical[pair]:.3f}",
                        references[number].record_id,
                        f"{similar:.3f}",
                        f"{own:.3f}",
                    ]
                )
    print(f"{len(found)} candidates in {len(pairs)} species pairs -> {args.output}")
    print("Pairs, most clearly different species first (pair median, records):")
    for pair in sorted(pairs, key=lambda p: typical[p])[:40]:
        print(f"  {pair[0]} -> {pair[1]}: {typical[pair]:.2f}, {len(pairs[pair])}")


if __name__ == "__main__":
    main()
