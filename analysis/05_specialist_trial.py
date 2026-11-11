"""Specialist trial: how does it behave with its starting thresholds?

Runs a specialist on a random sample of validation records from its gene:
records of known species in the group (``val_known``), of held-out species
(``val_novel``), and of the background group outside the target (for
example, other flies). Prints how often each outcome happens, how often the
answer is right, and how long identification takes. This guides tuning
before a full evaluation.

Usage:
    pixi run python analysis/05_specialist_trial.py apps/dnabarcode/dnabarcode.toml \\
        culicidae [--per-set 300]
"""

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from prokopton.core.config import load_config

from prokopton.ml.cli import ENV_PREFIX
from prokopton.ml.config import AppConfig
from prokopton.ml.pipeline.reference import reference_path
from prokopton.ml.specialists.identify import Identification, Specialist

COLUMNS = ["split", "order", "family", "genus", "species", "species_named", "sequence"]


def sample_sets(
    config: AppConfig, name: str, per_set: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    rules = config.specialist(name)
    folder = f"midori2_{config.data.source.release.lower()}"
    path = (
        config.data.paths.splits_dir / folder / f"gene={rules.gene}" / "records.parquet"
    )
    rng = random.Random(seed)
    chosen: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=COLUMNS):
        for row in batch.to_pylist():
            if row["split"] not in ("val_known", "val_novel"):
                continue
            if row[rules.target_rank] == rules.target:
                if not row["species_named"]:
                    continue
                name_of_set = f"target {row['split']}"
            elif row[rules.background_rank] == rules.background:
                name_of_set = "background"
            else:
                continue
            seen[name_of_set] += 1
            if len(chosen[name_of_set]) < per_set:
                chosen[name_of_set].append(row)
            elif (slot := rng.randrange(seen[name_of_set])) < per_set:
                chosen[name_of_set][slot] = row
    return chosen


NAMED = ("species", "complex", "species_group")
"""Outcomes that name species with confidence."""


def correct(found: Identification, row: dict[str, Any], set_name: str) -> bool:
    if set_name == "background":
        return found.outcome == "not_member"
    if found.outcome in ("species", "closest_species"):
        return found.label == row["species"]
    if found.outcome in ("complex", "species_group"):
        return row["species"] in found.species
    if found.outcome == "genus":
        return found.label == row["genus"]
    if found.outcome == "genus_label_only":
        # No genus on the record means the answer was at family level
        return found.label in (None, row["genus"])
    return found.outcome == "family"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("specialist")
    parser.add_argument("--per-set", type=int, default=300)
    parser.add_argument("--seed", type=int, default=273)
    args = parser.parse_args()

    config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
    rules = config.specialist(args.specialist)
    started = time.perf_counter()
    specialist = Specialist.from_parquet(reference_path(config, args.specialist), rules)
    print(
        f"Loaded {len(specialist.references):,} references in "
        f"{time.perf_counter() - started:.1f} s"
    )
    sets = sample_sets(config, args.specialist, args.per_set, args.seed)
    for set_name, rows in sorted(sets.items()):
        started = time.perf_counter()
        found = specialist.identify_many([row["sequence"] for row in rows])
        seconds = time.perf_counter() - started
        outcomes = Counter(f.outcome for f in found)
        right = sum(
            correct(f, row, set_name) for f, row in zip(found, rows, strict=True)
        )
        each = seconds / len(rows) * 1000
        print(f"\n## {set_name}: {len(rows)} records, {each:.0f} ms each")
        print(f"Right (at the level given): {right / len(rows):.1%}")
        if set_name != "background":
            wrongly_named = sum(
                f.outcome in NAMED and not correct(f, row, set_name)
                for f, row in zip(found, rows, strict=True)
            )
            print(
                f"Wrong species named with confidence: {wrongly_named / len(rows):.1%}"
            )
        for outcome, count in outcomes.most_common():
            print(f"  {outcome:22} {count:5}  ({count / len(rows):.1%})")
        wrong = [
            (row["species"], f.outcome, f.label, f"{f.identity:.3f}")
            for f, row in zip(found, rows, strict=True)
            if not correct(f, row, set_name)
        ][:8]
        if wrong:
            print(
                "  Examples of wrong answers (true species, outcome, label, identity):"
            )
            for example in wrong:
                print(f"    {example}")


if __name__ == "__main__":
    main()
