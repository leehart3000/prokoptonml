"""Reference matches: which references decide an answer, and do they agree
with RefSeq?

Identifies each FASTA file with a serving bundle and lists the top matches
from the specialist's reference set. For each match it also gives the
alignment identity and coverage to the RefSeq COI anchors of the species
named with ``--species``, so a mislabelled reference (close to another
species' anchor, far from its own) stands out.

Usage:
    pixi run python analysis/07_reference_matches.py \\
        apps/dnabarcode/dnabarcode.toml data/models/dnabarcode/bundle culicidae \\
        --species "Aedes aegypti" "Aedes albopictus" \\
        --fasta query1.fasta query2.fasta
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from prokopton.core.config import load_config

from prokopton.ml.cli import ENV_PREFIX
from prokopton.ml.config import AppConfig
from prokopton.ml.inference.bundle import load_bundle
from prokopton.ml.sources.refseq import read_anchors
from prokopton.ml.specialists.identity import align

TOP = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("specialist")
    parser.add_argument("--species", nargs="+", required=True)
    parser.add_argument("--fasta", nargs="+", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config, AppConfig, env_prefix=ENV_PREFIX)
    rules = config.specialist(args.specialist)
    if rules.anchors is None:
        parser.error(f"Specialist {args.specialist} has no RefSeq anchors")
    anchors = [a for a in read_anchors(rules.anchors) if a.species in args.species]
    print("Anchors:", [(a.accession, a.species, len(a.sequence)) for a in anchors])

    reference = args.bundle / "specialists" / args.specialist / "reference.parquet"
    sequences = {
        row["record_id"]: row["sequence"]
        for row in pq.read_table(reference).to_pylist()
    }
    identifier = load_bundle(args.bundle).identifier
    for path in args.fasta:
        result = identifier.identify(path.read_text("utf-8"))
        answer = result.answer.text if result.answer else result.reason
        print(f"\n== {path.stem}: {answer}")
        for step in result.trace:
            if step.name != f"specialist:{args.specialist}":
                continue
            for match in step.details["matches"][:TOP]:
                line = (
                    f"  {match['record_id']:<34} {match['species']!s:<26} "
                    f"{match['identity']:.2%} cov {match['coverage']:.2f} "
                    f"n {match['aligned']}"
                )
                for anchor in anchors:
                    found = align(sequences[match["record_id"]], anchor.sequence)
                    line += (
                        f" | {anchor.species} {anchor.accession} "
                        f"{found.identity:.2%} cov {found.coverage:.2f}"
                    )
                print(line)


if __name__ == "__main__":
    main()
