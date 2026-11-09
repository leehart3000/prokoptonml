"""List the phyla in an ingested MIDORI2 dataset, with record counts.

Usage:
    pixi run python analysis/01_list_phyla.py <dataset_dir> <output.tsv>

Example:
    pixi run python analysis/01_list_phyla.py \\
        data/interim/dnabarcode/midori2_gb273 data/analysis/phyla.tsv
"""

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


def count_by_phylum(table: pa.Table) -> dict[tuple[str | None, int | None], int]:
    grouped = table.group_by(["phylum", "phylum_taxid"]).aggregate([("gene", "count")])
    return {
        (row["phylum"], row["phylum_taxid"]): row["gene_count"]
        for row in grouped.to_pylist()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    dataset = ds.dataset(args.dataset_dir, format="parquet", partitioning="hive")
    table = dataset.to_table(columns=["gene", "phylum", "phylum_taxid"])
    all_counts = count_by_phylum(table)
    co1_counts = count_by_phylum(table.filter(pc.equal(table["gene"], "CO1")))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        out.write("phylum\tphylum_taxid\tco1_records\tall_records\n")
        for (name, taxid), total in sorted(all_counts.items(), key=lambda i: -i[1]):
            co1 = co1_counts.get((name, taxid), 0)
            out.write(f"{name or '(unknown)'}\t{taxid or ''}\t{co1}\t{total}\n")
    print(f"Wrote {len(all_counts)} phyla to {args.output}")


if __name__ == "__main__":
    main()
