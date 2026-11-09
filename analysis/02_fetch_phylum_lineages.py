"""Fetch each phylum's full lineage from NCBI Taxonomy, to map phyla to kingdoms.

Reads the TSV written by ``01_list_phyla.py`` and adds three columns:
``kingdom`` (NCBI's kingdom-rank taxon, e.g. Metazoa, Fungi, Viridiplantae,
or empty if the phylum has none), ``major_group`` (the first group below
Eukaryota, e.g. Opisthokonta or Sar) and ``lineage`` (the full lineage).

Uses one request to NCBI E-utilities (no API key is needed at this rate).

Usage:
    pixi run python analysis/02_fetch_phylum_lineages.py <phyla.tsv> <output.tsv>
"""

import argparse
import csv
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def fetch_lineages(taxids: list[str]) -> dict[str, list[tuple[str, str]]]:
    """Return ``{taxid: [(name, rank), ...]}`` from the root down to the phylum."""
    query = urllib.parse.urlencode(
        {
            "db": "taxonomy",
            "id": ",".join(taxids),
            "retmode": "xml",
            "tool": "prokopton-ml",
        }
    )
    with urllib.request.urlopen(f"{EFETCH_URL}?{query}", timeout=60) as response:
        root = ET.parse(response).getroot()

    lineages: dict[str, list[tuple[str, str]]] = {}
    for taxon in root.findall("Taxon"):
        taxid = taxon.findtext("TaxId", default="")
        nodes = [
            (node.findtext("ScientificName", ""), node.findtext("Rank", ""))
            for node in taxon.findall("LineageEx/Taxon")
        ]
        nodes.append((taxon.findtext("ScientificName", ""), taxon.findtext("Rank", "")))
        lineages[taxid] = nodes
    return lineages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("phyla", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.phyla.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    lineages = fetch_lineages(
        [row["phylum_taxid"] for row in rows if row["phylum_taxid"]]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as out:
        fields = [*rows[0].keys(), "kingdom", "major_group", "lineage"]
        writer = csv.DictWriter(out, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            nodes = lineages.get(row["phylum_taxid"], [])
            names = [name for name, _ in nodes]
            kingdom = next((name for name, rank in nodes if rank == "kingdom"), "")
            major = ""
            if "Eukaryota" in names and names.index("Eukaryota") + 1 < len(names):
                major = names[names.index("Eukaryota") + 1]
            writer.writerow(
                {
                    **row,
                    "kingdom": kingdom,
                    "major_group": major,
                    "lineage": "; ".join(names),
                }
            )
    missing = [row["phylum"] for row in rows if row["phylum_taxid"] not in lineages]
    print(f"Wrote {len(rows)} rows to {args.output}; no lineage for: {missing}")


if __name__ == "__main__":
    main()
