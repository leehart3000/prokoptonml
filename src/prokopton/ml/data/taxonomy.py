"""Reviewed taxonomy reference data, such as the phylum -> group table."""

import csv
from pathlib import Path

__all__ = [
    "NO_PHYLUM",
    "OTHER_EUKARYOTES",
    "UNMAPPED",
    "ReferenceDataError",
    "load_phylum_groups",
]

OTHER_EUKARYOTES = "Other eukaryotes"
"""Group for phyla with no NCBI kingdom rank (e.g. Sar, Rhodophyta)."""

NO_PHYLUM = "No phylum"
"""Group for records whose phylum is unknown."""

UNMAPPED = "Unmapped"
"""Group for records whose phylum is missing from the reference table."""


class ReferenceDataError(ValueError):
    """Raised when a reference data file is missing or invalid."""


def load_phylum_groups(path: Path) -> dict[int, str]:
    """Load ``{phylum_taxid: group}`` from a reviewed phylum lineage TSV.

    The group is the NCBI kingdom (e.g. Metazoa, Fungi, Viridiplantae) when
    there is one, otherwise ``OTHER_EUKARYOTES``. Rows without a taxid (the
    "(unknown)" row) are skipped.

    Raises:
        ReferenceDataError: If the file is missing, lacks required columns or
            contains no phyla.
    """
    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            missing = {"phylum_taxid", "kingdom"} - set(reader.fieldnames or ())
            if missing:
                msg = f"{path} is missing columns: {sorted(missing)}"
                raise ReferenceDataError(msg)
            groups = {
                int(row["phylum_taxid"]): row["kingdom"] or OTHER_EUKARYOTES
                for row in reader
                if row["phylum_taxid"]
            }
    except OSError as error:
        msg = f"Cannot read phylum reference {path}: {error}"
        raise ReferenceDataError(msg) from error
    if not groups:
        msg = f"No phyla found in {path}"
        raise ReferenceDataError(msg)
    return groups
