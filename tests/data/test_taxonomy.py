"""Tests for prokopton.ml.data.taxonomy."""

from pathlib import Path

import pytest

from prokopton.ml.data.taxonomy import (
    OTHER_EUKARYOTES,
    ReferenceDataError,
    load_phylum_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HEADER = "phylum\tphylum_taxid\tco1_records\tall_records\tkingdom\tmajor_group\n"


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "phyla.tsv"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_groups_with_other_eukaryotes_fallback(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        HEADER
        + "Arthropoda\t6656\t1\t1\tMetazoa\tOpisthokonta\n"
        + "Rhodophyta\t2763\t1\t1\t\tRhodophyta\n"
        + "(unknown)\t\t1\t1\t\t\n",
    )

    assert load_phylum_groups(path) == {6656: "Metazoa", 2763: OTHER_EUKARYOTES}


def test_real_reference_table_loads() -> None:
    groups = load_phylum_groups(
        REPO_ROOT / "apps/dnabarcode/reference/phylum_lineages_gb273.tsv"
    )

    assert len(groups) == 68
    assert groups[6656] == "Metazoa"
    assert groups[4890] == "Fungi"
    assert groups[35493] == "Viridiplantae"
    assert groups[5794] == OTHER_EUKARYOTES


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDataError, match="Cannot read"):
        load_phylum_groups(tmp_path / "missing.tsv")


def test_missing_columns_raise(tmp_path: Path) -> None:
    path = write(tmp_path, "phylum\tphylum_taxid\nArthropoda\t6656\n")

    with pytest.raises(ReferenceDataError, match="missing columns"):
        load_phylum_groups(path)


def test_empty_table_raises(tmp_path: Path) -> None:
    with pytest.raises(ReferenceDataError, match="No phyla"):
        load_phylum_groups(write(tmp_path, HEADER))
