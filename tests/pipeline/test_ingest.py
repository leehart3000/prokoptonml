"""Tests for prokopton.ml.pipeline.ingest, using a tiny fake MIDORI2 release."""

import gzip
import hashlib
import json
import zipfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from prokopton.ml.cli import main
from prokopton.ml.config import AppConfig
from prokopton.ml.data.checksums import ChecksumMismatchError
from prokopton.ml.pipeline.ingest import REPORT_NAME, IngestError, ingest

TEMPLATE = "SINTAX_sp/uniq/MIDORI2_UNIQ_NUC_SP_GB999_{gene}_SINTAX.fasta.gz"
FASTA = {
    "CO1": (
        ">AB000001.1.1.12;tax=k:Eukaryota_2759,p:Arthropoda_6656,c:Insecta_50557,"
        "o:Diptera_7147,f:Culicidae_7157,g:Anopheles_7164,s:Anopheles gambiae_7165;\n"
        "ACGTACGT\nACGT\n"
        ">AB000002.1.<1.>9;tax=k:Eukaryota_2759,p:Chordata_7711,c:Aves_8782,"
        "o:order_family_genus_Aves sp._8783,f:family_genus_Aves sp._8783,"
        "g:genus_Aves sp._8783,s:Aves sp._8783;\n"
        "ttgaccatg\n"
    ),
    "Cytb": (
        ">AB000003.1.1.6;tax=k:Eukaryota_2759,p:Mollusca_6447,c:Bivalvia_6544,"
        "o:Ostreida_2681325,f:Ostreidae_6563,g:Ostrea_6564,s:Ostrea edulis_37623;\n"
        "GGCCAA\n"
    ),
}


def make_release(folder: Path) -> tuple[Path, Path]:
    """Write a fake archive and checksum file; return their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "fake.zip"
    sums = ["# Fake distribution checksums, with a comment like the real file\n"]
    with zipfile.ZipFile(archive, "w") as zf:
        for gene, text in FASTA.items():
            member = TEMPLATE.format(gene=gene)
            data = gzip.compress(text.encode("utf-8"))
            zf.writestr(member, data)
            sums.append(f"{hashlib.sha256(data).hexdigest()}  ./{member}\n")
    checksums = folder / "release_sha256.txt"
    checksums.write_text("".join(sums), encoding="utf-8")
    return archive, checksums


def make_config(tmp_path: Path, genes: list[str]) -> AppConfig:
    archive, checksums = make_release(tmp_path / "upstream")
    return AppConfig.model_validate(
        {
            "name": "testapp",
            "title": "Test app",
            "data": {
                "source": {
                    "release": "GB999",
                    "doi": "10.0000/example",
                    "licence": "CC-BY-4.0",
                    "archive_url": archive.as_uri(),
                    "checksums_url": checksums.as_uri(),
                    "archive_name": "fake.zip",
                    "member_template": TEMPLATE,
                    "genes": genes,
                },
                "paths": {
                    "raw_dir": str(tmp_path / "raw"),
                    "interim_dir": str(tmp_path / "interim"),
                    "processed_dir": str(tmp_path / "processed"),
                },
            },
            "validation": {"phylum_reference": str(tmp_path / "phyla.tsv")},
        }
    )


def test_ingest_writes_parquet_and_report(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["CO1", "Cytb"])

    report = ingest(config, batch_size=1)

    assert {gene: g.records for gene, g in report.genes.items()} == {
        "CO1": 2,
        "Cytb": 1,
    }
    co1 = pq.read_table(report.genes["CO1"].output).to_pylist()
    assert co1[0]["sequence"] == "ACGTACGTACGT"
    assert co1[0]["length"] == 12
    assert co1[0]["species"] == "Anopheles gambiae"
    assert co1[0]["species_taxid"] == 7165
    assert co1[1]["sequence"] == "TTGACCATG"
    assert co1[1]["order"] is None
    assert co1[1]["class"] == "Aves"
    assert (co1[1]["start"], co1[1]["end"]) == ("<1", ">9")

    report_path = tmp_path / "interim" / "midori2_gb999" / REPORT_NAME
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["release"] == "GB999"
    assert saved["genes"]["Cytb"]["records"] == 1


def test_ingest_reuses_downloaded_files(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["CO1"])
    ingest(config)
    (tmp_path / "upstream" / "fake.zip").unlink()

    report = ingest(config)

    assert report.genes["CO1"].records == 2


def test_ingest_rejects_gene_without_checksum(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["ND1"])

    with pytest.raises(IngestError, match="No published checksum"):
        ingest(config)


def test_ingest_rejects_tampered_file(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["CO1"])
    checksums = tmp_path / "upstream" / "release_sha256.txt"
    member = TEMPLATE.format(gene="CO1")
    checksums.write_text(f"{'0' * 64}  ./{member}\n", encoding="utf-8")

    with pytest.raises(ChecksumMismatchError):
        ingest(config)


def test_cli_ingest_succeeds(tmp_path: Path) -> None:
    config = make_config(tmp_path, ["CO1"])
    source = config.data.source
    config_path = tmp_path / "app.toml"
    config_path.write_text(
        f"""
name = "testapp"
title = "Test app"

[data.source]
release = "{source.release}"
doi = "{source.doi}"
licence = "{source.licence}"
archive_url = "{source.archive_url}"
checksums_url = "{source.checksums_url}"
archive_name = "{source.archive_name}"
member_template = "{source.member_template}"
genes = ["CO1"]

[data.paths]
raw_dir = "{config.data.paths.raw_dir}"
interim_dir = "{config.data.paths.interim_dir}"
processed_dir = "{config.data.paths.processed_dir}"

[validation]
phylum_reference = "{config.validation.phylum_reference}"
""",
        encoding="utf-8",
    )

    assert main(["ingest", str(config_path)]) == 0
    assert (config.data.paths.interim_dir / "midori2_gb999" / REPORT_NAME).exists()


def test_cli_returns_error_for_invalid_manifest(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "fake.zip").write_bytes(b"")
    (raw_dir / "release_sha256.txt").write_text("not a checksum\n", encoding="utf-8")
    config_path = tmp_path / "app.toml"
    config_path.write_text(
        f"""
name = "testapp"
title = "Test app"

[data.source]
release = "GB999"
doi = "10.0000/example"
licence = "CC-BY-4.0"
archive_url = "https://example.org/fake.zip"
checksums_url = "https://example.org/release_sha256.txt"
archive_name = "fake.zip"
member_template = "{TEMPLATE}"
genes = ["CO1"]

[data.paths]
raw_dir = "{raw_dir}"
interim_dir = "{tmp_path / "interim"}"
processed_dir = "{tmp_path / "processed"}"

[validation]
phylum_reference = "{tmp_path / "phyla.tsv"}"
""",
        encoding="utf-8",
    )

    assert main(["ingest", str(config_path)]) == 1


def test_cli_returns_error_for_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.toml"
    config_path.write_text('name = "testapp"\n', encoding="utf-8")

    assert main(["ingest", str(config_path)]) == 1
