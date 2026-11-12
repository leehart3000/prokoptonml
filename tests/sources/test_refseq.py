"""Tests for prokopton.ml.sources.refseq."""

import io
import json
from pathlib import Path

import pytest

from prokopton.ml import cli
from prokopton.ml.config import AppConfig
from prokopton.ml.sources import refseq
from prokopton.ml.sources.refseq import binomial, fetch_anchors, read_anchors
from specialist_world import config_dict

COI_A = "ATGC" * 30
COI_B = "GGCA" * 30


def fake_fetch(url: str) -> bytes:
    if "esearch" in url:
        return json.dumps({"esearchresult": {"idlist": ["1", "2", "3", "4"]}}).encode()
    if "esummary" in url:
        names = {
            "1": ("NC_000001.1", "Aedes aegypti"),
            "2": ("NC_000002.1", "Ochlerotatus serratus"),
            "3": ("NC_000003.1", "Anopheles albitarsis F"),
            "4": ("NC_000004.1", "Culex pipiens pipiens"),
        }
        result = {
            uid: {"accessionversion": acc, "organism": organism}
            for uid, (acc, organism) in names.items()
        }
        return json.dumps({"result": {"uids": list(names), **result}}).encode()
    return (
        f">lcl|NC_000001.1_cds_YP_1.1_1 [gene=COX1] [protein=cytochrome c oxidase I]\n"
        f"{COI_A.lower()}\n"
        ">lcl|NC_000001.1_cds_YP_2.1_2 [gene=ND1] [protein=NADH dehydrogenase 1]\n"
        "ACGTACGT\n"
        f">lcl|NC_000002.1_cds_YP_3.1_1 [gene=COI]\n{COI_B}\n"
        f">lcl|NC_000003.1_cds_YP_4.1_1 [gene=cox1]\n{COI_B}\n"
        ">lcl|NC_000004.1_cds_YP_5.1_1 [gene=ND2]\nACGT\n"
    ).encode()


@pytest.mark.parametrize(
    ("organism", "expected"),
    [
        ("Aedes aegypti", "Aedes aegypti"),
        ("Culex pipiens pipiens", "Culex pipiens"),
        ("Anopheles albitarsis F", None),
        ("Anopheles nr. costai SP02_17_3", None),
        ("Chagasia sp. 1 SJ-2020", None),
        ("Anopheles funestus-like", None),
        ("aedes aegypti", None),
        ("Aedes", None),
    ],
)
def test_binomial(organism: str, expected: str | None) -> None:
    assert binomial(organism) == expected


def test_fetches_coi_anchors_and_lists_every_mitogenome(tmp_path: Path) -> None:
    output = tmp_path / "reference" / "anchors.fasta"

    anchors = fetch_anchors("Culicidae", output, fetch=fake_fetch)

    assert [(a.accession, a.species) for a in anchors] == [
        ("NC_000001.1", "Aedes aegypti"),
        ("NC_000002.1", "Ochlerotatus serratus"),
    ]
    assert anchors[0].sequence == COI_A
    assert read_anchors(output) == anchors
    table = output.with_suffix(".tsv").read_text().splitlines()
    assert table[0].startswith("accession\torganism")
    assert table[3].endswith("no: not a plain species name")
    assert table[4] == (
        "NC_000004.1\tCulex pipiens pipiens\tCulex pipiens\t0\t"
        "no: no COI gene in the annotation"
    )


def test_cli_fetches_anchors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = config_dict(tmp_path)
    config = AppConfig.model_validate(data)
    monkeypatch.setattr(cli, "load_config", lambda *_, **__: config)

    # Not configured
    assert cli.main(["anchors", "app.toml", "--specialist", "culicidae"]) == 1

    data["specialists"]["culicidae"].update(
        anchors=str(tmp_path / "anchors.fasta"), anchor_taxon="Culicidae"
    )
    config = AppConfig.model_validate(data)
    monkeypatch.setattr(
        cli, "fetch_anchors", lambda taxon, path: fetch_anchors(taxon, path, fake_fetch)
    )
    assert cli.main(["anchors", "app.toml", "--specialist", "culicidae"]) == 0
    assert (tmp_path / "anchors.fasta").exists()


def test_requests_name_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []

    class Response(io.BytesIO):
        def __enter__(self) -> "Response":
            return self

    def urlopen(url: str, timeout: float) -> Response:
        opened.append(url)
        return Response(b"data")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    assert refseq._urlopen(refseq._url("esearch", db="nuccore")) == b"data"
    assert opened[0].startswith(refseq.EUTILS + "esearch.fcgi?db=nuccore")
    assert "tool=prokopton-ml" in opened[0]
