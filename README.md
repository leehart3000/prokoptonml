# ProkoptonML

[![CI](https://github.com/leehart3000/prokoptonml/actions/workflows/ci.yml/badge.svg)](https://github.com/leehart3000/prokoptonml/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/prokopton-ml?include_prereleases)](https://pypi.org/project/prokopton-ml/)

**ProkoptonML** is an end-to-end MLOps platform: a Python package and
command-line tool for turning data into evaluated, documented and deployable
models, used by separate products that pin it from PyPI.

It covers the lifecycle:

1. Data ingestion, with checksums and validation
2. Splitting and leakage checks
3. Feature engineering
4. Training and experiment tracking (MLflow), with a model registry
5. Calibration and evaluation, with thresholds chosen by the evaluation
6. Model cards
7. Serving bundles (models, reference data, config, pinned versions,
   checksums) and a FastAPI service that serves them

Each product is defined by a configuration file and runs the platform's
stages on it; the product owns its data, decisions, website and deployment.

## Package

Published as [`prokopton-ml`](https://pypi.org/project/prokopton-ml/) and
imported as `prokopton.ml`, part of the `prokopton` namespace; it builds on
the [`prokopton`](https://pypi.org/project/prokopton/) core package
(`prokopton.core`).

```bash
pip install "prokopton-ml[serve,bio]==0.2.0a7"
prokopton-ml stages                  # an app's pipeline, and when each stage ran
prokopton-ml run router-train        # run stages (or --all, or --from STAGE)
prokopton-ml identify --input x.fa   # identify one sequence
prokopton-ml check bundle tests/known-answers   # known-answer checks
```

A known-answer check is a request (`<name>.json`) and the start of the
answer it must give (`<name>.expected`: the answer's text, or
`declined: <reason>`). `prokopton-ml check` runs every check against a
serving bundle, or with `--url` against a running service (its API key from
`PROKOPTON_API_KEY`), and fails if any answer differs.

An app is described by `prokopton.toml` (found in the current folder or a
parent folder): its models, and its pipeline as named stages, each with a
kind such as `bio.ingest` or `core.bundle`. A package can add stage kinds
with an entry point in the `prokopton.ml.stages` group: a function that
returns a list of `prokopton.ml.stages.StageKind`.

Answers are built by the app's serving steps (`[serving]`), each also with a
kind (such as `bio.sequence-input` or `bio.specialist`), and served at
`POST /<endpoint>` with the app's own request, result and description.
Packages can add serving step kinds (`prokopton.ml.serving_steps`) and model
kinds (`prokopton.ml.model_kinds`) in the same way.

The config itself (schema 4) holds only what every app has: its name, data
release (`[data]`), models folder (`[paths]`), models, pipeline, serving steps
and MLflow settings. Each kind's own settings, and app-wide sections such as
`[bio]` for sequence apps, are declared by the package that reads them
(`prokopton.ml.settings`) and checked when the config is read. Schema 3
configs are converted when read, and `prokopton-ml migrate-config --write`
rewrites them as schema 4, keeping their comments. Schemas 1 and 2 are no
longer read: rewrite them with prokopton-ml 0.2.0a7 first.

```python
import prokopton.ml
```

Releases are made with `pixi run release vVERSION` (`prokopton-ml release`),
which checks that the version matches `pyproject.toml` and that the commit is
new work on `main`, then tags it and follows the Release workflow. The
workflow checks the same and publishes to TestPyPI; when it waits for
approval to publish to PyPI (with Trusted Publishing), the command offers to
approve it, then checks that PyPI lists the new version. Products release
with the same command, with `--check-url` to check that their live page shows
the new version.

The platform is not tied to sequences: [`examples/penguins/`](examples/penguins/)
is a small tabular app (Palmer Penguins) built only on the core's tabular
kinds (`core.tabular-classifier`, its stages and serving steps). CI runs it
end to end with `prokopton.ml.bio` removed; locally, `pixi run penguins`.

A product deploys its API to Google Cloud Run with the reusable workflow
[`deploy-cloud-run.yml`](.github/workflows/deploy-cloud-run.yml), pinned to a
platform release's commit (`@<commit> # v0.2.0`, as for actions). It builds the product's image (from
[`templates/Dockerfile`](templates/Dockerfile)) with its serving bundle, runs
the product's known-answer checks against the image with `prokopton-ml check`,
then pushes and deploys it.

## Products built with ProkoptonML

| Product | Domain | Repository | Status |
|---|---|---|---|
| [DNAbarcode.org](https://dnabarcode.org) | Identifying animals from COI DNA barcodes; species-level for mosquitoes | [leehart3000/dnabarcode](https://github.com/leehart3000/dnabarcode) | Live |
| [pepbarcode.org](https://pepbarcode.org) | Peptide barcoding | | Planned |

Each product has its own repository and its own cloud project, and pins an
exact platform version. A new product can start from the template
repository,
[leehart3000/prokoptonml-product-template](https://github.com/leehart3000/prokoptonml-product-template):
the Penguins example as a product, with its CI, deploy and release
workflows.

## Repository layout

| Path | Contents |
|---|---|
| `src/prokopton/ml/` | The generic platform: config, pipeline stages, training, evaluation, model cards, bundles, serving, command-line tool, and the tabular kinds (`tabular/`) |
| `src/prokopton/ml/bio/` | Sequence-specific code: MIDORI2 and RefSeq sources, FASTA, k-mer features, specialists, sequence input checks (the `bio` extra) |
| `tests/` | Tests (100% coverage), with frozen fixtures in `tests/fixtures/` |
| `.github/workflows/` | CI, secret scanning, lockfile updates and releases to PyPI |

## Working on it

```bash
pixi install
pixi run check    # format, lint, type-check, test
```

What's next: [`ROADMAP.md`](ROADMAP.md).

## Licence

Copyright (C) 2026 Lee Hart

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the [LICENSE](https://github.com/leehart3000/prokoptonml/blob/main/LICENSE) file for details.

SPDX-License-Identifier: `AGPL-3.0-only`

Data and trained models built with the platform belong to each product and
are licensed there.
