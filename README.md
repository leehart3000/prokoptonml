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
pip install "prokopton-ml[serve,bio]==0.2.0a1"
prokopton-ml --help
```

```python
import prokopton.ml
```

Releases are made with `scripts/release.sh vVERSION`, which checks that the
version matches `pyproject.toml` and that the commit is new work on `main`,
then tags it. The Release workflow checks the same, publishes to TestPyPI
and then, after approval, to PyPI, with Trusted Publishing.

## Products built with ProkoptonML

| Product | Domain | Repository | Status |
|---|---|---|---|
| [DNAbarcode.org](https://dnabarcode.org) | Identifying animals from COI DNA barcodes; species-level for mosquitoes | [leehart3000/dnabarcode](https://github.com/leehart3000/dnabarcode) | Live |
| [pepbarcode.org](https://pepbarcode.org) | Peptide barcoding | | Planned |

Each product has its own repository and its own cloud project, and pins an
exact platform version.

## Repository layout

| Path | Contents |
|---|---|
| `src/prokopton/ml/` | The generic platform: pipeline stages, training, evaluation, model cards, bundles, serving, command-line tool |
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
