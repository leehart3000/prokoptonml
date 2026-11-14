# ProkoptonML

[![CI](https://github.com/leehart3000/prokoptonml/actions/workflows/ci.yml/badge.svg)](https://github.com/leehart3000/prokoptonml/actions/workflows/ci.yml)

**ProkoptonML** is an end-to-end MLOps platform for building, deploying and
monitoring scientific machine learning systems for data-driven bio apps.

It covers the full lifecycle:

1. Data ingestion
2. Feature engineering
3. Training and experiment tracking
4. Model evaluation
5. Packaging and deployment
6. Monitoring

The platform is designed to be repeatable: each application is defined by a
configuration and a small data adapter, and reuses the same pipeline.

## Package

ProkoptonML is published as the `prokopton-ml` package and imported as
`prokopton.ml`, part of the wider `prokopton` namespace:

```python
import prokopton.ml
```

It builds on the [`prokopton`](https://pypi.org/project/prokopton/) core
package (`prokopton.core`), which provides shared foundations such as
configuration loading.

## Applications

| Application | Domain | Status |
|---|---|---|
| [DNAbarcode.org](https://dnabarcode.org) | Identifying animals from COI DNA barcodes; species-level for mosquitoes | Live |
| [pepbarcode.org](https://pepbarcode.org) | Peptide barcoding | Planned |

## How DNAbarcode.org is built

```
MIDORI2 + NCBI RefSeq ─► pipeline (ingest → validate → split → leakage → sample
                         → train → calibrate → evaluate → model cards)
                       ─► serving bundle (models, curated references, config,
                          pinned versions, checksums) ─► Cloud Storage
                       ─► CI: check bundle → build image → test → deploy to Cloud Run
Visitor ─► DNAbarcode.org (Cloudflare Worker: pages, firewall rate limit,
           Turnstile) ─► API key ─► Cloud Run (FastAPI) ─► answer + decision trace
```

- **Models:** a marker classifier (is this COI, which strand?), a router
  (which group?) and a mosquito specialist that aligns the query to curated
  references and answers cautiously: a species, a complex, "one of", or a
  genus. Experiments and model versions are tracked in MLflow.
- **Evaluation:** held-out known and novel species, leakage-aware scores and
  thresholds chosen by the evaluation; see the
  [Culicidae model card](apps/dnabarcode/model-cards/culicidae.md).
- **Serving:** a FastAPI app in a container on Cloud Run (scale to zero),
  deployed by GitHub Actions with Workload Identity Federation (no stored
  Google keys) and least-privilege service accounts; see
  [`infra/gcp/`](infra/gcp/README.md).
- **Website and gateway:** a Cloudflare Worker in
  [`sites/dnabarcode/`](sites/dnabarcode/README.md), with unit and Playwright
  browser tests.
- **Releasing and operating:** [`docs/releasing-dnabarcode.md`](docs/releasing-dnabarcode.md).
- **What's next:** [`ROADMAP.md`](ROADMAP.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/prokopton/ml/` | The platform: data, features, pipeline stages, training, specialists, inference, serving, reporting |
| `apps/dnabarcode/` | DNAbarcode's config, reviewed reference files, model cards and known-answer checks |
| `analysis/` | One-off analyses, with each question, finding and decision |
| `infra/` | Google Cloud setup runbook and deploy settings |
| `sites/dnabarcode/` | The website and API gateway |
| `docs/` | Operating guides |

## Status

The first product, DNAbarcode.org, launched in November 2026. The platform
is young and changing quickly; see [`ROADMAP.md`](ROADMAP.md).

## Licence

Copyright (C) 2026 Lee Hart

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the [LICENSE](https://github.com/leehart3000/prokoptonml/blob/main/LICENSE) file for details.

SPDX-License-Identifier: `AGPL-3.0-only`

Data and trained models may be subject to separate licences, which are
documented alongside them.
