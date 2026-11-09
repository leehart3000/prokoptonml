# ProkoptonML

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

## Applications

| Application | Domain | Status |
|---|---|---|
| [dnabarcode.org](https://dnabarcode.org) | DNA barcoding | Planned |
| [pepbarcode.org](https://pepbarcode.org) | Peptide barcoding | Planned |

## Status

This project is in early development.

## Licence

Copyright (C) 2026 Lee Hart

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, version 3 only.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the [LICENSE](LICENSE) file for details.

SPDX-License-Identifier: `AGPL-3.0-only`

Data and trained models may be subject to separate licences, which are
documented alongside them.
