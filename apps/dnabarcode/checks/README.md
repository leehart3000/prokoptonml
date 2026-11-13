# Deploy checks

Known sequences sent to every new image before it is pushed (see
`.github/workflows/deploy-dnabarcode.yml`). Each `<name>.json` is an
`/identify` request body, and `<name>.expected` is the start of the answer
text it must give. This is the seed of a regression set: add a pair of
files for each case worth guarding.

| Check | Source | Expected |
|---|---|---|
| `aedes-aegypti-HQ945416` | GenBank HQ945416.1, *Aedes aegypti* COI barcode | *Aedes aegypti* |
