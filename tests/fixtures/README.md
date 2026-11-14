# Test fixtures

Frozen copies of real application files, so the platform's tests do not
depend on any application living in this repository.

| File | Copied from | Used by |
|---|---|---|
| `dnabarcode.toml` | DNAbarcode's app config (November 2026) | `tests/test_config.py` |
| `phylum_lineages_gb273.tsv` | DNAbarcode's reviewed phylum table (MIDORI2 GB273) | `tests/data/test_taxonomy.py` |
