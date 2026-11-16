# Test fixtures

Frozen copies of real application files, so the platform's tests do not
depend on any application living in this repository. The configs'
`phylum_reference` points at the copy here, so the tests can check it exists.

| File | Copied from | Used by |
|---|---|---|
| `dnabarcode.toml` | DNAbarcode's app config (November 2026), rewritten as schema 4 by `migrate-config` | `tests/test_config.py` and others |
| `dnabarcode-schema-3.toml` | The same config as schema 3, as `migrate-config` in prokopton-ml 0.2.0a7 wrote it | `tests/test_migrate.py` |
| `phylum_lineages_gb273.tsv` | DNAbarcode's reviewed phylum table (MIDORI2 GB273) | `tests/data/test_taxonomy.py` |
