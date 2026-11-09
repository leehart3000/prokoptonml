# Analysis scripts

One-off analyses that informed a decision in the project. Each script is small,
numbered in the order it was written, and run in the project's locked
environment:

```bash
pixi run python analysis/<script>.py [arguments]
```

Outputs go to the git-ignored `data/` folder (or a folder you choose), never to
the repo. Anything that every pipeline run should produce belongs in a pipeline
stage instead, so it never goes stale. Scripts are linted with Ruff but are not
type-checked or covered by tests.

| Script | Question | Finding | Decision |
|---|---|---|---|
| `01_list_phyla.py` | Which phyla are in MIDORI2 GB273, and how many records does each have? | 69 named phyla across 15 genes (5,581,260 records); CO1 has 3,213,682 records, 5,704 without a phylum. Includes animals, fungi, plants, algae and other eukaryotes. MIDORI2's README counts are slightly higher than the files (e.g. CO1 +15, CO2 +7); our `grep -c '^>'` check matches our counts. | Build a reviewed phylum → kingdom table from NCBI lineages (`02`). Tell the MIDORI2 maintainers about the README counts after launch. |
| `02_fetch_phylum_lineages.py` | Which kingdom (or major group) does each phylum belong to? | _To be recorded after review._ | _To be recorded._ |
