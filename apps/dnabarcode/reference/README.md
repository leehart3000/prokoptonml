# dnabarcode reference data

Small, reviewed reference files that the pipeline reads. They are committed so
that every run uses exactly the same, version-controlled values.

## `phylum_lineages_gb273.tsv`

Every phylum found in MIDORI2 GB273 (68 named phyla, plus records with no
phylum), with its record counts and its full lineage from NCBI Taxonomy.

| Column | Meaning |
|---|---|
| `phylum`, `phylum_taxid` | Phylum name and NCBI taxonomy ID, as given by MIDORI2 |
| `co1_records`, `all_records` | Records in CO1, and across all 15 genes |
| `kingdom` | NCBI's kingdom-rank taxon (Metazoa, Fungi or Viridiplantae), or empty |
| `major_group` | The first group below Eukaryota, e.g. Opisthokonta, Sar, Amoebozoa |
| `lineage` | Full NCBI lineage, from "cellular organisms" down to the phylum |

**How it was made:** `analysis/01_list_phyla.py` listed the phyla in the
ingested data; `analysis/02_fetch_phylum_lineages.py` fetched their lineages
from NCBI E-utilities on 2026-11-09. The result was reviewed by hand: every
named phylum received a lineage.

**How the pipeline uses it:** each record's top-level **group** is the
`kingdom` if present, otherwise "Other eukaryotes":

| Group | Phyla | CO1 records |
|---|---|---|
| Metazoa (animals) | 31 | 3,162,015 |
| Fungi | 14 | 8,583 |
| Viridiplantae (green plants and algae) | 3 | 4,632 |
| Other eukaryotes | 20 | 32,748 |
| _(no phylum; excluded from training)_ | | 5,704 |

**Updating:** when moving to a new MIDORI2 release, re-run both scripts, review
the differences (`git diff`), and commit the new file under a new name.

NCBI Taxonomy data is in the public domain. Cite: Schoch CL, et al. NCBI
Taxonomy: a comprehensive update on curation, resources and tools. Database
(Oxford). 2020: baaa062.
