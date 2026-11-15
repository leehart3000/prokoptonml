# Design 0001: separating the platform from its products

- **Status:** proposed (for review)
- **Date:** 2026-11-15
- **Applies to:** `prokopton-ml` 0.1.0a2 → 0.2.0, and DNAbarcode.org

## 1. Why

ProkoptonML is meant to be a general-purpose MLOps platform, with DNAbarcode.org as its first product. The two are already separated at the boundaries:

- separate repositories, rules and releases;
- the product installs the platform from PyPI at a pinned version;
- separate Google Cloud projects and budgets;
- the data, models, config, model cards and checks all live with the product.

Inside, though, the platform still assumes that every product identifies DNA sequences:

1. **Sequence code is in the core.** FASTA, MIDORI2 and RefSeq readers, DNA k-mers and fragments, alignment, taxonomy and `vsearch` leakage checks are all part of `prokopton.ml`.
2. **The steps are fixed in code.** The `Identifier` always runs marker check → router → specialists, with COI region finding and strand handling. The CLI's commands, and its `--model router|marker` choice, are fixed lists.
3. **The config schema is barcode-shaped.** `AppConfig` has fixed `router`, `marker`, `specialists` and `inference` sections, with fields such as `barcode_gene`, `species_identity` and `complexes`.
4. **Product wording is in the platform.** For example, `inference/identify.py` says that rbcL, ITS and 16S support "is planned". Card defaults describe the input as "A DNA sequence".
5. **Deployment can't be reused.** The Dockerfile, deploy workflow, known-answer check runner and release script live in `dnabarcode`. A second product would have to copy them.

pepbarcode.org could probably bend the current code to fit, but a product that doesn't work with sequences (step 7 of the plan) couldn't use it at all.

## 2. Goals

- **G1. A generic core.** `prokopton.ml` knows about stages, datasets, models, experiments, evaluation, model cards, bundles, serving and releases, but nothing about biology.
- **G2. A sequence extension.** Everything sequence-specific moves to `prokopton.ml.bio`, which DNAbarcode.org and pepbarcode.org use.
- **G3. Products describe themselves in config.** A product declares its pipeline stages and serving steps, and all its user-facing wording, in its own TOML file, and can add its own stage or step in Python without changing the platform.
- **G4. Reusable deployment.** A product deploys with the platform's reusable workflow and check runner, with only its own settings file and bundle.
- **G5. A product template,** so that a new product starts with the repository layout, CI, rules, deploy and release already in place.
- **G6. No change for users of DNAbarcode.org.** Every step of the migration keeps the same answers. The 13 known-answer checks pass, and a rebuilt bundle gives identical predictions and identical references.

## 3. Non-goals (for now)

- Monitoring (Evidently), remote MLflow or DVC, traffic splitting, LLMOps and prokoptonml.org. These are on the roadmap, but come after this work.
- Clouds other than Google Cloud Run. The deploy workflow is written for Cloud Run, with the cloud-specific parts kept in one place.
- Retraining any DNAbarcode.org model.

## 4. Where the current code belongs

This is a first pass, from each module's imports and contents. It will be confirmed module by module in phase 1.

| Module | Destination | Notes |
|---|---|---|
| `config.py` | split | Core: app name and title, paths, MLflow, stage and step registry. Bio: `Midori2SourceConfig`, validation (phylum lineages), split by species, leakage, k-mer classifiers, specialists, inference rules. |
| `cli.py` | core | Commands become "run a named stage", not a fixed list. |
| `data/checksums.py`, `data/download.py` | core | |
| `data/fasta.py`, `data/taxonomy.py` | bio | |
| `sources/midori2.py`, `sources/refseq.py` | bio | |
| `features/kmers.py`, `features/fragments.py` | bio | |
| `training/tracking.py`, `training/metrics.py`, `training/calibration.py` | core | No platform imports beyond config. |
| `training/router.py` | split | The model search (naive Bayes and logistic regression, MLflow runs, registry aliases) is core. Its k-mer features and fragments are bio, and are passed in. |
| `pipeline/split.py` | split | A salted, group-aware hash split is core. Grouping by species and holding out novel species is bio. |
| `pipeline/sample.py` | bio | Diversity by species or order. |
| `pipeline/train.py`, `pipeline/calibrate.py`, `pipeline/evaluate.py` | split | The stage machinery is core; fragment lengths and lengths as a calibration feature are bio. |
| `pipeline/ingest.py`, `pipeline/validate.py`, `pipeline/leakage.py`, `pipeline/reference.py`, `pipeline/specialist_evaluate.py` | bio | |
| `evaluation/scores.py` | split | Metrics are core; scoring by fragment length is bio. |
| `reporting/charts.py` | core | |
| `reporting/card.py` | split | The card framework (sections, charts, notes, MLflow lineage) is core; DNA wording and sections are bio. |
| `reporting/specialist_card.py` | bio | |
| `specialists/*` | bio | |
| `inference/identify.py` | split | `Step`, `Result`, the decision trace and the step runner are core; the marker, region search, router and specialist steps are bio. |
| `inference/inputs.py` | bio | FASTA, FASTQ and protein detection, ambiguity. |
| `inference/bundle.py`, `inference/load.py` | core | The artifacts a bundle holds come from the configured steps, not a fixed list. |
| `serving/app.py` | core | The request schema comes from the first serving step. |

## 5. Design

### 5.1 Packages

`prokopton.ml.bio` ships in the same distribution as the core, as an optional extra: `prokopton-ml[bio]`. One package keeps one version and one release while the interfaces settle. A test enforces the boundary: no core module may import from `prokopton.ml.bio`. CI also runs the core tests with the bio dependencies absent.

Splitting `bio` into its own distribution (`prokopton-ml-bio`) can come later, once the interfaces are stable. It would then need no code changes.

### 5.2 Stages

A **stage** is one step of a product's pipeline, such as ingest, validate, split, train or bundle. Each stage:

- has a **kind** (such as `bio.midori2-ingest` or `core.train-classifier`) with its own typed config model;
- declares its inputs and outputs (paths, or MLflow models);
- records its run in MLflow and writes a small `stage.json` with its inputs' checksums, so that a stage that is already up to date can say so.

Kinds are registered through Python entry points (`prokopton.ml.stages`). The core and `bio` register theirs this way, and so can a product's own small package. The CLI becomes:

```bash
prokopton-ml stages apps/dnabarcode/dnabarcode.toml        # list the configured stages
prokopton-ml run apps/dnabarcode/dnabarcode.toml router-train
prokopton-ml run apps/dnabarcode/dnabarcode.toml --all     # in order, skipping those up to date
```

Pixi tasks in the product keep their current names, so how a product is worked on doesn't change.

### 5.3 Config (schema version 2)

The product's TOML file keeps its structure of named sections, but each section says what kind it is:

```toml
schema = 2
name = "dnabarcode"
title = "DNAbarcode.org"

[stages.router-train]
kind = "core.train-classifier"
features = { kind = "bio.kmers", k = [8] }
# ...the current [router.training] settings...

[serving]
steps = ["input", "marker", "router", "culicidae"]

[serving.steps.marker]
kind = "bio.marker"
model = "dnabarcode-marker@candidate"

[serving.messages]
unsupported_marker = "Support for other markers (plant rbcL, fungal ITS and bacterial 16S) is planned."
```

Pydantic discriminated unions, built from the registered kinds, validate the file, so a typo is still an error at load time. `prokopton-ml migrate-config` converts a version 1 file to version 2. For one minor release, the platform also reads version 1 files directly (with a warning), so DNAbarcode.org can move in its own time.

### 5.4 Serving steps

Serving becomes a chain of configured **steps**. Each step gets the request and the answer so far, and returns one of:

- **continue,** with what it found (such as the oriented sequence, the region, or the router's top groups);
- **answer,** a more specific answer that replaces the current one;
- **decline,** with a reason code and a message.

The core runs the chain, times each step, keeps the decision trace, and returns the same `Result` shape as today. The current pipeline becomes the bio steps `input` (sequence checks), `marker` (strand and region), `router`, and `specialist` (one per specialist). Routing rules, such as the 10% threshold, stay in each step's config.

The first step defines the request schema. For `bio.input`, that's one sequence of text, as today. The endpoint stays `/identify`, which suits most prediction products. `/health` and `/version` are unchanged.

### 5.5 Bundles

Each step declares the artifacts it needs: a model from the registry, a reference set, an index. `prokopton-ml bundle` collects them, and adds the config, `requirements.txt` and the SHA-256 manifest as today. Loading still checks every file's checksum, and the numpy, scipy and scikit-learn versions. The bundle format gets a version number, and the loader refuses formats it doesn't know.

### 5.6 Wording

Every message a user can see (declines, notes, answer templates, model card text) comes from the product's config. `bio` provides neutral defaults, with no product or organism named.

### 5.7 Reusable deployment

The platform's repository provides:

- **A reusable workflow, `deploy-cloud-run.yml`,** called by a product with `uses: leehart3000/prokoptonml/.github/workflows/deploy-cloud-run.yml@v0.2.0`. It reads the product's settings file (today's `infra/deploy/dnabarcode.env`), then:
  1. downloads and checks the bundle;
  2. builds the image;
  3. runs the known-answer checks;
  4. pushes the image;
  5. deploys;
  6. checks the live service.
- **A check runner, `prokopton-ml check BUNDLE CHECKS_DIR`,** replacing the `jq` loop in the deploy workflow. It runs every check, reports every mismatch, and also works locally before a release.
- **A Dockerfile template,** taking the platform version, title and port as build arguments. Each bundle pins its own numpy and scikit-learn versions, so the image is still built per product rather than published as one shared image. The template is also the place for the cold-start work (a smaller image and lazy imports).
- **`release.sh` and the tag check,** made generic: the site URL and the product name are inputs.

### 5.8 Product template

A template repository, `prokopton-product-template`, holds:

- the layout (`apps/<name>/`, `checks/`, `infra/deploy/`, `sites/<name>/`, `docs/`);
- `pixi.toml` pinned to the platform, and a minimal version 2 config;
- the rulesets to copy, CI, Dependabot, secret scanning, and the release and deploy workflows calling the platform's;
- a `docs/new-product.md` checklist for the cloud side (a GCP project, budgets, workload identity, the image repository and the secret).

A new product is created from the template with `gh repo create --template`.

## 6. How we'll know it worked

1. **A non-sequence example runs end to end.** The platform's repository gets a small tabular example under `examples/`, using a public dataset with a permissive licence: config, stages, training, evaluation, a model card, a bundle and serving. Its tests pass with `bio` absent. This proves G1 and G3.
2. **DNAbarcode.org is unchanged.**
   - Its 13 known-answer checks pass on every phase's release.
   - A bundle rebuilt with the new platform has identical references (compared by SHA-256), and its models give identical probabilities on the golden set. The model files themselves may differ (see §9).
   - The answers to a fixed set of about 200 sequences are identical, compared field by field, including the trace.
3. **No biology in the core.** The import-boundary test passes, and a search for `mosquito`, `COI`, `MIDORI`, `species` and `FASTA` in the core finds only comments explaining the boundary.

## 7. Plan

Each phase is one or more platform alpha releases (0.2.0a1, a2, …), followed by a DNAbarcode.org release that pins the new version and passes its checks.

| Phase | What | DNAbarcode.org changes |
|---|---|---|
| 0. Guards | Release script and tag check for the platform; the import-boundary test (allowing today's violations, listed); a golden set of about 200 sequences and their answers, frozen as a test fixture. | none |
| 1. Move | Move the bio modules into `prokopton.ml.bio`, with re-exports at the old paths for one release; move wording into config with neutral defaults. | wording added to its config |
| 2. Stages and config | The stage registry, `prokopton-ml run` and `stages`, config schema version 2 and `migrate-config`; reading version 1 files is kept for one release. | config migrated; pixi tasks call `run` |
| 3. Serving steps and bundles | The serving step chain and the bundle format version 2, built from steps. | bundle rebuilt and compared |
| 4. Deployment | The reusable deploy workflow, check runner, Dockerfile template and generic release script. | its workflows call the platform's |
| 5. Proof and template | The tabular example; the product template; `resave-models`; remove the re-exports and version 1 config reading; release 0.2.0. | models re-saved, bundle rebuilt; pins 0.2.0 |

The platform's 100% test coverage and strict typing stay throughout.

## 8. Decisions needed

1. **Packaging of `bio`:** an extra in the same distribution (recommended, §5.1), or a separate package from the start?
2. **Template mechanism:** a GitHub template repository (recommended; simple, but products don't pick up later template changes), or `copier` (products can be updated from the template, at the cost of another tool)?
3. **Endpoint name:** keep `/identify` for every product (recommended), or a generic `/predict` with `/identify` as an alias?
4. **Example dataset for the proof:** a small public tabular dataset with a permissive licence, such as Palmer Penguins (CC0). Or would you prefer a domain closer to the step 7 product?

## 9. Risks

- **Moving code can change behaviour.** This is guarded by the golden answers, bundle hashes and known-answer checks at every phase.
- **Over-generalising before a second product exists.** Only the extension points needed by DNAbarcode.org, the tabular example and pepbarcode are built; anything else waits for a real use.
- **Config migration mistakes.** `migrate-config` is tested on DNAbarcode.org's real config: loading the version 2 file must produce the same effective settings as version 1.
- **Pickled models referring to old module paths.** Models saved with joblib refer to their classes' import paths, and the router and marker include the platform's own k-mer vectoriser, so moving it would stop saved models and bundles from loading. In phase 1 the old path stays as an alias. Before it's removed in phase 5, a `prokopton-ml resave-models` step loads each registered model and saves it again as a new registry version under the new paths; it must give identical probabilities on the golden set. A test checks that the live bundle still loads until it's replaced.
