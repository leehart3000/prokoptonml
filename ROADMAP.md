# Roadmap

What comes after the launch of DNAbarcode.org (November 2026), roughly in
order.

## Platform and products

1. **Split DNAbarcode into its own repository.** `prokoptonml` becomes the
   platform only: the `prokopton-ml` package on PyPI, the pipeline and its
   command-line tool, the bundle format, a generic serving image and a
   reusable deploy workflow. A new `dnabarcode` repository holds the product:
   its config, reference files, model cards, checks, website and deploy
   settings, and pins a platform version. History is kept with
   `git filter-repo`.
2. **Tagged releases** in the product repository: a `dnabarcode-v*` tag
   deploys the API and the website and runs the browser tests, through a
   protected `production` environment. Google trusts release tags only.
3. **Broad identification beyond animals:** plant rbcL/matK (animals and
   plants) and/or bacterial 16S, to phylum or class, as routers for new
   markers. Licences are checked first.
4. **pepbarcode.org**, built on the same platform, styled to match
   DNAbarcode.

## MLOps

- **Monitoring:** answer types and confidence over time, input lengths and
  drift (Evidently), from the service's JSON logs.
- **Safer releases:** Cloud Run traffic splitting (a new bundle gets a share
  of traffic first), with automatic rollback if the known-answer checks fail.
- **Registry-driven releases:** a new MLflow model version leads to a bundle,
  then CI, then a staged rollout.
- **A regression set** of 10 to 20 known sequences per specialist, run on
  every deploy (started in `apps/dnabarcode/checks/`).
- **Load testing** (Locust or k6) to set concurrency and instance limits.
- **DVC** for data and bundle versions, with the bundle bucket as storage.

## Data and models

- Review the RefSeq curation's questionable removals (*An. costai* /
  *An. forattinii*, *Ae. flavopictus* / *Ae. albopictus*).
- BOLD data and BINs as a second source; alignment-based curation.
- Router weights as float32 (half the bundle size).
- Tell the MIDORI2 maintainers about the README count discrepancy.

## Websites and brand

- prokoptonml.org, with ProkoptonML and Prokopton logos that belong
  together; DNAbarcode and pepbarcode as a pair.
- Move the sites to Astro and Tailwind for shared layouts.
- Email forwarding and DMARC for the three domains.

## Later

- An MCP server, so AI assistants can call DNAbarcode as a tool, and a
  LangGraph "barcoding analyst" agent with evaluation and tracing.
- Species identification from photos, served in the browser.
- A Gradio demo on Hugging Face Spaces.
