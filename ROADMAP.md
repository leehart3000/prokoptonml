# Roadmap

What comes next for the ProkoptonML platform. Product plans live in each
product's repository (for example
[leehart3000/dnabarcode](https://github.com/leehart3000/dnabarcode)).

## A general-purpose platform

- **A `prokopton.ml.bio` module** for the sequence-specific parts (MIDORI2
  source, k-mer features, the sequence aligner and specialists, RefSeq
  anchors), so the rest of the platform is general-purpose and non-bio
  products can use it.
- **A generic serving image** (platform plus FastAPI, any bundle), so
  products only add their bundle.
- **A reusable deploy workflow** that products call.
- **Plug-ins for product-specific data sources and rules**, so products do
  not need platform changes.

## MLOps

- **Monitoring:** answer types, confidence, input lengths and drift
  (Evidently), from the service's JSON logs.
- **Safer releases:** traffic splitting with automatic rollback.
- **Registry-driven releases:** a new MLflow model version leads to a bundle,
  then a staged rollout.
- **Data and model storage independent of a working copy:** a remote MLflow
  store (for example Cloud Storage or DagsHub), and DVC for data and bundle
  versions, so products' `data/` folders can move freely.
- **Load testing** to set serving limits from evidence.
- **LLMOps:** evaluation sets, tracing, prompt versions and cost tracking for
  agentic products (for example a LangGraph agent), and an MCP server.

## Website and brand

- **prokoptonml.org** (Astro and Tailwind), with SVG logos for ProkoptonML
  and Prokopton that belong together.

## Products

- **DNAbarcode.org:** live; its plans are in its repository.
- **pepbarcode.org:** next, on the same platform, styled to match DNAbarcode.
- **More:** candidates include forecasting, photo identification served in
  the browser, and AI features for other projects.
