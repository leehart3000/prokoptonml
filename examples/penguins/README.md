# Example: Palmer Penguins

A small app built on ProkoptonML that has nothing to do with DNA: it
predicts a penguin's species (Adélie, Chinstrap or Gentoo) from its bill
length and depth, flipper length, body mass, island and sex. It is the proof
that the platform is general: it uses only the core, and CI runs it end to
end with `prokopton.ml.bio` removed.

| File | What it is |
|---|---|
| `prokopton.toml` | The app: one tabular classifier, its pipeline and its serving steps |
| `data/` | The Palmer Penguins data (CC0), with its source, checksums and citations |
| `tests/known-answers/` | Records whose answers must not change, checked with `prokopton-ml check` |

## Run it

From the platform's repository:

```bash
pixi run penguins
```

That runs every stage (split, train, evaluate, card, bundle) in this folder,
with MLflow in `mlflow/`, then checks the known answers against the bundle in
`models/bundle`. The model card is written to `model-card.md`. To serve the
bundle locally (`POST /classify`):

```bash
cd examples/penguins
PROKOPTON_BUNDLE=models/bundle pixi run uvicorn --factory prokopton.ml.serving.app:create_app_from_env --port 8080
```

On this data the classifier is right about 99% of the time on the test set:
Palmer Penguins is an easy problem, which suits a proof.
