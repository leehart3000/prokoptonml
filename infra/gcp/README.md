# Google Cloud setup

How the Google Cloud resources for ProkoptonML were created, so they can be
checked or rebuilt. Commands are run once, by hand, from the repository
root with `gcloud` logged in as the project owner. Nothing here holds a
secret: the API key lives only in Secret Manager.

| Setting | Value |
|---|---|
| Project | `prokoptonml` |
| Region | `europe-west2` (London) |
| Budget alert | £10 a month (set in the console) |

## 1. Project

Created in the console, with billing linked and the needed APIs enabled.

```bash
gcloud config set project prokoptonml
```

## 2. Artifact Registry (Docker images for Cloud Run)

One Docker repository. The cleanup policy keeps the 5 newest images and
deletes anything else older than 7 days, so storage stays near the free
tier. `set-cleanup-policies` only previews a policy without `--no-dry-run`.

```bash
gcloud services enable artifactregistry.googleapis.com
gcloud artifacts repositories create prokoptonml --repository-format=docker \
    --location=europe-west2 --description="ProkoptonML service images"
gcloud artifacts repositories set-cleanup-policies prokoptonml \
    --location=europe-west2 \
    --policy=infra/gcp/artifact-registry-cleanup-policy.json --no-dry-run
```

Images are pushed to `europe-west2-docker.pkg.dev/prokoptonml/prokoptonml`.
Registry vulnerability scanning is off (it is charged per image).
