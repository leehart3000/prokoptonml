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

## 3. Cloud Storage (serving bundles)

A private bucket of serving bundles, one dated folder per bundle per app
(`<app>/<date>/`). Bundles are built locally (they need the full data) and
uploaded; CI downloads the one named in `infra/deploy/<app>.env`, checks it
against its manifest and builds it into the image.

```bash
gcloud storage buckets create gs://prokoptonml-bundles --location=europe-west2 \
    --uniform-bucket-level-access --public-access-prevention
gcloud storage rsync data/models/dnabarcode/bundle \
    gs://prokoptonml-bundles/dnabarcode/2026-11-12 --recursive
```

## 4. Service accounts and Workload Identity Federation

**One Cloud Run service, and one runtime service account, per app.** CI
signs in without stored keys: GitHub's OIDC token is exchanged for a
short-lived Google token, accepted only for this repository (by its fixed
numeric ID) on `main`.

- `dnabarcode-api`: the service's runtime identity. Only permission: read
  its own API-key secret.
- `github-deployer`: CI's identity. Push to the one Artifact Registry
  repository; read the one bundle bucket; deploy Cloud Run services
  (`run.developer`, which cannot change who may call a service); act as
  `dnabarcode-api` only.

```bash
PROJECT_ID=prokoptonml
PROJECT_NUMBER="$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')"
REPO_ID="$(gh api repos/leehart3000/prokoptonml --jq .id)"
DEPLOYER="github-deployer@$PROJECT_ID.iam.gserviceaccount.com"
RUNTIME="dnabarcode-api@$PROJECT_ID.iam.gserviceaccount.com"
gcloud services enable iam.googleapis.com iamcredentials.googleapis.com \
    sts.googleapis.com run.googleapis.com
gcloud iam service-accounts create dnabarcode-api \
    --display-name="dnabarcode API (Cloud Run runtime)"
gcloud iam service-accounts create github-deployer \
    --display-name="GitHub Actions deployer"
gcloud artifacts repositories add-iam-policy-binding prokoptonml \
    --location=europe-west2 --member="serviceAccount:$DEPLOYER" \
    --role=roles/artifactregistry.writer
gcloud storage buckets add-iam-policy-binding gs://prokoptonml-bundles \
    --member="serviceAccount:$DEPLOYER" --role=roles/storage.objectViewer
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$DEPLOYER" --role=roles/run.developer --condition=None
gcloud iam service-accounts add-iam-policy-binding $RUNTIME \
    --member="serviceAccount:$DEPLOYER" --role=roles/iam.serviceAccountUser
gcloud iam workload-identity-pools create github --location=global \
    --display-name="GitHub Actions"
gcloud iam workload-identity-pools providers create-oidc prokoptonml-main \
    --location=global --workload-identity-pool=github \
    --display-name="prokoptonml main branch" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.repository_id=assertion.repository_id,attribute.ref=assertion.ref" \
    --attribute-condition="assertion.repository_id=='$REPO_ID' && assertion.ref=='refs/heads/main'"
gcloud iam service-accounts add-iam-policy-binding $DEPLOYER \
    --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/github/attribute.repository_id/$REPO_ID"
```

## 5. Secret Manager (API key)

A random key, generated and stored without being shown, kept in London
only, readable only by the runtime service account. Without a key the app
leaves `/identify` open, so the service is never deployed without one.

```bash
gcloud services enable secretmanager.googleapis.com
openssl rand -hex 32 | tr -d '\n' | gcloud secrets create dnabarcode-api-key \
    --replication-policy=user-managed --locations=europe-west2 --data-file=-
gcloud secrets add-iam-policy-binding dnabarcode-api-key \
    --member="serviceAccount:$RUNTIME" --role=roles/secretmanager.secretAccessor
```

## 6. Deploy

`.github/workflows/deploy-dnabarcode.yml`, run by hand from `main`, with the
settings in `infra/deploy/dnabarcode.env`. It downloads and checks the
bundle, builds the image, tests it (health, key required, and the known
answers in `apps/dnabarcode/checks/`), pushes it, deploys it by digest and
checks the live service.
