# Releasing and operating DNAbarcode.org

Step-by-step guides for the common jobs. Commands run from the repository
root on a Mac with `pixi`, `gcloud` (signed in as the project owner), `gh`
and Node 22+. Nothing here prints a secret.

| Part | Where it runs | How it is deployed |
|---|---|---|
| Models and references (the bundle) | Built locally | Uploaded to `gs://prokoptonml-bundles/dnabarcode/<date>/` |
| API | Cloud Run `dnabarcode-api`, `europe-west2` | `.github/workflows/deploy-dnabarcode.yml` (by hand, from `main`) |
| Website and gateway | Cloudflare Worker `dnabarcode-site` | `wrangler deploy` from a Mac |

## 1. Release a new bundle (new models or reference data)

1. Make the change (config, data, code) in a pull request, as usual.
2. Rebuild what changed. For the mosquito specialist:

   ```bash
   pixi run dnabarcode-culicidae-anchors      # only to refresh the RefSeq anchors
   pixi run dnabarcode-culicidae-reference
   pixi run dnabarcode-culicidae-evaluate     # chooses the thresholds
   pixi run dnabarcode-culicidae-card         # refuses if the config's thresholds differ
   ```

   The router and marker models have their own `dnabarcode-*` and
   `dnabarcode-marker-*` tasks (sample, train, calibrate, evaluate, card).
3. Commit the updated model cards, then build and upload the bundle under a
   new dated folder (never overwrite one that is in use):

   ```bash
   pixi run dnabarcode-bundle
   gcloud storage rsync data/models/dnabarcode/bundle \
       gs://prokoptonml-bundles/dnabarcode/<YYYY-MM-DD> --recursive
   ```

4. In a pull request, set `BUNDLE=` in `infra/deploy/dnabarcode.env` to the
   new folder. If answers are expected to change, update or add known-answer
   checks in `apps/dnabarcode/checks/`.
5. After merging, deploy and watch:

   ```bash
   gh workflow run deploy-dnabarcode.yml --ref main -f check_public=true
   gh run watch "$(gh run list --workflow=deploy-dnabarcode.yml --limit 1 --json databaseId --jq '.[0].databaseId')" --exit-status
   ```

   The workflow checks the bundle against its manifest, tests the image
   (health, key required, known answers) before pushing it, deploys it by
   digest and checks the live service.
6. Run the browser tests: `gh workflow run site.yml --ref main`.

## 2. Change the website or gateway

1. Change `sites/dnabarcode/` in a pull request (the unit tests run in CI).
   Optionally run the browser tests against a local preview first.
2. After merging, deploy and test:

   ```bash
   npx --yes wrangler@4.162.0 deploy --config sites/dnabarcode/wrangler.jsonc
   gh workflow run site.yml --ref main
   ```

## 3. Roll back

- **API:** list revisions and send all traffic to the previous one:

  ```bash
  gcloud run revisions list --service dnabarcode-api --region europe-west2
  gcloud run services update-traffic dnabarcode-api --region europe-west2 --to-revisions <REVISION>=100
  ```

  The next deploy sends traffic to the new revision again.
- **Website:** `npx --yes wrangler@4.162.0 rollback --config sites/dnabarcode/wrangler.jsonc`
  (choose the version to return to).

## 4. Replace the API key

Do this if the key may have leaked. Identifications fail for a minute or
two while the two sides switch over.

```bash
openssl rand -hex 32 | tr -d '\n' | gcloud secrets versions add dnabarcode-api-key --data-file=-
gcloud secrets versions access latest --secret=dnabarcode-api-key \
    | npx --yes wrangler@4.162.0 secret put API_KEY --config sites/dnabarcode/wrangler.jsonc
gcloud run services update dnabarcode-api --region europe-west2 \
    --update-secrets PROKOPTON_API_KEY=dnabarcode-api-key:latest
gcloud secrets versions list dnabarcode-api-key       # then disable the old version:
gcloud secrets versions disable <OLD_VERSION> --secret=dnabarcode-api-key
```

The Turnstile secret is replaced in the Cloudflare dashboard (Turnstile >
the widget > rotate secret), then with
`npx --yes wrangler@4.162.0 secret put TURNSTILE_SECRET --config sites/dnabarcode/wrangler.jsonc`.

## 5. Look at what is happening

```bash
# API logs (JSON lines; sequences are never logged)
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="dnabarcode-api"' --limit 30 --freshness 1d
# Gateway logs, live
npx --yes wrangler@4.162.0 tail --config sites/dnabarcode/wrangler.jsonc
```

## 6. Costs and limits

- Cloud Run: 1 CPU, 2 GiB, 4 requests at a time, 0 to 2 instances (scale to
  zero). Settings in `infra/deploy/dnabarcode.env`.
- Firewall rate limit: 5 identifications per 10 seconds per IP (Cloudflare
  dashboard); every identification also needs a Turnstile token.
- Budget alert: £10 a month on the Google Cloud project.
- Storage: old bundles and images are cleaned up as described in
  [`infra/gcp/README.md`](../infra/gcp/README.md).
