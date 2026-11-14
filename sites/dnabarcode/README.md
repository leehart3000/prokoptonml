# dnabarcode.org

The website (static pages in `public/`: Identify, What can I submit?,
Finding the COI region, Scope and limits) and a small API gateway
(`src/index.js`), deployed together as one Cloudflare Worker on the
`dnabarcode.org` custom domain. Visitors' browsers never see the API key:
the gateway limits calls per visitor, checks a Cloudflare Turnstile token
for every identification, and forwards only the sequence to the Cloud Run
service with the key.

| Setting | Where |
|---|---|
| API URL, expected hostname, Turnstile action | `vars` in `wrangler.jsonc` |
| Rate limits (identify 10/min, health 30/min per visitor) | `ratelimits` in `wrangler.jsonc` |
| `API_KEY` (the service's key) | Worker secret, piped from Secret Manager |
| `TURNSTILE_SECRET` | Worker secret, from the Turnstile widget |
| Security headers (CSP etc.) for pages | `public/_headers` |
| Turnstile site key (public) | `public/app.js` |
| Firewall rate limit (identify: 5 per 10 s per IP, block 10 s) | Cloudflare dashboard, see below |

The Worker's own rate limiter is approximate (counts are per machine and
shared with a delay), so it only slows sustained floods. The strict limit
is a Cloudflare firewall rate-limiting rule on the `dnabarcode.org` zone
(the Free plan allows one): Security > Security rules > Rate limiting rule
"Limit identify": URI Path equals `/api/identify`; same IP; more than 5
requests per 10 seconds; Block for 10 seconds.

The Turnstile widget ("dnabarcode.org", Managed mode, hostname
`dnabarcode.org` only) was created in the dashboard; its secret key was set
with `wrangler secret put TURNSTILE_SECRET`.

Until both secrets are set, `/api/identify` answers 503.

```bash
# Tests (Node 22+)
node --test sites/dnabarcode/test/*.test.js
# Deploy, from the repository root
npx --yes wrangler@4.162.0 deploy --config sites/dnabarcode/wrangler.jsonc
# Set the API key without showing it
gcloud secrets versions access latest --secret=dnabarcode-api-key \
    | npx --yes wrangler@4.162.0 secret put API_KEY --config sites/dnabarcode/wrangler.jsonc
```
