// API gateway for dnabarcode.org.
//
// The browser never sees the API key. For each identification the Worker:
// 1. limits how often each visitor (IP address) may call,
// 2. checks the request's size and shape,
// 3. checks a Cloudflare Turnstile token (one per submission),
// 4. forwards only the sequence to the Cloud Run service, adding the key.
// Logs never contain sequences.

export const MAX_BODY_BYTES = 40_000; // The service accepts 30,000 characters
export const API_TIMEOUT_MS = 90_000;
const HEALTH_TIMEOUT_MS = 30_000;
const SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

export default {
  async fetch(request, env) {
    const { pathname } = new URL(request.url);
    if (pathname === "/api/health") return health(request, env);
    if (pathname === "/api/identify") return identify(request, env);
    if (pathname.startsWith("/api/")) return json({ error: "Not found" }, 404);
    return env.ASSETS.fetch(request);
  },
};

function json(body, status = 200, headers = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      ...headers,
    },
  });
}

function visitor(request) {
  return request.headers.get("CF-Connecting-IP") ?? "unknown";
}

function log(event, fields) {
  console.log(JSON.stringify({ event, ...fields }));
}

// Wakes the service (the site calls this when a page opens, so the first
// identification does not wait for a cold start)
async function health(request, env) {
  if (request.method !== "GET") {
    return json({ error: "Use GET" }, 405, { Allow: "GET" });
  }
  const { success } = await env.HEALTH_LIMITER.limit({ key: visitor(request) });
  if (!success) return json({ error: "Too many requests" }, 429, { "Retry-After": "60" });
  try {
    const response = await fetch(`${env.API_URL}/health`, {
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    return response.ok
      ? json({ status: "ok" })
      : json({ status: "unavailable" }, 503);
  } catch {
    return json({ status: "unavailable" }, 503);
  }
}

async function identify(request, env) {
  const started = Date.now();
  if (request.method !== "POST") {
    return json({ error: "Use POST" }, 405, { Allow: "POST" });
  }
  const { success } = await env.IDENTIFY_LIMITER.limit({ key: visitor(request) });
  if (!success) {
    log("identify", { outcome: "rate_limited" });
    return json(
      { error: "Too many identifications. Please wait a minute and try again." },
      429,
      { "Retry-After": "60" },
    );
  }
  if (!(request.headers.get("Content-Type") ?? "").startsWith("application/json")) {
    return json({ error: "Send JSON" }, 415);
  }
  if (Number(request.headers.get("Content-Length") ?? 0) > MAX_BODY_BYTES) {
    return json({ error: "The submission is too long" }, 413);
  }
  const text = await request.text();
  if (new TextEncoder().encode(text).length > MAX_BODY_BYTES) {
    return json({ error: "The submission is too long" }, 413);
  }
  let body;
  try {
    body = JSON.parse(text);
  } catch {
    return json({ error: "Send JSON" }, 400);
  }
  const sequence = body?.sequence;
  const token = body?.turnstileToken;
  if (typeof sequence !== "string" || sequence.trim() === "") {
    return json({ error: "Paste a sequence" }, 400);
  }
  if (typeof token !== "string" || token === "") {
    return json({ error: "Please complete the check that you are human" }, 400);
  }
  if (!env.API_KEY || !env.TURNSTILE_SECRET) {
    log("identify", { outcome: "not_configured" });
    return json({ error: "The service is not available yet" }, 503);
  }
  if (!(await humanCheckPasses(token, request, env))) {
    log("identify", { outcome: "turnstile_failed" });
    return json({ error: "The check that you are human failed. Please try again." }, 403);
  }

  let response;
  try {
    response = await fetch(`${env.API_URL}/identify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Prokopton-Key": env.API_KEY },
      body: JSON.stringify({ sequence }),
      signal: AbortSignal.timeout(API_TIMEOUT_MS),
    });
  } catch (error) {
    const timedOut = error?.name === "TimeoutError";
    log("identify", { outcome: timedOut ? "timeout" : "unreachable", ms: Date.now() - started });
    return timedOut
      ? json({ error: "The identification took too long. Please try again." }, 504)
      : json({ error: "The service could not be reached. Please try again." }, 502);
  }
  log("identify", { outcome: "forwarded", status: response.status, ms: Date.now() - started });
  if (response.ok || response.status === 413 || response.status === 422) {
    // The service's own answer, or its explanation of what was wrong
    return json(await response.json(), response.status);
  }
  // 401/403 mean the gateway's key is wrong; never tell the visitor more
  return json({ error: "The service had a problem. Please try again later." }, 502);
}

async function humanCheckPasses(token, request, env) {
  const form = new FormData();
  form.append("secret", env.TURNSTILE_SECRET);
  form.append("response", token);
  form.append("remoteip", visitor(request));
  try {
    const response = await fetch(SITEVERIFY, { method: "POST", body: form });
    const outcome = await response.json();
    return (
      outcome.success === true &&
      outcome.hostname === env.EXPECTED_HOSTNAME &&
      outcome.action === env.TURNSTILE_ACTION
    );
  } catch {
    return false;
  }
}
