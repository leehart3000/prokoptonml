// Tests for the API gateway, with Cloudflare's bindings and fetch replaced
// by fakes. Run: node --test sites/dnabarcode/test/*.test.js
import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";

import worker, { MAX_BODY_BYTES } from "../src/index.js";

const API_URL = "https://api.example";
let calls;
let allowed;
let turnstile;
let api;

function env(overrides = {}) {
  const limiter = { limit: async () => ({ success: allowed }) };
  return {
    API_URL,
    API_KEY: "test-key",
    TURNSTILE_SECRET: "test-secret",
    EXPECTED_HOSTNAME: "dnabarcode.org",
    TURNSTILE_ACTION: "identify",
    IDENTIFY_LIMITER: limiter,
    HEALTH_LIMITER: limiter,
    ASSETS: { fetch: async () => new Response("page") },
    ...overrides,
  };
}

function post(body, headers = {}) {
  return new Request("https://dnabarcode.org/api/identify", {
    method: "POST",
    headers: { "Content-Type": "application/json", "CF-Connecting-IP": "192.0.2.1", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

const GOOD = { sequence: ">x\nACGT", turnstileToken: "token" };

beforeEach(() => {
  calls = [];
  allowed = true;
  turnstile = { success: true, hostname: "dnabarcode.org", action: "identify" };
  api = () => new Response(JSON.stringify({ status: "identified" }), { status: 200 });
  globalThis.fetch = async (url, init = {}) => {
    calls.push({ url: String(url), init });
    if (String(url).includes("siteverify")) return new Response(JSON.stringify(turnstile));
    return api(url, init);
  };
});

afterEach(() => {
  delete globalThis.fetch;
});

test("static pages come from the assets", async () => {
  const response = await worker.fetch(new Request("https://dnabarcode.org/"), env());
  assert.equal(await response.text(), "page");
});

test("unknown API paths are not found", async () => {
  const response = await worker.fetch(new Request("https://dnabarcode.org/api/x"), env());
  assert.equal(response.status, 404);
});

test("a good request is forwarded with the key and only the sequence", async () => {
  const response = await worker.fetch(post(GOOD), env());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { status: "identified" });
  const forwarded = calls.find((c) => c.url === `${API_URL}/identify`);
  assert.equal(forwarded.init.headers["X-Prokopton-Key"], "test-key");
  assert.deepEqual(JSON.parse(forwarded.init.body), { sequence: GOOD.sequence });
  const verify = calls.find((c) => c.url.includes("siteverify"));
  assert.equal(verify.init.body.get("secret"), "test-secret");
  assert.equal(verify.init.body.get("remoteip"), "192.0.2.1");
});

test("identify needs POST", async () => {
  const response = await worker.fetch(new Request("https://dnabarcode.org/api/identify"), env());
  assert.equal(response.status, 405);
  assert.equal(response.headers.get("Allow"), "POST");
});

test("too many requests are refused before anything else", async () => {
  allowed = false;
  const response = await worker.fetch(post(GOOD), env());
  assert.equal(response.status, 429);
  assert.equal(response.headers.get("Retry-After"), "60");
  assert.equal(calls.length, 0);
});

test("only JSON is accepted", async () => {
  const response = await worker.fetch(post("x", { "Content-Type": "text/plain" }), env());
  assert.equal(response.status, 415);
});

test("broken JSON is refused", async () => {
  const response = await worker.fetch(post("{"), env());
  assert.equal(response.status, 400);
});

test("long submissions are refused, by header or by size", async () => {
  const big = { ...GOOD, sequence: "A".repeat(MAX_BODY_BYTES) };
  const byHeader = post(GOOD, { "Content-Length": String(MAX_BODY_BYTES + 1) });
  assert.equal((await worker.fetch(byHeader, env())).status, 413);
  assert.equal((await worker.fetch(post(big), env())).status, 413);
});

test("a sequence and a token are required", async () => {
  for (const body of [{ turnstileToken: "t" }, { sequence: " ", turnstileToken: "t" }, null]) {
    assert.equal((await worker.fetch(post(body), env())).status, 400);
  }
  const noToken = await worker.fetch(post({ sequence: "ACGT" }), env());
  assert.equal(noToken.status, 400);
  assert.match((await noToken.json()).error, /human/);
});

test("without its secrets the gateway is closed", async () => {
  for (const missing of [{ API_KEY: undefined }, { TURNSTILE_SECRET: "" }]) {
    assert.equal((await worker.fetch(post(GOOD), env(missing))).status, 503);
  }
  assert.equal(calls.length, 0);
});

test("a failed, foreign or wrong-action Turnstile token is refused", async () => {
  for (const outcome of [
    { success: false },
    { success: true, hostname: "evil.example", action: "identify" },
    { success: true, hostname: "dnabarcode.org", action: "other" },
  ]) {
    turnstile = outcome;
    assert.equal((await worker.fetch(post(GOOD), env())).status, 403);
  }
  globalThis.fetch = async () => {
    throw new Error("down");
  };
  assert.equal((await worker.fetch(post(GOOD), env())).status, 403);
});

test("the service's own errors are passed on, but key errors are hidden", async () => {
  api = () => new Response(JSON.stringify({ detail: "too long" }), { status: 422 });
  const invalid = await worker.fetch(post(GOOD), env());
  assert.equal(invalid.status, 422);
  assert.deepEqual(await invalid.json(), { detail: "too long" });
  api = () => new Response(JSON.stringify({ detail: "Missing or wrong API key" }), { status: 401 });
  const keyError = await worker.fetch(post(GOOD), env());
  assert.equal(keyError.status, 502);
  assert.doesNotMatch(JSON.stringify(await keyError.json()), /key/i);
});

test("timeouts and unreachable services give clear errors", async () => {
  const timeout = new Error("slow");
  timeout.name = "TimeoutError";
  api = () => {
    throw timeout;
  };
  assert.equal((await worker.fetch(post(GOOD), env())).status, 504);
  api = () => {
    throw new Error("down");
  };
  assert.equal((await worker.fetch(post(GOOD), env())).status, 502);
});

test("health wakes the service and reports its state", async () => {
  const get = () => new Request("https://dnabarcode.org/api/health");
  api = () => new Response("{}", { status: 200 });
  assert.equal((await worker.fetch(get(), env())).status, 200);
  assert.equal(calls[0].url, `${API_URL}/health`);
  api = () => new Response("{}", { status: 500 });
  assert.equal((await worker.fetch(get(), env())).status, 503);
  api = () => {
    throw new Error("down");
  };
  assert.equal((await worker.fetch(get(), env())).status, 503);
  allowed = false;
  assert.equal((await worker.fetch(get(), env())).status, 429);
  const posted = new Request("https://dnabarcode.org/api/health", { method: "POST" });
  assert.equal((await worker.fetch(posted, env())).status, 405);
});
