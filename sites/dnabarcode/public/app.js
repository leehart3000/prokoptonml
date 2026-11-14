// dnabarcode.org identify page. No inline scripts (see the CSP in _headers).
import { inspect } from "./check.js";

const TURNSTILE_SITE_KEY = "0x4AAAAAAGAqDhdbjAKn0ZJt";
const SLOW_NOTICE_MS = 4000;
const TOKEN_TIMEOUT_MS = 60_000;

const form = document.getElementById("identify");
const input = document.getElementById("sequence");
const submit = document.getElementById("submit");
const exampleButton = document.getElementById("example");
const statusLine = document.getElementById("status");
const output = document.getElementById("result");
const check = document.getElementById("sequence-check");
let widgetId = null;
let pendingToken = null;

// Wake the service as soon as the page opens, so the first identification
// does not wait for it to start
fetch("/api/health").catch(() => {});

// Turnstile runs only when Identify is pressed, and stays invisible unless
// Cloudflare needs the visitor to tick a box
window.onTurnstileLoad = () => {
  widgetId = window.turnstile.render("#turnstile", {
    sitekey: TURNSTILE_SITE_KEY,
    action: "identify",
    execution: "execute",
    appearance: "interaction-only",
    callback: (token) => settleToken(null, token),
    "error-callback": () => settleToken(new Error("turnstile")),
    "expired-callback": () => settleToken(new Error("expired")),
  });
};

function settleToken(error, token) {
  if (!pendingToken) return;
  const { resolve, reject, timer } = pendingToken;
  pendingToken = null;
  clearTimeout(timer);
  if (error) reject(error);
  else resolve(token);
}

function getToken() {
  return new Promise((resolve, reject) => {
    if (widgetId === null || !window.turnstile) {
      reject(new Error("not loaded"));
      return;
    }
    const timer = setTimeout(() => settleToken(new Error("timeout")), TOKEN_TIMEOUT_MS);
    pendingToken = { resolve, reject, timer };
    window.turnstile.execute(widgetId);
  });
}

function updateCheck() {
  const { level, message } = inspect(input.value);
  check.textContent = message;
  check.className = level ? `check ${level}` : "check";
  submit.disabled = level === "bad";
}

input.addEventListener("input", updateCheck);

function setStatus(text) {
  statusLine.textContent = text;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function show(kind, build) {
  output.replaceChildren();
  output.className = `result ${kind}`;
  build(output);
  output.hidden = false;
  output.focus();
}

function showError(message) {
  show("error", (box) => box.append(element("p", message, "answer")));
}

function showResult(result) {
  if (result.status === "identified" && result.answer) {
    show("identified", (box) => {
      box.append(element("p", result.answer.text, "answer"));
      const facts = [`${result.length.toLocaleString()} bases`];
      if (result.strand === "reverse") facts.push("read from the reverse strand");
      if (result.region) {
        facts.push(`barcode region found at ${result.region.start.toLocaleString()}–${result.region.end.toLocaleString()}`);
      }
      facts.push(`${result.seconds.toFixed(1)} s`);
      box.append(element("p", facts.join(" · "), "meta"));
      appendNotes(box, result.notes);
      appendTrace(box, result.trace);
    });
  } else {
    show("declined", (box) => {
      box.append(element("p", "Not identified", "answer"));
      box.append(element("p", result.message || result.reason || "The sequence could not be identified."));
      const help = element("p", "", "meta");
      const link = element("a", "What can I submit?");
      link.href = "/submit";
      help.append("See ", link, ".");
      box.append(help);
      appendNotes(box, result.notes);
      appendTrace(box, result.trace);
    });
  }
}

function appendNotes(box, notes) {
  if (!notes || notes.length === 0) return;
  const list = element("ul");
  for (const note of notes) list.append(element("li", note));
  box.append(list);
}

function appendTrace(box, trace) {
  if (!trace || trace.length === 0) return;
  const details = element("details");
  details.open = true;
  details.append(element("summary", "How this answer was reached"));
  const steps = element("ol", undefined, "trace");
  for (const step of trace) {
    const item = element("li");
    item.append(element("span", `${step.name} `, "step"), step.summary);
    steps.append(item);
  }
  details.append(steps);
  box.append(details);
}

async function errorMessage(response) {
  if (response.status === 429) {
    return "Too many identifications in a short time. Please wait a few seconds and try again.";
  }
  try {
    const body = await response.json();
    if (typeof body.error === "string") return body.error;
    if (Array.isArray(body.detail)) return "The submission was not accepted. Please check its length and format.";
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // Not JSON (for example a firewall page)
  }
  return "Something went wrong. Please try again.";
}

exampleButton.addEventListener("click", async () => {
  const response = await fetch("/examples/aedes-aegypti-HQ945416.fasta");
  input.value = (await response.text()).trim();
  updateCheck();
  input.focus();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const sequence = input.value.trim();
  if (!sequence) {
    setStatus("Paste a sequence first.");
    input.focus();
    return;
  }
  if (inspect(sequence).level === "bad") {
    input.focus();
    return;
  }
  submit.disabled = true;
  form.setAttribute("aria-busy", "true");
  setStatus("Checking that you are human…");
  let token;
  try {
    token = await getToken();
  } catch {
    setStatus("");
    showError("The check that you are human did not complete. Please try again, or reload the page.");
    form.removeAttribute("aria-busy");
    updateCheck();
    if (widgetId !== null) window.turnstile.reset(widgetId);
    return;
  }
  setStatus("Identifying…");
  const slow = setTimeout(
    () => setStatus("Identifying… The service may be waking up, which can take up to half a minute."),
    SLOW_NOTICE_MS,
  );
  try {
    const response = await fetch("/api/identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequence, turnstileToken: token }),
    });
    if (response.ok) {
      showResult(await response.json());
    } else {
      showError(await errorMessage(response));
    }
  } catch {
    showError("The service could not be reached. Please check your connection and try again.");
  } finally {
    clearTimeout(slow);
    setStatus("");
    form.removeAttribute("aria-busy");
    updateCheck();
    // Each token works once: get a fresh one for the next identification
    if (widgetId !== null) window.turnstile.reset(widgetId);
  }
});
