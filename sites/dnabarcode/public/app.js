// dnabarcode.org identify page. No inline scripts (see the CSP in _headers).
"use strict";

const TURNSTILE_SITE_KEY = "0x4AAAAAAGAqDhdbjAKn0ZJt";
const SLOW_NOTICE_MS = 4000;

const form = document.getElementById("identify");
const input = document.getElementById("sequence");
const submit = document.getElementById("submit");
const exampleButton = document.getElementById("example");
const statusLine = document.getElementById("status");
const output = document.getElementById("result");
let widgetId = null;

// Wake the service as soon as the page opens, so the first identification
// does not wait for it to start
fetch("/api/health").catch(() => {});

window.onTurnstileLoad = () => {
  widgetId = window.turnstile.render("#turnstile", {
    sitekey: TURNSTILE_SITE_KEY,
    action: "identify",
    callback: () => setStatus(""),
    "error-callback": () => setStatus("The check that you are human could not load. Please reload the page."),
  });
};

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
  const token = widgetId === null ? "" : window.turnstile.getResponse(widgetId);
  if (!token) {
    setStatus("Please complete the check that you are human, then try again.");
    return;
  }
  submit.disabled = true;
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
    submit.disabled = false;
    // Each token works once: get a fresh one for the next identification
    if (widgetId !== null) window.turnstile.reset(widgetId);
  }
});
