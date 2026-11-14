// Browser tests of DNAbarcode.org. Identifications are tested with the
// service and Turnstile replaced by fakes (a real identification needs a
// human check), and the live gateway is checked separately.
import { expect, test } from "@playwright/test";

const PAGES = [
  ["/", "DNAbarcode.org"],
  ["/submit", "What can I submit? · DNAbarcode.org"],
  ["/finding-coi", "Finding the COI region · DNAbarcode.org"],
  ["/scope", "Scope and limits · DNAbarcode.org"],
];

const FAKE_TURNSTILE = `
  window.turnstile = {
    render: (element, options) => { window.__turnstile = options; return 0; },
    execute: () => setTimeout(() => window.__turnstile.callback("test-token"), 10),
    reset: () => {},
  };
  window.onTurnstileLoad();
`;

const ANSWER = {
  status: "identified",
  answer: { level: "species", name: "Aedes aegypti", text: "Aedes aegypti (100.0% identical).", confident: true },
  length: 658,
  strand: "given",
  region: null,
  notes: [],
  trace: [
    { name: "input", summary: "One sequence of 658 bases.", seconds: 0 },
    { name: "marker", summary: "CO1 (probability 1.00), read on the given strand.", seconds: 0.1 },
    { name: "router", summary: "Diptera (probability 0.93).", seconds: 0.1 },
    { name: "specialist:culicidae", summary: "Species, Aedes aegypti, best identity 100.0%.", seconds: 0.8 },
  ],
  seconds: 1.04,
};

// True only for scripts served by Turnstile itself (an exact host match)
function fromTurnstile(url) {
  try {
    return new URL(url).hostname === "challenges.cloudflare.com";
  } catch {
    return false;
  }
}

// Console errors from our own pages (Turnstile's own messages are ignored)
function watchErrors(page) {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && !fromTurnstile(message.location().url)) {
      errors.push(message.text());
    }
  });
  return errors;
}

async function fakeService(page, identify) {
  await page.route("https://challenges.cloudflare.com/**", (route) =>
    route.fulfill({ contentType: "text/javascript", body: FAKE_TURNSTILE }),
  );
  await page.route("**/api/health", (route) => route.fulfill({ json: { status: "ok" } }));
  await page.route("**/api/identify", identify);
}

// The example is fetched, so wait until it is in the box before going on
async function loadExample(page) {
  await page.getByRole("button", { name: "Load an example" }).click();
  await expect(page.locator("#sequence-check")).toHaveText("658 bases.");
}

for (const [path, title] of PAGES) {
  test(`${path} loads cleanly and fits the screen`, async ({ page }) => {
    const errors = watchErrors(page);
    await fakeService(page, (route) => route.abort());
    const response = await page.goto(path);
    expect(response.status()).toBe(200);
    expect(response.headers()["content-security-policy"]).toContain("frame-ancestors 'none'");
    await expect(page).toHaveTitle(title);
    await expect(page.getByRole("link", { name: "DNAbarcode.org home" })).toBeVisible();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
    expect(overflow).toBeLessThanOrEqual(0);
    expect(errors).toEqual([]);
  });
}

test("the navigation marks the current page", async ({ page }) => {
  await fakeService(page, (route) => route.abort());
  await page.goto("/");
  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Scope and limits" }).click();
  await expect(page).toHaveURL(/\/scope$/);
  await expect(page.getByRole("link", { name: "Scope and limits" })).toHaveAttribute("aria-current", "page");
});

test("external links open in a new tab and say so", async ({ page }) => {
  await page.goto("/scope");
  const external = page.locator('a[target="_blank"]');
  expect(await external.count()).toBeGreaterThan(0);
  for (const link of await external.all()) {
    await expect(link).toHaveAttribute("rel", /noopener/);
    await expect(link).toContainText("(opens in a new tab)");
  }
});

test("the light/dark choice is remembered across pages", async ({ page }) => {
  await fakeService(page, (route) => route.abort());
  await page.goto("/");
  await page.getByRole("button", { name: "Switch to dark mode" }).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.goto("/submit");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  await expect(page.getByRole("button", { name: "Switch to light mode" })).toBeVisible();
});

test("the sequence is checked while typing", async ({ page }) => {
  await fakeService(page, (route) => route.abort());
  await page.goto("/");
  const identify = page.getByRole("button", { name: "Identify" });
  await page.getByLabel("DNA sequence (FASTA or plain)").fill("ACGTACGT");
  await expect(page.locator("#sequence-check")).toContainText("too short");
  await expect(identify).toBeDisabled();
  await page.getByRole("button", { name: "Load an example" }).click();
  await expect(page.locator("#sequence-check")).toHaveText("658 bases.");
  await expect(identify).toBeEnabled();
});

test("an identification shows the answer and how it was reached", async ({ page }) => {
  const errors = watchErrors(page);
  let sent;
  await fakeService(page, (route) => {
    sent = route.request().postDataJSON();
    return route.fulfill({ json: ANSWER });
  });
  await page.goto("/");
  await loadExample(page);
  await page.getByRole("button", { name: "Identify" }).click();
  const result = page.locator("#result");
  await expect(result).toHaveClass(/identified/);
  await expect(result.locator(".answer")).toHaveText("Aedes aegypti (100.0% identical).");
  await expect(result.getByRole("heading", { name: "How this answer was reached" })).toBeVisible();
  await expect(result.locator("ol.trace li")).toHaveCount(4);
  expect(sent.turnstileToken).toBe("test-token");
  expect(sent.sequence).toContain(">HQ945416.1");
  expect(errors).toEqual([]);
});

test("a refusal from the firewall is explained", async ({ page }) => {
  await fakeService(page, (route) => route.fulfill({ status: 429, contentType: "text/html", body: "<h1>Blocked</h1>" }));
  await page.goto("/");
  await loadExample(page);
  await page.getByRole("button", { name: "Identify" }).click();
  await expect(page.locator("#result")).toContainText("Too many identifications");
});

test.describe("live gateway", () => {
  test.skip(({ isMobile }) => isMobile, "Checked once, from the desktop project");

  test("the service is healthy (this also wakes it)", async ({ request }) => {
    const response = await request.get("/api/health", { timeout: 40_000 });
    expect(response.status()).toBe(200);
    expect(await response.json()).toEqual({ status: "ok" });
  });

  test("identifications need the human check", async ({ request }) => {
    const response = await request.post("/api/identify", { data: { sequence: "ACGT" } });
    expect(response.status()).toBe(400);
    expect((await response.json()).error).toMatch(/human/);
  });
});
