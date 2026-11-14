// Browser tests of the live site (or another address, with SITE_URL).
// Run from sites/dnabarcode: npx playwright test
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 45_000,
  retries: process.env.CI ? 1 : 0,
  reporter: "list",
  use: {
    baseURL: process.env.SITE_URL ?? "https://dnabarcode.org",
    colorScheme: "light",
    trace: "retain-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], colorScheme: "light" } },
    // iPhone SE size and touch, in Chromium (the only browser installed)
    { name: "iphone-se", use: { ...devices["iPhone SE"], browserName: "chromium", colorScheme: "light" } },
  ],
});
