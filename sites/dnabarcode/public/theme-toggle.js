// The sun/moon button in the header: switches between light and dark mode
// and remembers the choice in this browser.
"use strict";
(() => {
  const button = document.getElementById("theme-toggle");
  const root = document.documentElement;
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

  const current = () => root.dataset.theme || (systemDark.matches ? "dark" : "light");

  function label() {
    const next = current() === "dark" ? "light" : "dark";
    button.setAttribute("aria-label", `Switch to ${next} mode`);
    button.title = `Switch to ${next} mode`;
  }

  button.addEventListener("click", () => {
    const next = current() === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {
      // Not remembered, but still switched for this page
    }
    label();
  });
  systemDark.addEventListener("change", label);
  label();
})();
