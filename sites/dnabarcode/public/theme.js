// Applies a saved light/dark choice before the page is drawn (no flash).
// Without a saved choice, the page follows the system setting.
try {
  const saved = localStorage.getItem("theme");
  if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
} catch {
  // Storage can be unavailable (private modes); the system setting applies
}
