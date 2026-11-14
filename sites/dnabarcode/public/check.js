// Quick checks of a pasted sequence, shown as the visitor types. The
// service still makes the final decision.

export const MIN_BASES = 100;
export const SHORT_BASES = 300;
export const MAX_BASES = 20_000;

export function inspect(text) {
  const trimmed = text.trim();
  if (!trimmed) return { level: "", message: "" };
  if (trimmed.startsWith("@")) {
    return { level: "bad", message: "This looks like FASTQ reads. Paste one assembled sequence instead." };
  }
  const lines = trimmed.split(/\r?\n/);
  const headers = lines.filter((line) => line.startsWith(">")).length;
  if (headers > 1) {
    return { level: "bad", message: `This has ${headers} sequences. Paste one at a time.` };
  }
  const bases = lines.filter((line) => !line.startsWith(">")).join("").replace(/\s+/g, "");
  if (/[EFIJLOPQZ]/i.test(bases)) {
    return { level: "bad", message: "This looks like a protein sequence. Paste the DNA sequence instead." };
  }
  const count = bases.length;
  const n = count.toLocaleString("en-GB");
  if (count < MIN_BASES) {
    return { level: "bad", message: `${n} bases: too short. At least ${MIN_BASES} are needed.` };
  }
  if (count > MAX_BASES) {
    return { level: "bad", message: `${n} bases: too long. The limit is ${MAX_BASES.toLocaleString("en-GB")}.` };
  }
  if (count < SHORT_BASES) {
    return { level: "warn", message: `${n} bases. Under ${SHORT_BASES}, answers are less certain.` };
  }
  return { level: "ok", message: `${n} bases.` };
}

