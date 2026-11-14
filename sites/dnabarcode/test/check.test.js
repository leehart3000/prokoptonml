// Tests for the identify page's quick sequence checks.
import assert from "node:assert/strict";
import { test } from "node:test";

import { inspect } from "../public/check.js";

const level = (text) => inspect(text).level;

test("an empty box shows nothing", () => {
  assert.deepEqual(inspect("  \n "), { level: "", message: "" });
});

test("a normal barcode is fine, with or without a FASTA header", () => {
  assert.equal(level("ACGT".repeat(165)), "ok");
  assert.equal(inspect(">x\r\nACGT ACGT\n" + "ACGT".repeat(163)).message, "660 bases.");
});

test("short sequences warn, and very short or long ones are refused", () => {
  assert.equal(level("A".repeat(299)), "warn");
  assert.equal(level("A".repeat(99)), "bad");
  assert.equal(level("A".repeat(100)), "warn");
  assert.equal(level("A".repeat(20_000)), "ok");
  assert.equal(level("A".repeat(20_001)), "bad");
});

test("reads, several sequences and protein are refused with a reason", () => {
  assert.match(inspect("@read1\nACGT\n+\nIIII").message, /FASTQ/);
  assert.match(inspect(">a\nACGT\n>b\nACGT").message, /2 sequences/);
  assert.match(inspect("MKLVEFIQ".repeat(20)).message, /protein/);
});

test("ambiguity codes are not mistaken for protein", () => {
  assert.equal(level("ACGTRYSWKMBDHVN".repeat(10)), "warn");
});
