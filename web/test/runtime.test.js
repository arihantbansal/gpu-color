import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ColorModel, labToRgb, parseMetadata, tokenize } from "../src/color/runtime";

const metadata = parseMetadata(JSON.parse(readFileSync("public/color/model.json", "utf8")));
const bytes = new Uint8Array(readFileSync("public/color/model.bin")).buffer;
const model = new ColorModel(metadata, bytes);
const fixtures = JSON.parse(readFileSync("public/color/fixtures.json", "utf8"));

describe("exported Python contract", () => {
  for (const fixture of fixtures) {
    it(`matches ${JSON.stringify(fixture.phrase)}`, () => {
      expect(tokenize(fixture.phrase, metadata.buckets)).toEqual(fixture.ids);
      const actual = model.predict(fixture.phrase);
      actual.forEach((value, i) => expect(Math.abs(value - fixture.lab[i])).toBeLessThan(0.001));
      expect(labToRgb(actual)).toEqual(fixture.rgb);
    });
  }
  it("fits the complete model in 50 KB", () => {
    const size = bytes.byteLength + readFileSync("public/color/model.json").byteLength;
    expect(size).toBeLessThan(50_000);
    expect(metadata.metrics.model_bytes).toBe(size);
  });
  it("rejects truncated or mismatched weights", () => {
    expect(() => new ColorModel(metadata, bytes.slice(1))).toThrow();
    expect(() => new ColorModel({ ...metadata, dim: metadata.dim + 1 }, bytes)).toThrow();
    expect(() => parseMetadata({ ...metadata, version: 2 })).toThrow();
  });
});
