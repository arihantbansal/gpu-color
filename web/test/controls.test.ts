import { expect, it } from "vitest";
import { hsvToRgb, rgbToHsv } from "../src/color/controls";

it("round-trips display colours through manual controls", () => {
  for (const rgb of [[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [192, 115, 122], [25, 26, 26]]) {
    const [h = 0, s = 0, v = 0] = rgbToHsv(rgb);
    expect(hsvToRgb(h, s, v)).toEqual(rgb);
  }
});

it("keeps hue endpoints equivalent", () => {
  expect(hsvToRgb(0, 1, 1)).toEqual([255, 0, 0]);
  expect(hsvToRgb(360, 1, 1)).toEqual([255, 0, 0]);
});
