// Run against a running Vite server using an existing Playwright installation.
import assert from "node:assert/strict";
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || "playwright");
const url = process.env.DEMO_URL || "http://127.0.0.1:5173";
const browser = await chromium.launch({ channel: "chrome", headless: true });
const viewports = [{ width: 1440, height: 900 }, { width: 1366, height: 768 }, { width: 1280, height: 720 }, { width: 390, height: 844 }, { width: 375, height: 667 }];
async function waitForReady(page) {
  await page.waitForFunction(() => !document.querySelector(".loading"));
  assert.equal(await page.locator("#status").textContent(), "");
  assert.equal(await page.locator("#hex").isVisible(), true);
  assert.equal(await page.locator("#hex").getAttribute("readonly"), "");
  assert.equal(await page.locator(".copy-label").textContent().then(text => text.trim()), "Copy");
}
try {
  const errors = [];
  const context = await browser.newContext({ permissions: ["clipboard-read", "clipboard-write"] });
  const page = await context.newPage();
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(url);
  await waitForReady(page);

  await page.locator("#phrase").fill("honeydew");
  await page.waitForFunction(() => document.querySelector("#hex").value === "#f0fff0");
  assert.equal(await page.locator("#copy").getAttribute("title"), "Copy #f0fff0");
  assert.equal(await page.locator("#copy").getAttribute("aria-label"), "Copy #f0fff0");
  assert.equal(await page.locator("#hex").inputValue(), "#f0fff0");
  await page.locator("#phrase").fill("#ff000080");
  await page.waitForFunction(() => document.querySelector("#hex").value === "#ff000080");
  await page.locator("#phrase").fill("#not-a-colour");
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("valid standalone"));
  assert.equal(await page.locator("#copy").getAttribute("title"), "Copy #ff000080");

  await page.locator("#phrase").fill("dusty rose");
  await page.waitForFunction(() => document.querySelector("#status").textContent === "");
  const phrase = await page.locator("#phrase").inputValue();
  const beforeDescription = await page.locator("#copy").getAttribute("title");
  assert.match(beforeDescription, /^Copy #[0-9a-f]{6}(?:[0-9a-f]{2})?$/);

  const field = page.locator(".field");
  assert.equal(await field.getAttribute("tabindex"), "0");
  assert.match(await field.getAttribute("aria-label"), /saturation/i);
  assert.equal(await page.locator("#saturation").count(), 0);
  await field.focus();
  assert.equal(await page.evaluate(() => document.activeElement?.matches(".field")), true);
  const beforeSaturation = await page.locator("#hex").inputValue();
  await page.keyboard.press("ArrowRight");
  await page.waitForFunction(previous => document.querySelector("#hex").value !== previous, beforeSaturation);
  assert.equal(await page.locator("#phrase").inputValue(), phrase);
  assert.equal(await page.locator("#status").textContent(), "");
  assert.notEqual(await page.locator("#copy").getAttribute("title"), beforeDescription);
  for (const key of ["ArrowLeft", "Shift+ArrowUp", "ArrowDown"]) {
    const before = await page.locator("#hex").inputValue();
    await page.keyboard.press(key);
    await page.waitForFunction(previous => document.querySelector("#hex").value !== previous, before);
  }
  assert.equal(await page.locator("#phrase").inputValue(), phrase);

  const examples = page.locator(".examples button");
  assert.ok(await examples.count() >= 3);
  for (const label of ["dusty rose", "light blue", "dark blue"]) assert.ok(await examples.filter({ hasText: label }).count(), `missing example: ${label}`);
  await examples.filter({ hasText: "light blue" }).click();
  assert.equal(await page.locator("#phrase").inputValue(), "light blue");

  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await page.evaluate(() => scrollTo(0, 0));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false, `${viewport.width}x${viewport.height} horizontal overflow`);
    assert.equal(await page.evaluate(() => document.documentElement.scrollHeight > innerHeight), false, `${viewport.width}x${viewport.height} vertical overflow`);
    await field.focus();
    assert.equal(await page.evaluate(() => document.activeElement?.matches(".field")), true);
  }

  await page.setViewportSize(viewports[2]);
  const beforeCopyInput = await page.locator("#hex").inputValue();
  await page.locator("#phrase").fill("dusty rose");
  await page.waitForFunction(previous => document.querySelector("#hex").value !== previous, beforeCopyInput);
  const clipboardPhrase = await page.locator("#phrase").inputValue();
  await page.locator("#copy").click();
  assert.equal(await page.evaluate(() => navigator.clipboard.readText()), await page.locator("#hex").inputValue());
  assert.equal(await page.locator("#phrase").inputValue(), clipboardPhrase);
  const deniedContext = await browser.newContext();
  const deniedPage = await deniedContext.newPage();
  deniedPage.on("pageerror", error => errors.push(error.message));
  await deniedPage.addInitScript(() => {
    navigator.clipboard.writeText = async () => { throw new DOMException("Clipboard denied", "NotAllowedError"); };
  });
  await deniedPage.goto(url);
  await waitForReady(deniedPage);
  await deniedPage.locator("#phrase").fill("dusty rose");
  await deniedPage.waitForFunction(() => document.querySelector("#hex").value !== "#1c1c24");
  const deniedPhrase = await deniedPage.locator("#phrase").inputValue();
  const hex = await deniedPage.locator("#hex").inputValue();
  await deniedPage.locator("#copy").click();
  assert.equal(await deniedPage.locator("#phrase").inputValue(), deniedPhrase);
  await deniedPage.waitForFunction(() => document.activeElement?.id === "hex");
  assert.equal(await deniedPage.evaluate(() => { const input = document.querySelector("#hex"); return input === document.activeElement && input.selectionStart === 0 && input.selectionEnd === input.value.length; }), true);
  assert.equal(await deniedPage.locator("#hex").inputValue(), hex);
  await deniedContext.close();

  await page.setViewportSize({ width: 1280, height: 1000 });
  const gpuResult = await page.evaluate(async () => {
    const [{ ColorModel, parseMetadata }, { createGpuPredictor }] = await Promise.all([
      import("/src/color/runtime.ts"), import("/src/color/gpu.ts"),
    ]);
    const [metadataResponse, modelResponse, fixturesResponse] = await Promise.all([
      fetch("/color/model.json"), fetch("/color/model.bin"), fetch("/color/fixtures.json"),
    ]);
    const model = new ColorModel(parseMetadata(await metadataResponse.json()), await modelResponse.arrayBuffer());
    const fixtures = (await fixturesResponse.json()).slice(0, 100);
    const gpu = await createGpuPredictor(model);
    if (!gpu) return { available: false };
    try {
      const mismatches = [];
      for (const fixture of fixtures) {
        const actual = await gpu.predict(fixture.phrase);
        const error = Math.max(...fixture.lab.map((value, index) => Math.abs(value - actual[index])));
        if (!Number.isFinite(error) || error > 0.01) mismatches.push({ phrase: fixture.phrase, error });
      }
      return { available: true, count: fixtures.length, mismatches };
    } finally { gpu.destroy(); }
  });
  if (!gpuResult.available) console.log("WebGPU is unavailable; parity check skipped.");
  else {
    assert.equal(gpuResult.mismatches.length, 0, JSON.stringify(gpuResult.mismatches.slice(0, 3)));
    assert.equal(gpuResult.count, 100);
    console.log(`WebGPU parity passed for ${gpuResult.count} fixtures.`);
  }

  await page.route("**/color/model.bin", route => route.abort());
  await page.reload();
  await page.waitForFunction(() => document.querySelector("#status").textContent.includes("could not load") && !document.querySelector(".loading"));
  await page.locator("#phrase").fill("rebeccapurple");
  await page.waitForFunction(() => document.querySelector("#hex").value === "#663399");
  assert.equal(await page.locator("#status").textContent(), "");
  assert.deepEqual(errors, []);
  await context.close();
  console.log("UI, keyboard, clipboard, viewport, GPU parity, and load-failure checks passed.");
} finally { await browser.close(); }
