import "./style.css";
import { ColorModel, labToRgb, parseMetadata, toHex } from "./color/runtime";
import { hsvToRgb, parseCssColor, rgbToHsv } from "./color/controls";

const app = document.querySelector<HTMLElement>("#app")!;
const phraseInput = document.querySelector<HTMLInputElement>("#phrase")!;
const field = document.querySelector<HTMLElement>(".field")!;
const cursor = document.querySelector<HTMLElement>(".cursor")!;
const hueSlider = document.querySelector<HTMLInputElement>("#hue")!;
const opacitySlider = document.querySelector<HTMLInputElement>("#alpha")!;
const hexOutput = document.querySelector<HTMLInputElement>("#hex")!;
const statusMessage = document.querySelector<HTMLElement>("#status")!;
const copyButton = document.querySelector<HTMLButtonElement>("#copy")!;
const copyLabel = document.querySelector<HTMLElement>(".copy-label")!;
const fieldValue = document.querySelector<HTMLElement>("#field-value")!;

let model: ColorModel | null = null;
let hueDegrees = 240;
let saturation = 0.22;
let brightness = 0.14;
let opacity = 1;
let inputTimer: ReturnType<typeof setTimeout> | undefined;
let colorValue = "";
let copyFeedbackTimer: ReturnType<typeof setTimeout> | undefined;

function renderColor() {
  const rgb = hsvToRgb(hueDegrees, saturation, brightness);
  const hex = toHex(rgb);
  const alphaHex = Math.round(opacity * 255).toString(16).padStart(2, "0");
  colorValue = opacity === 1 ? hex : `${hex}${alphaHex}`;

  field.style.backgroundColor = `hsl(${hueDegrees} 100% 50%)`;
  cursor.style.left = `${saturation * 100}%`;
  cursor.style.top = `${(1 - brightness) * 100}%`;
  app.style.setProperty("--colour", hex);
  app.style.setProperty("--alpha", String(opacity));
  hueSlider.value = String(hueDegrees);
  opacitySlider.value = String(opacity * 100);
  hexOutput.value = colorValue;
  fieldValue.textContent = `Saturation ${Math.round(saturation * 100)}%, brightness ${Math.round(brightness * 100)}%.`;
  copyButton.classList.remove("copied");
  copyLabel.textContent = "Copy";
  clearTimeout(copyFeedbackTimer);
  hueSlider.setAttribute("aria-valuetext", `${Math.round(hueDegrees)} degrees`);
  opacitySlider.setAttribute("aria-valuetext", `${Math.round(opacity * 100)} percent`);
  copyButton.title = `Copy ${colorValue}`;
  copyButton.setAttribute("aria-label", `Copy ${colorValue}`);
}

function applyManualChange() {
  clearTimeout(inputTimer);
  renderColor();
  statusMessage.textContent = "";
}

function applyTextInput() {
  const text = phraseInput.value.trim();
  if (!text) {
    statusMessage.textContent = "";
    return;
  }
  const parsed = parseCssColor(text);
  let rgb: number[];
  if (parsed) {
    rgb = parsed.rgb;
    opacity = parsed.alpha;
  } else {
    if (/^[#]|[()]/.test(text) || /\b(currentcolor|inherit|initial|unset|revert)\b/i.test(text)) {
      statusMessage.textContent = "Enter a valid standalone CSS color or a short description.";
      return;
    }
    if (!/[a-z]{2}/i.test(text)) {
      statusMessage.textContent = "Try a short description in English.";
      return;
    }
    if (!model) {
      statusMessage.textContent = "Descriptions are unavailable. CSS colors still work.";
      return;
    }
    rgb = labToRgb(model.predict(text));
  }
  statusMessage.textContent = "";
  const hsv = rgbToHsv(rgb);
  if (hsv[1]! > 0) {
    hueDegrees = hsv[0]!;
  }
  saturation = hsv[1]!;
  brightness = hsv[2]!;
  renderColor();
}

phraseInput.addEventListener("input", () => {
  clearTimeout(inputTimer);
  inputTimer = setTimeout(applyTextInput, 140);
});

hueSlider.oninput = () => {
  hueDegrees = Number(hueSlider.value);
  applyManualChange();
};

opacitySlider.oninput = () => {
  opacity = Number(opacitySlider.value) / 100;
  applyManualChange();
};

field.onkeydown = event => {
  if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
    return;
  }
  event.preventDefault();
  const step = event.shiftKey ? 0.1 : 0.01;
  if (event.key === "ArrowLeft") {
    saturation = Math.max(0, saturation - step);
  }
  if (event.key === "ArrowRight") {
    saturation = Math.min(1, saturation + step);
  }
  if (event.key === "ArrowUp") {
    brightness = Math.min(1, brightness + step);
  }
  if (event.key === "ArrowDown") {
    brightness = Math.max(0, brightness - step);
  }
  applyManualChange();
};

for (const button of document.querySelectorAll<HTMLButtonElement>(".examples button")) {
  button.onclick = () => {
    clearTimeout(inputTimer);
    phraseInput.value = button.textContent!.trim();
    applyTextInput();
  };
}

function updateFromPointer(event: PointerEvent) {
  const rect = field.getBoundingClientRect();
  saturation = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  brightness = 1 - Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height));
  applyManualChange();
}

field.onpointerdown = event => {
  field.setPointerCapture(event.pointerId);
  updateFromPointer(event);
};

field.onpointermove = event => {
  if (field.hasPointerCapture(event.pointerId)) {
    updateFromPointer(event);
  }
};

copyButton.onclick = async () => {
  const copied = colorValue;
  try {
    await navigator.clipboard.writeText(copied);
    if (colorValue !== copied) {
      return;
    }
    copyButton.classList.add("copied");
    copyLabel.textContent = "Copied";
    statusMessage.textContent = `Copied ${copied}.`;
    clearTimeout(copyFeedbackTimer);
    copyFeedbackTimer = setTimeout(() => {
      copyButton.classList.remove("copied");
      copyLabel.textContent = "Copy";
      if (statusMessage.textContent === `Copied ${copied}.`) {
        statusMessage.textContent = "";
      }
    }, 1600);
  } catch {
    statusMessage.textContent = "Press ⌘C or Ctrl+C to copy the selected value.";
    hexOutput.focus();
    hexOutput.select();
  }
};

renderColor();

async function loadModel() {
  try {
    const [metadataResponse, weightsResponse] = await Promise.all([
      fetch(`${import.meta.env.BASE_URL}color/model.json`),
      fetch(`${import.meta.env.BASE_URL}color/model.bin`),
    ]);
    if (!metadataResponse.ok || !weightsResponse.ok) {
      throw new Error("Model download failed");
    }
    const metadata = parseMetadata(await metadataResponse.json());
    model = new ColorModel(metadata, await weightsResponse.arrayBuffer());
    if (phraseInput.value.trim()) {
      applyTextInput();
    } else {
      statusMessage.textContent = "";
    }
  } catch (error) {
    statusMessage.textContent = "Model could not load. CSS colors still work. Reload to retry.";
    console.error(error);
  } finally {
    document.querySelector(".loading")!.remove();
  }
}
void loadModel();
