export function hsvToRgb(hue: number, saturation: number, brightness: number): number[] {
  const channelValue = (channelOffset: number) => {
    const sector = (channelOffset + hue / 60) % 6;
    const saturationFactor = Math.max(0, Math.min(sector, 4 - sector, 1));
    return Math.round(255 * brightness * (1 - saturation * saturationFactor));
  };
  return [channelValue(5), channelValue(3), channelValue(1)];
}

export function rgbToHsv(rgb: readonly number[]): number[] {
  const [r = 0, g = 0, b = 0] = rgb.map(v => v / 255);
  const maximum = Math.max(r, g, b);
  const minimum = Math.min(r, g, b);
  const range = maximum - minimum;
  let hueSector = 0;
  if (range !== 0) {
    if (maximum === r) {
      hueSector = ((g - b) / range + 6) % 6;
    } else if (maximum === g) {
      hueSector = (b - r) / range + 2;
    } else {
      hueSector = (r - g) / range + 4;
    }
  }
  const saturation = maximum === 0 ? 0 : range / maximum;
  return [hueSector * 60, saturation, maximum];
}

export function parseCssColor(value: string): { rgb: number[]; alpha: number } | null {
  // Context-dependent values have no single colour outside a stylesheet.
  const hasContextDependentColor = /\b(var|env|currentcolor|inherit|initial|unset|revert|light-dark)\b/i.test(value);
  if (!CSS.supports("color", value) || hasContextDependentColor) {
    return null;
  }

  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 1;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) {
    throw new Error("Colour parsing is unavailable");
  }

  context.fillStyle = value;
  context.fillRect(0, 0, 1, 1);
  const [r = 0, g = 0, b = 0, a = 255] = context.getImageData(0, 0, 1, 1).data;
  return { rgb: [r, g, b], alpha: a / 255 };
}
