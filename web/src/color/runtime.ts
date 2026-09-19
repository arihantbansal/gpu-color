export interface Metadata {
  version: number;
  buckets: number;
  dim: number;
  hidden: number;
  tokenizer: string;
  lab_scale: number;
  tensors: { name: string; shape: number[]; offset: number; length: number; scale: number }[];
  metrics: { float_delta_e: number; int8_delta_e: number; weights_bytes: number; model_bytes: number };
}

export function parseMetadata(value: unknown): Metadata {
  if (!value || typeof value !== "object") {
    throw new Error("Invalid model metadata");
  }

  const metadata = value;
  if (!("version" in metadata) || metadata.version !== 1) {
    throw new Error("Unsupported model format");
  }
  if (
    !("buckets" in metadata) ||
    typeof metadata.buckets !== "number" ||
    !Number.isInteger(metadata.buckets) ||
    metadata.buckets <= 0 ||
    metadata.buckets > 65536
  ) {
    throw new Error("Unsupported model format");
  }
  if (
    !("dim" in metadata) ||
    typeof metadata.dim !== "number" ||
    !Number.isInteger(metadata.dim) ||
    metadata.dim <= 0 ||
    metadata.dim > 256
  ) {
    throw new Error("Unsupported model format");
  }
  if (
    !("hidden" in metadata) ||
    typeof metadata.hidden !== "number" ||
    !Number.isInteger(metadata.hidden) ||
    metadata.hidden <= 0 ||
    metadata.hidden > 256
  ) {
    throw new Error("Unsupported model format");
  }
  if (!("tokenizer" in metadata) || metadata.tokenizer !== "lower-word-boundary-trigrams-fnv1a32-utf8") {
    throw new Error("Unsupported model format");
  }
  if (
    !("lab_scale" in metadata) ||
    metadata.lab_scale !== 100 ||
    !("tensors" in metadata) ||
    !Array.isArray(metadata.tensors)
  ) {
    throw new Error("Unsupported model format");
  }
  if (!("metrics" in metadata) || !metadata.metrics || typeof metadata.metrics !== "object") {
    throw new Error("Unsupported model format");
  }

  const tensors = metadata.tensors.map((tensor: unknown) => {
    if (!tensor || typeof tensor !== "object") {
      throw new Error("Invalid weight descriptor");
    }
    if (!("name" in tensor) || typeof tensor.name !== "string") {
      throw new Error("Invalid weight descriptor");
    }
    if (!("shape" in tensor) || !Array.isArray(tensor.shape)) {
      throw new Error("Invalid weight descriptor");
    }
    if (!tensor.shape.every((dimension: unknown) => typeof dimension === "number" && Number.isInteger(dimension) && dimension > 0)) {
      throw new Error("Invalid weight descriptor");
    }
    if (
      !("offset" in tensor) ||
      typeof tensor.offset !== "number" ||
      !Number.isInteger(tensor.offset) ||
      tensor.offset < 0
    ) {
      throw new Error("Invalid weight descriptor");
    }
    if (
      !("length" in tensor) ||
      typeof tensor.length !== "number" ||
      !Number.isInteger(tensor.length) ||
      tensor.length <= 0
    ) {
      throw new Error("Invalid weight descriptor");
    }
    if (
      !("scale" in tensor) ||
      typeof tensor.scale !== "number" ||
      !Number.isFinite(tensor.scale) ||
      tensor.scale <= 0
    ) {
      throw new Error("Invalid weight descriptor");
    }
    return { name: tensor.name, shape: tensor.shape.map(Number), offset: tensor.offset, length: tensor.length, scale: tensor.scale };
  });

  const metrics = metadata.metrics;
  if (
    !("float_delta_e" in metrics) ||
    typeof metrics.float_delta_e !== "number" ||
    !Number.isFinite(metrics.float_delta_e)
  ) {
    throw new Error("Invalid model metrics");
  }
  if (
    !("int8_delta_e" in metrics) ||
    typeof metrics.int8_delta_e !== "number" ||
    !Number.isFinite(metrics.int8_delta_e)
  ) {
    throw new Error("Invalid model metrics");
  }
  if (
    !("weights_bytes" in metrics) ||
    typeof metrics.weights_bytes !== "number" ||
    !Number.isInteger(metrics.weights_bytes)
  ) {
    throw new Error("Invalid model metrics");
  }
  if (
    !("model_bytes" in metrics) ||
    typeof metrics.model_bytes !== "number" ||
    !Number.isInteger(metrics.model_bytes)
  ) {
    throw new Error("Invalid model metrics");
  }

  return {
    version: 1,
    buckets: metadata.buckets,
    dim: metadata.dim,
    hidden: metadata.hidden,
    tokenizer: metadata.tokenizer,
    lab_scale: 100,
    tensors,
    metrics: {
      float_delta_e: metrics.float_delta_e,
      int8_delta_e: metrics.int8_delta_e,
      weights_bytes: metrics.weights_bytes,
      model_bytes: metrics.model_bytes,
    },
  };
}

export function tokenize(phrase: string, buckets: number): number[] {
  const ids: number[] = [];
  const encoder = new TextEncoder();
  // Match Python str.split(), including its extra ASCII information separators.
  const words = phrase.toLowerCase().split(/[\p{White_Space}\u001c-\u001f]+/u).filter(Boolean);
  for (const word of words) {
    const characters = Array.from(`<${word}>`);
    for (let start = 0; start + 3 <= characters.length; start++) {
      let hash = 2166136261;
      const trigram = characters.slice(start, start + 3).join("");
      for (const byte of encoder.encode(trigram)) {
        hash = Math.imul(hash ^ byte, 16777619) >>> 0;
      }
      ids.push(hash % buckets);
    }
  }
  return ids;
}

export class ColorModel {
  readonly embedding: Float32Array;
  readonly w1: Float32Array;
  readonly b1: Float32Array;
  readonly w2: Float32Array;
  readonly b2: Float32Array;

  constructor(readonly metadata: Metadata, bytes: ArrayBuffer) {
    const expected = [
      { name: "embedding_bag.weight", shape: [metadata.buckets, metadata.dim] },
      { name: "fc1.weight", shape: [metadata.hidden, metadata.dim] },
      { name: "fc1.bias", shape: [metadata.hidden] },
      { name: "fc2.weight", shape: [3, metadata.hidden] },
      { name: "fc2.bias", shape: [3] },
    ];
    let end = 0;
    const weights = expected.map((expectedTensor, index) => {
      const tensor = metadata.tensors[index];
      const tensorLength = expectedTensor.shape.reduce((product, dimension) => product * dimension, 1);
      const hasExpectedDescriptor = tensor
        && tensor.name === expectedTensor.name
        && tensor.shape.join() === expectedTensor.shape.join()
        && tensor.offset === end
        && tensor.length === tensorLength
        && end + tensorLength <= bytes.byteLength;
      if (!hasExpectedDescriptor) {
        throw new Error("Model weights do not match metadata");
      }
      end += tensorLength;
      return Float32Array.from(new Int8Array(bytes, tensor.offset, tensorLength), value => value * tensor.scale);
    });
    if (metadata.tensors.length !== 5 || end !== bytes.byteLength) {
      throw new Error("Unexpected model weights");
    }
    // Each position was validated against the five required tensor descriptors above.
    this.embedding = weights[0]!;
    this.w1 = weights[1]!;
    this.b1 = weights[2]!;
    this.w2 = weights[3]!;
    this.b2 = weights[4]!;
  }

  embed(phrase: string): Float32Array<ArrayBuffer> {
    const { dim, buckets } = this.metadata;
    const ids = tokenize(phrase, buckets);
    const out = new Float32Array(dim);
    for (const id of ids) {
      for (let dimension = 0; dimension < dim; dimension++) {
        out[dimension] = out[dimension]! + this.embedding[id * dim + dimension]!;
      }
    }
    if (ids.length) {
      for (let dimension = 0; dimension < dim; dimension++) {
        out[dimension] = out[dimension]! / ids.length;
      }
    }
    return out;
  }

  predict(phrase: string): number[] {
    const input = this.embed(phrase);
    const { dim, hidden } = this.metadata;
    const intermediate = new Float32Array(hidden);
    for (let hiddenIndex = 0; hiddenIndex < hidden; hiddenIndex++) {
      let activation = this.b1[hiddenIndex]!;
      for (let dimension = 0; dimension < dim; dimension++) {
        activation += this.w1[hiddenIndex * dim + dimension]! * input[dimension]!;
      }
      intermediate[hiddenIndex] = Math.max(0, activation);
    }
    return Array.from({ length: 3 }, (_, channel) => {
      let prediction = this.b2[channel]!;
      for (let hiddenIndex = 0; hiddenIndex < hidden; hiddenIndex++) {
        prediction += this.w2[channel * hidden + hiddenIndex]! * intermediate[hiddenIndex]!;
      }
      return prediction * this.metadata.lab_scale;
    });
  }
}

export function labToRgb(lab: readonly number[]): number[] {
  const fy = (lab[0]! + 16) / 116;
  const transformed = [fy + lab[1]! / 500, fy, fy - lab[2]! / 200];
  const white = [0.95047, 1, 1.08883];
  const xyz = transformed.map((value, index) => {
    const normalized = value > 6 / 29 ? value ** 3 : (value - 16 / 116) / 7.787;
    return normalized * white[index]!;
  });
  // Inverse of the exact sRGB matrix used by the Python evaluation.
  const inverse = [
    [3.240625477320053, -1.537207972210319, -0.498628598698248],
    [-0.968930714729320, 1.875756060885241, 0.041517523842954],
    [0.055710120445511, -0.204021050598487, 1.056995942254388],
  ];
  return inverse.map(row => {
    const linearRgb = row.reduce((sum, coefficient, index) => sum + coefficient * xyz[index]!, 0);
    const srgb = linearRgb > 0.0031308 ? 1.055 * linearRgb ** (1 / 2.4) - 0.055 : 12.92 * linearRgb;
    return Math.round(Math.min(1, Math.max(0, srgb)) * 255);
  });
}

export function toHex(rgb: readonly number[]): string {
  return `#${rgb.map(v => Math.round(v).toString(16).padStart(2, "0")).join("")}`;
}
