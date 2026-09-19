import { ColorModel } from "./runtime";

export async function createGpuPredictor(model: ColorModel) {
  if (!navigator.gpu) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return null;
  }

  const device = await adapter.requestDevice();
  const { dim, hidden } = model.metadata;
  const bias1Offset = model.w1.length;
  const weight2Offset = bias1Offset + model.b1.length;
  const bias2Offset = weight2Offset + model.w2.length;
  const weights = new Float32Array(bias2Offset + 3);
  weights.set(model.w1);
  weights.set(model.b1, bias1Offset);
  weights.set(model.w2, weight2Offset);
  weights.set(model.b2, bias2Offset);

  const weightBuffer = device.createBuffer({
    size: weights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightBuffer, 0, weights);

  const shaderModule = device.createShaderModule({ code: `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, read> weights: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;

    var<workgroup> hiddenActivations: array<f32, ${hidden}>;

    @compute @workgroup_size(${Math.max(3, hidden)})
    fn main(@builtin(local_invocation_index) invocationIndex: u32) {
      if (invocationIndex < ${hidden}u) {
        var activation = weights[${bias1Offset}u + invocationIndex];
        for (var dimension = 0u; dimension < ${dim}u; dimension++) {
          activation += input[dimension] * weights[invocationIndex * ${dim}u + dimension];
        }
        hiddenActivations[invocationIndex] = max(0.0, activation);
      }

      workgroupBarrier();

      if (invocationIndex < 3u) {
        var prediction = weights[${bias2Offset}u + invocationIndex];
        for (var hiddenIndex = 0u; hiddenIndex < ${hidden}u; hiddenIndex++) {
          prediction += hiddenActivations[hiddenIndex] * weights[${weight2Offset}u + invocationIndex * ${hidden}u + hiddenIndex];
        }
        output[invocationIndex] = prediction * 100.0;
      }
    }` });

  let pipeline: GPUComputePipeline;
  try {
    pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });
  } catch (error) {
    weightBuffer.destroy();
    device.destroy();
    throw error;
  }

  let lost = false;
  void device.lost.then(() => {
    lost = true;
  });

  return {
    async predict(phrase: string): Promise<number[]> {
      if (lost) {
        throw new Error("GPU device is unavailable");
      }

      const input = model.embed(phrase);
      const inputBuffer = device.createBuffer({
        size: input.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const outputBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      const readBuffer = device.createBuffer({
        size: 12,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      try {
        device.queue.writeBuffer(inputBuffer, 0, input);
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: weightBuffer } },
            { binding: 2, resource: { buffer: outputBuffer } },
          ],
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(1);
        pass.end();
        encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, 12);
        device.queue.submit([encoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const predictions = Array.from(new Float32Array(readBuffer.getMappedRange()));
        readBuffer.unmap();
        return predictions;
      } finally {
        inputBuffer.destroy();
        outputBuffer.destroy();
        readBuffer.destroy();
      }
    },
    destroy() {
      weightBuffer.destroy();
      device.destroy();
    },
  };
}
