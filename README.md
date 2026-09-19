# gpu-color

A small local model that turns color descriptions into editable color swatches.

CSS colours use the browser's parser. Other descriptions use a local neural
model and can be adjusted with the picker controls. The demo uses CPU
inference; a WebGPU runtime is included.

## Run the picker

```bash
cd web
bun install --frozen-lockfile
bun run dev
```

Type a description such as `dusty rose`, paste a CSS colour, or adjust the
swatch. The exported model is already included, so training is not required.

The model is 43.4 KB. Predictions are approximate, especially for unfamiliar
names. [Training details](training/README.md) include evaluation results and
known errors.

## Check the project

```bash
cd web
bun run test
bun run build
```

Inspired by [gpu-lexer](https://github.com/vercel-labs/gpu-lexer).

MIT. Training data comes from the [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/).
