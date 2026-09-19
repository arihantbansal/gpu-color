# gpu-color

A tiny model that turns color descriptions into colors, locally in your browser.

[Try the demo](https://arihantbansal.com/gpu-color/)

Type a description such as `dusty rose`, adjust the color, and copy its hex value.
CSS colors use the browser's parser. Other descriptions use a 43.4 KB neural
model. Predictions are approximate, especially for unfamiliar names.

The demo uses CPU inference. A WebGPU runtime is also included.

## Run locally

```sh
cd web
bun install --frozen-lockfile
bun run dev
```

The trained model is included.

## Test and build

From `web/`:

```sh
bun run test
bun run build
```

See [training details](training/README.md) for evaluation results and known errors.

Inspired by [gpu-lexer](https://github.com/vercel-labs/gpu-lexer).
Training data comes from the [xkcd color survey](https://blog.xkcd.com/2010/05/03/color-survey-results/).
See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution.

[MIT](LICENSE) © arihantbansal.
