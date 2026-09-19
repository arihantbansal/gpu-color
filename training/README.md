# Training

The Python scripts live in `color/` and run with `uv run python -m color.<script>`.

## Reproduce the export

From `training/`:

```bash
uv sync
./scripts/fetch_data.sh
uv run python -m color.prepare
uv run python -m color.train --export ../web/public/color
```

The selected run uses 2,048 buckets, a 20-dimensional embedding, a 64-unit
hidden layer, 4,000 steps, seed 0, and AdamW weight decay 0.1. The trainer
keeps the checkpoint with the lowest validation ΔE. Use `--out runs/<name>` for
a separate run.

The export contains quantised weights, metadata, and 100 Python parity
fixtures. Weights plus metadata must stay below 50,000 bytes. Quantisation
may increase mean ΔE by at most 1%.

## Model comparison

These candidates use 11,541 training names and 403 validation names. The same
validation names choose checkpoints and candidates, so the scores are not
final test results.

| Candidate | Mean int8 ΔE | Names above ΔE 40 | Bytes |
|---|---:|---:|---:|
| Original, 2048 × 16 | 17.8233 | 28 | 34,902 |
| 4096 × 10 | 16.8842 | 30 | 42,800 |
| 4096 × 11 | 16.3481 | 33 | 46,959 |
| Selected, 2048 × 20 | 17.2261 | 26 | 43,441 |

The selected model improves the mean and the count of large errors over the
original while staying within the size budget. These are single-seed results.

## Remaining errors

The selected model still predicts `poiple` as yellow-brown and `fandango` as
yellow-green. Its worst ΔE is 118.08, higher than the original model's 108.35.
Misspellings, object names such as `raisin`, and entries such as `your mom`
remain difficult. Spelling alone may not supply the missing associations.

A spot check moves `light blue` and `dark blue` in the expected brightness
directions, but `dusty rose at dusk` turns greenish. An independent test set
is still needed before stronger quality claims.
