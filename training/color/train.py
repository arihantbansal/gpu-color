"""Train, quantise, and export gpu-color. Run: uv run python -m color.train."""

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch

from color.baseline import rgb_to_lab
from color.model import ColorModel, tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "training/data/processed"


def batch(names, buckets):
    ids, offsets = [], []
    for name in names:
        offsets.append(len(ids))
        ids.extend(tokenize(name, buckets))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


def lab_to_rgb(lab):
    """D65 Lab to display sRGB; clip colours outside the display gamut."""
    lab = np.asarray(lab, dtype=float)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack((fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200), axis=-1)
    xyz = np.where(f > 6 / 29, f**3, (f - 16 / 116) / 7.787)
    xyz *= [0.95047, 1.0, 1.08883]
    matrix = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    linear = xyz @ np.linalg.inv(matrix).T
    srgb = np.where(
        linear > 0.0031308,
        1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
        12.92 * linear,
    )
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buckets", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=ROOT / "training/runs/color")
    parser.add_argument(
        "--export", type=Path, help="Directory for browser model and fixtures"
    )
    args = parser.parse_args()
    if min(args.buckets, args.dim, args.hidden, args.steps) <= 0:
        parser.error("Model dimensions and steps must be positive")
    if not np.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be finite and nonnegative")
    torch.manual_seed(args.seed)
    # Small full batches are faster without a large CPU thread pool.
    torch.set_num_threads(4)
    with np.load(DATA / "answers.npz", allow_pickle=True) as data:
        name_id, rgb, vocab = data["name_id"], data["rgb"], data["vocab"]
    split = json.loads((DATA / "split.json").read_text())
    counts = np.bincount(name_id, minlength=len(vocab))
    sums = np.stack(
        [
            np.bincount(name_id, weights=rgb[:, c], minlength=len(vocab))
            for c in range(3)
        ],
        axis=1,
    )
    valid = counts > 0
    means = sums[valid] / counts[valid, None]
    id_of = {name: i for i, name in enumerate(vocab[valid])}
    lab = rgb_to_lab(means)
    train_names, heldout_names = split["train"], split["heldout"]
    if set(train_names) & set(heldout_names):
        raise ValueError("Training and held-out names overlap")
    train_targets = torch.tensor(
        np.stack([lab[id_of[n]] for n in train_names]) / 100, dtype=torch.float32
    )
    truth = torch.tensor(
        np.stack([lab[id_of[n]] for n in heldout_names]), dtype=torch.float32
    )
    train_inputs = batch(train_names, args.buckets)
    heldout_inputs = batch(heldout_names, args.buckets)
    model = ColorModel(args.buckets, args.dim, args.hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, weight_decay=args.weight_decay
    )
    best_error, best_step, best_state = float("inf"), 0, None
    for step in range(args.steps):
        optimizer.zero_grad()
        prediction = model(*train_inputs)
        loss = (prediction - train_targets).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.steps:
            with torch.no_grad():
                error = (
                    torch.linalg.vector_norm(
                        model(*heldout_inputs) * 100 - truth, dim=1
                    )
                    .mean()
                    .item()
                )
            if error < best_error:
                best_error, best_step = error, step + 1
                best_state = copy.deepcopy(model.state_dict())
        if step % 250 == 0:
            print(f"step {step}: training MSE {loss.item():.6f}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        float_predictions = model(*heldout_inputs) * 100
        float_error = (
            torch.linalg.vector_norm(float_predictions - truth, dim=1).mean().item()
        )
    args.out.mkdir(parents=True, exist_ok=True)
    config = {"buckets": args.buckets, "dim": args.dim, "hidden": args.hidden}
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
            "seed": args.seed,
            "steps": best_step,
        },
        args.out / "model.pt",
    )

    # Symmetric per-tensor int8 quantisation; bytes are row-major PyTorch weights.
    payload = bytearray()
    tensors = []
    quantized = ColorModel(**config)
    state = {}
    for name, parameter in model.state_dict().items():
        values = parameter.numpy()
        scale = float(np.max(np.abs(values))) / 127 or 1.0
        integers = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
        tensors.append(
            {
                "name": name,
                "shape": list(values.shape),
                "offset": len(payload),
                "length": values.size,
                "scale": scale,
            }
        )
        payload.extend(integers.tobytes())
        state[name] = torch.from_numpy(integers.astype(np.float32) * scale)
    quantized.load_state_dict(state)
    quantized.eval()
    with torch.no_grad():
        predictions = quantized(*heldout_inputs) * 100
        errors = torch.linalg.vector_norm(predictions - truth, dim=1)
    quant_error = errors.mean().item()
    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "train_names": len(train_names),
        "validation_names": len(heldout_names),
        "float_delta_e": float_error,
        "int8_delta_e": quant_error,
        "quantization_relative_error_increase": (quant_error / float_error - 1),
        "weights_bytes": len(payload),
    }
    metrics.update(
        best_step=best_step,
        weight_decay=args.weight_decay,
        p90_delta_e=torch.quantile(errors, 0.9).item(),
        large_errors=int((errors > 40).sum()),
    )
    metadata = {
        "version": 1,
        **config,
        "tokenizer": "lower-word-boundary-trigrams-fnv1a32-utf8",
        "lab_scale": 100,
        "tensors": tensors,
        "metrics": metrics,
    }
    metadata_text = json.dumps(metadata, separators=(",", ":"))
    metrics["model_bytes"] = len(payload) + len(metadata_text.encode()) + 1
    # Re-serialise until the self-reported byte count is stable.
    for _ in range(3):
        metadata_text = json.dumps(metadata, separators=(",", ":"))
        metrics["model_bytes"] = len(payload) + len(metadata_text.encode()) + 1
    metadata_text = json.dumps(metadata, separators=(",", ":"))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (args.out / "errors.csv").open("w") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "delta_e", "prediction_hex", "target_hex"])
        for index in torch.argsort(errors, descending=True)[:50].tolist():
            pred_hex = "#" + "".join(
                f"{v:02x}" for v in lab_to_rgb(predictions[index].numpy())
            )
            target_hex = "#" + "".join(
                f"{v:02x}" for v in lab_to_rgb(truth[index].numpy())
            )
            writer.writerow(
                [
                    heldout_names[index],
                    round(errors[index].item(), 4),
                    pred_hex,
                    target_hex,
                ]
            )
    print(json.dumps(metrics, indent=2), flush=True)
    if args.export:
        if metrics["model_bytes"] >= 50_000:
            raise ValueError("Export exceeds the 50 KB model budget")
        if (
            not np.isfinite(quant_error)
            or metrics["quantization_relative_error_increase"] > 0.01
        ):
            raise ValueError("Export exceeds the 1% quantisation error budget")
        args.export.mkdir(parents=True, exist_ok=True)
        (args.export / "model.bin").write_bytes(payload)
        (args.export / "model.json").write_text(metadata_text + "\n")
        # Independent PyTorch fixtures for the exact quantised weights shipped to JS.
        phrases = heldout_names[:90] + [
            "Blue",
            "blue blue",
            "  light\tblue  ",
            "café",
            "蓝色",
            "💙 blue",
            "hi",
            "",
            "   ",
            "dusty rose at dusk",
        ]
        with torch.no_grad():
            outputs = quantized(*batch(phrases, args.buckets)).numpy() * 100
        fixtures = [
            {
                "phrase": p,
                "ids": tokenize(p, args.buckets),
                "lab": v.tolist(),
                "rgb": lab_to_rgb(v).tolist(),
            }
            for p, v in zip(phrases, outputs)
        ]
        (args.export / "fixtures.json").write_text(
            json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
