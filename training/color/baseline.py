import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "data" / "processed"


def rgb_to_lab(rgb: np.ndarray):
    srgb = rgb / 255.0
    mask = srgb > 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055) ** 2.4
    srgb[~mask] = srgb[~mask] / 12.92

    M = np.array(
        [
            [0.4124, 0.3576, 0.1805],
            [0.2126, 0.7152, 0.0722],
            [0.0193, 0.1192, 0.9505],
        ]
    )
    xyz = srgb @ M.T

    f = xyz / np.array([0.95047, 1.0, 1.08883])

    mask = f > 0.008856
    f[mask] = np.cbrt(f[mask])
    f[~mask] = (f[~mask] * 7.787) + 16 / 116

    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def main() -> None:
    data = np.load(DATA / "answers.npz", allow_pickle=True)
    name_id = data["name_id"]
    rgb = data["rgb"]
    vocab = data["vocab"]

    split_data = json.loads((DATA / "split.json").read_text())

    r = np.bincount(name_id, weights=rgb[:, 0].astype(float), minlength=len(vocab))
    g = np.bincount(name_id, weights=rgb[:, 1].astype(float), minlength=len(vocab))
    b = np.bincount(name_id, weights=rgb[:, 2].astype(float), minlength=len(vocab))

    counts = np.bincount(name_id, minlength=len(vocab))

    sums = np.stack([r, g, b], axis=1)
    # Removed names still have vocabulary entries but no defined mean.
    mean_of = np.full_like(sums, np.nan)
    np.divide(sums, counts[:, None], out=mean_of, where=counts[:, None] > 0)

    id_of = {name: i for i, name in enumerate(vocab)}
    for name in split_data["train"] + split_data["heldout"]:
        if name not in id_of or counts[id_of[name]] == 0:
            raise ValueError(f"Split name has no answers: {name!r}")

    print(f"Mean RGB for 'dusty rose' is {mean_of[id_of['dusty rose']]}")

    train_names = set(split_data["train"])
    train_mask = np.array([name in train_names for name in vocab])

    global_mean = rgb[train_mask[name_id]].mean(axis=0)
    print(f"Global mean RGB is {global_mean}")

    p1_list = []
    p1_lab_list = []
    p2_list = []
    p2_lab_list = []
    p2_fallback_count = 0

    for held_out_name in split_data["heldout"]:
        held_out_mean = mean_of[id_of[held_out_name]]

        pred_1 = global_mean

        val = [mean_of[id_of[w]] for w in held_out_name.split() if w in train_names]

        if len(val) > 0:
            pred_2 = np.mean(val, axis=0)
        else:
            pred_2 = global_mean
            p2_fallback_count += 1

        p1_list.append(np.linalg.norm(held_out_mean - pred_1))
        p2_list.append(np.linalg.norm(held_out_mean - pred_2))

        p1_lab_list.append(
            np.linalg.norm(rgb_to_lab(held_out_mean) - rgb_to_lab(pred_1))
        )
        p2_lab_list.append(
            np.linalg.norm(rgb_to_lab(held_out_mean) - rgb_to_lab(pred_2))
        )

    print(f"Mean L2 distance for prediction 1 (global mean): {np.mean(p1_list)}")
    print(
        f"Mean L2 distance for prediction 2 (mean of known words): {np.mean(p2_list)}"
    )
    print(
        f"Fallback to global mean for prediction 2 occurred {p2_fallback_count} times out of {len(split_data['heldout'])} held-out names."
    )

    print(
        f"Mean L2 distance in LAB space for prediction 1 (global mean): {np.mean(p1_lab_list)}"
    )
    print(
        f"Mean L2 distance in LAB space for prediction 2 (mean of known words): {np.mean(p2_lab_list)}"
    )


if __name__ == "__main__":
    main()
