"""Turn the raw survey into a fixed train/held-out split.

Applies the stage 1 decisions (drop spam and colourblind users), keeps names with
enough answers, and holds out whole *names* rather than rows. A held-out name
never appears in training, so the held-out score measures whether a model
understands words rather than whether it memorised a table.

Writes data/processed/:
  answers.npz   filtered rows: name_id (int32), rgb (int16, n x 3), user_id (int32),
                and vocab (str): vocab[name_id] is the row's name
  split.json    {"train": [...names], "heldout": [...names], "params": {...}}

Run: uv run python -m color.prepare
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent.parent / "data"
OUT = DATA / "processed"

MIN_ANSWERS = 5  # a name needs this many answers to be used at all
HELDOUT_MIN_ANSWERS = 20  # held-out names need a stable mean to score against
HELDOUT_FRACTION = 0.1
SEED = 0

QUERY = """
select a.colorname, a.r, a.g, a.b, a.user_id
from answers a join users u on a.user_id = u.id
where u.spamprob <= 0.5 and (u.colorblind is null or u.colorblind = 0)
"""


def is_junk(name: str) -> bool:
    return len(name) < 3 or any(c.isnumeric() for c in name)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    db = sqlite3.connect(DATA / "mainsurvey.sqlite")
    rows = db.execute(QUERY).fetchall()
    db.close()

    # Names go through a dict first: a numpy string array pads every element to
    # the longest name, and a few spam names are thousands of characters long.
    ids: dict[str, int] = {}
    name_id = np.fromiter(
        (ids.setdefault(r[0].strip().lower(), len(ids)) for r in rows),
        dtype=np.int32,
        count=len(rows),
    )
    vocab = np.array(list(ids), dtype=object)
    rgb = np.array([r[1:4] for r in rows], dtype=np.int16)
    user_id = np.array([r[4] for r in rows], dtype=np.int32)

    counts = np.bincount(name_id, minlength=len(vocab))
    junk = np.array([is_junk(n) for n in vocab])
    ok = (counts >= MIN_ANSWERS) & ~junk
    mask = ok[name_id]
    name_id, rgb, user_id = name_id[mask], rgb[mask], user_id[mask]
    keep = vocab[ok]

    rng = np.random.default_rng(SEED)
    stable = vocab[(counts >= HELDOUT_MIN_ANSWERS) & ~junk]
    heldout = rng.choice(
        stable, size=int(len(stable) * HELDOUT_FRACTION), replace=False
    )
    heldout_set = set(heldout.tolist())
    train = [n for n in keep.tolist() if n not in heldout_set]

    np.savez_compressed(
        OUT / "answers.npz", name_id=name_id, rgb=rgb, user_id=user_id, vocab=vocab
    )
    (OUT / "split.json").write_text(
        json.dumps(
            {
                "train": train,
                "heldout": sorted(heldout_set),
                "params": {
                    "min_answers": MIN_ANSWERS,
                    "heldout_min_answers": HELDOUT_MIN_ANSWERS,
                    "heldout_fraction": HELDOUT_FRACTION,
                    "seed": SEED,
                },
            },
            indent=1,
        )
    )
    print(f"rows kept: {len(name_id):,} of {len(rows):,}")
    print(
        f"names kept: {len(keep):,}  train: {len(train):,}  heldout: {len(heldout_set):,}"
    )


if __name__ == "__main__":
    main()
