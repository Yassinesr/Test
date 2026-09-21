#!/usr/bin/env python3
"""Generate a tiny synthetic dataset in the PraNet directory layout.

    python tools/make_smoke_data.py --out ./_smoke_data

Used by the CPU smoke test and by CI.  Images are ellipse-on-gradient with
noise; a controllable fraction get a deliberately *wrong* mask, so the tail of
the deficit distribution is non-degenerate and the POT-TC machinery has
something to fit.  Nothing here is a model of colonoscopy; it exists to prove
the plumbing runs, not to measure anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.data.manifest import build_manifest, write_manifest  # noqa: E402


def make_pair(rng: np.random.Generator, h: int, w: int, hard: bool):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = rng.uniform(0.3, 0.7) * h, rng.uniform(0.3, 0.7) * w
    ry, rx = rng.uniform(0.08, 0.22) * h, rng.uniform(0.08, 0.22) * w
    mask = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    # An independent low-frequency background per image.  A *shared* background
    # (a fixed gradient, say) would make every frame a perceptual near-duplicate
    # of every other, and tools/hash_collisions.py would correctly -- and
    # uselessly -- flag the whole corpus.
    base = np.stack([
        np.asarray(Image.fromarray(rng.normal(0, 1, (5, 6)).astype(np.float32), mode="F")
                   .resize((w, h), Image.BICUBIC)) * 45.0 + level
        for level in rng.uniform(70, 150, 3)
    ], -1)
    img = base + rng.normal(0, 8, (h, w, 3))
    contrast = 12.0 if hard else 70.0
    img += mask[..., None] * contrast
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, (mask.astype(np.uint8) * 255)


def write_split(out: Path, split: str, n: int, seed: int, hard_frac: float,
                size: tuple[int, int], prefix: str = ""):
    """Write one split. ``prefix`` keeps stems distinct across splits.

    Without it every split numbers from 0000, and a training and a validation
    image with the same stem look to the trainer exactly like the same image
    held out -- which it refuses to run on, correctly.
    """
    rng = np.random.default_rng(seed)
    (out / split / "images").mkdir(parents=True, exist_ok=True)
    (out / split / "masks").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        hard = rng.random() < hard_frac
        h = size[0] + int(rng.integers(0, 17))
        w = size[1] + int(rng.integers(0, 17))
        img, msk = make_pair(rng, h, w, hard)
        Image.fromarray(img).save(out / split / "images" / f"{prefix}{i:04d}.png")
        Image.fromarray(msk).save(out / split / "masks" / f"{prefix}{i:04d}.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("./_smoke_data"))
    ap.add_argument("--n-train", type=int, default=48)
    ap.add_argument("--n-val", type=int, default=12,
                    help="held-out validation pairs; the smoke run selects its "
                         "checkpoint on these, exercising the same path a real run uses")
    ap.add_argument("--n-test", type=int, default=16)
    ap.add_argument("--hard-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    write_split(args.out, "TrainDataset", args.n_train, args.seed, args.hard_frac, (80, 96))
    write_split(args.out, "ValidationDataset", args.n_val, args.seed + 2, args.hard_frac,
                (80, 96), prefix="val")
    write_split(args.out, "TestDataset/Fake", args.n_test, args.seed + 1, args.hard_frac,
                (72, 88), prefix="test")
    splits = ["TrainDataset", "ValidationDataset", "TestDataset/Fake"]
    manifest = build_manifest(args.out, splits)
    write_manifest(manifest, args.out / "manifest.json", args.out / "manifest.sha256")
    print(f"wrote {args.n_train} train, {args.n_val} validation and {args.n_test} test "
          f"pairs under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
