#!/usr/bin/env python3
"""Publish the duplicate-collision matrix across all splits.

    python tools/hash_collisions.py --root ./dataset \
        --manifest manifests/pranet_protocol.json --out manifests/collisions.json

Discharges blocking item 2 of §6.2 and, as a bonus, audits train/test leakage,
which the protocol's 900/550 split provides no guarantee against.

Read the output like this:
  * ``TestDataset/CVC-300 -> TestDataset/CVC-ColonDB`` non-zero means the two
    "independent external" sets overlap.  Report CVC-300 and
    CVC-ColonDB-minus-CVC-300 separately, and do not quote a pooled external
    average until you have.
  * ``TrainDataset -> TestDataset/*`` non-zero means the in-domain numbers on
    that test set are contaminated.  Say so in the results table.
  * ``TrainDataset -> ValidationDataset`` non-zero means the checkpoint was
    selected on images the model trained on, so the selection is optimistic
    and every number downstream of it inherits that.  This is the one to check
    first if you re-split the training pool yourself: the trainer already
    refuses a shared *stem*, but the same frame saved twice under two names
    passes that check and fails this one.
  * A zero matrix is a *published negative result*, which is worth as much as
    a positive one: it is currently absent from the literature.

Flag counts are a screening signal, not a verdict. Colonoscopy frames share a
dark circular vignette and a narrow colour gamut, so genuinely distinct frames
collide more often than they would in a natural-image corpus. Read the flag
matrix together with the two threshold-free outputs the report also carries:
``nearest_neighbour_stats`` (an overlapping pair of splits shows a spike of
near-zero nearest-neighbour distances that no threshold can hide, while a
disjoint pair is unimodal near 32) and ``exact_pixel_duplicates`` (decoded-pixel
SHA-256 collisions, which admit no interpretation at all). Inspect the pairs in
``--out``'s ``.pairs.csv`` by eye before acting on a count.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.data.manifest import load_manifest  # noqa: E402
from polyptail.data.phash import audit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./dataset", type=Path)
    ap.add_argument("--manifest", default="manifests/pranet_protocol.json", type=Path)
    ap.add_argument("--out", default="manifests/collisions.json", type=Path)
    ap.add_argument("--splits", nargs="*", default=None)
    ap.add_argument("--ahash-max", type=int, default=8)
    ap.add_argument("--dhash-max", type=int, default=8)
    ap.add_argument("--phash-max", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    rep = audit(
        args.root, manifest, splits=args.splits,
        ahash_max=args.ahash_max, dhash_max=args.dhash_max, phash_max=args.phash_max,
        progress=not args.quiet,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    blob = rep.to_json()
    blob["thresholds"] = {"ahash": args.ahash_max, "dhash": args.dhash_max, "phash": args.phash_max}
    args.out.write_text(json.dumps(blob, indent=1) + "\n")

    csv_path = args.out.with_suffix(".pairs.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split_a", "image_a", "split_b", "image_b", "ahash", "dhash", "phash"])
        for p in rep.pairs:
            w.writerow([p["split_a"], p["image_a"], p["split_b"], p["image_b"],
                        p["ahash"], p["dhash"], p["phash"]])

    print()
    print("Images in ROW that have at least one near-duplicate in COLUMN")
    print(rep.summary())
    print(f"\nwrote {args.out} and {csv_path}")

    # The check this tool exists to make, for a dataset re-split by hand: a
    # checkpoint chosen on images the model trained on is optimistic, and
    # every number downstream of that checkpoint inherits it.
    if "ValidationDataset" not in rep.splits:
        print("\nNOTE: no ValidationDataset in this manifest, so the train -> validation")
        print("leakage check did not run. If you hold a validation split out, re-freeze")
        print("(tools/freeze_manifest.py picks it up automatically) and run this again.")
    else:
        leak = [p for p in rep.pairs
                if {p["split_a"], p["split_b"]} == {"TrainDataset", "ValidationDataset"}]
        exact_leak = [g for g in rep.exact_pixel_duplicates
                      if {"TrainDataset", "ValidationDataset"} <=
                      {m["split"] for m in g["members"]}]
        if exact_leak or leak:
            print(f"\n*** TrainDataset <-> ValidationDataset: {len(leak)} flagged pair(s), "
                  f"{len(exact_leak)} byte-identical. ***")
            print("The checkpoint would be selected on images the model trained on. Fix the")
            print("partition before running; this one blocks the run, not just the claims.")
        else:
            print("\nTrainDataset <-> ValidationDataset: clean. Selection is on held-out data.")

    cross = [p for p in rep.pairs if p["split_a"] != p["split_b"]]
    if cross:
        print(f"\n*** {len(cross)} cross-split near-duplicate pairs found. ***")
        print("The five test sets are not independent as distributed, and/or the")
        print("train split leaks into a test or validation split.  See docs/PROTOCOL.md")
        print("for what to report.  This does not block training; it blocks *claims* --")
        print("except for TrainDataset -> ValidationDataset, which blocks the run: fix")
        print("the partition before selecting a checkpoint on it.")
    else:
        print("\nNo cross-split near-duplicates at these thresholds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
