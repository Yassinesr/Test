#!/usr/bin/env python3
"""Freeze a dataset into a content-addressed manifest.

    python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json

Discharges blocking item 1 of §6.2: "a published frozen manifest with SHA-256
for every training and test image/mask".  Commit the output.  Every training
run records which manifest it used and (with ``data.verify=hash``) refuses to
start if the files on disk have drifted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.data.layout import EXPECTED_COUNTS, VALIDATION_SPLIT  # noqa: E402
from polyptail.data.manifest import build_manifest, write_manifest  # noqa: E402

DEFAULT_SPLITS = [
    "TrainDataset",
    "TestDataset/Kvasir",
    "TestDataset/CVC-ClinicDB",
    "TestDataset/CVC-ColonDB",
    "TestDataset/CVC-300",
    "TestDataset/ETIS-LaribPolypDB",
]

#: What the PraNet repository manifest says these directories contain, imported
#: rather than restated so the two cannot drift.  A mismatch is not
#: automatically an error -- §1.2 of the brief documents an unresolved
#: 300-vs-380 discrepancy for CVC-ColonDB -- but it must be seen.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./dataset", type=Path)
    ap.add_argument("--out", default="manifests/pranet_protocol.json", type=Path)
    ap.add_argument("--splits", nargs="*", default=None,
                    help=f"default: {' '.join(DEFAULT_SPLITS)}, plus {VALIDATION_SPLIT} "
                         "when that directory exists")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="do not fail when an image and its mask differ in size")
    args = ap.parse_args()

    splits = list(args.splits) if args.splits else list(DEFAULT_SPLITS)
    if args.splits is None and (args.root / VALIDATION_SPLIT).is_dir():
        # A held-out validation set selects the checkpoint, so it has to be
        # hashed like everything else: otherwise the one split that decides
        # which weights you report is the one split nothing can verify.
        splits.append(VALIDATION_SPLIT)
        print(f"including {VALIDATION_SPLIT}/ -- it exists, and a split that selects the "
              f"checkpoint must be frozen too")
    args.splits = splits

    manifest = build_manifest(args.root, args.splits, strict_sizes=not args.allow_size_mismatch)
    sha_path = args.out.with_suffix(".sha256")
    write_manifest(manifest, args.out, sha_path)

    print(f"wrote {args.out} and {sha_path}")

    # A held-out validation set makes TrainDataset short *by design*, so the
    # partition is reconciled before the table is printed rather than
    # contradicted underneath it.
    n_train = manifest["splits"].get("TrainDataset", {}).get("n_pairs")
    n_val = manifest["splits"].get(VALIDATION_SPLIT, {}).get("n_pairs")
    pool = EXPECTED_COUNTS["TrainDataset"]
    partition_ok = (n_train is not None and n_val is not None and n_train + n_val == pool)

    print(f"{'split':<34} {'pairs':>6}  {'expected':>8}  status")
    mismatched: list[str] = []
    for split in args.splits:
        n = manifest["splits"][split]["n_pairs"]
        exp = EXPECTED_COUNTS.get(split)
        if exp is None:
            status = "held out by you" if split == VALIDATION_SPLIT else "-"
        elif n == exp:
            status = "ok"
        elif partition_ok and split == "TrainDataset":
            status = f"re-split ({n - exp:+d})"
        else:
            status = f"MISMATCH ({n - exp:+d})"
            mismatched.append(split)
        print(f"{split:<34} {n:>6}  {str(exp) if exp is not None else '-':>8}  {status}")
    print(f"{'TOTAL':<34} {manifest['total_pairs']:>6}")

    if n_train is not None and n_val is not None:
        total = n_train + n_val
        print(f"\nTrainDataset + {VALIDATION_SPLIT} = {total} vs the {pool} PraNet "
              f"distributes as the training pool: "
              + ("accounted for." if partition_ok else
                 f"{abs(total - pool)} {'unaccounted for' if total < pool else 'too many'}."))
        print("Run tools/hash_collisions.py next: it is what proves no image is in both "
              "halves, by pixels rather than by name.")

    if mismatched:
        print(f"\nWARNING: {', '.join(mismatched)} does not match the PraNet-distributed counts.")
        print("Do not report results against published numbers until you can explain why.")
        print("See docs/PROTOCOL.md section 'Dataset provenance warnings'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
