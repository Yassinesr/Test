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

from polyptail.data.manifest import build_manifest, write_manifest  # noqa: E402

DEFAULT_SPLITS = [
    "TrainDataset",
    "TestDataset/Kvasir",
    "TestDataset/CVC-ClinicDB",
    "TestDataset/CVC-ColonDB",
    "TestDataset/CVC-300",
    "TestDataset/ETIS-LaribPolypDB",
]

#: What the PraNet repository manifest says these directories contain.  A
#: mismatch is not automatically an error -- §1.2 of the brief documents an
#: unresolved 300-vs-380 discrepancy for CVC-ColonDB -- but it must be seen.
EXPECTED_COUNTS = {
    "TrainDataset": 1450,
    "TestDataset/Kvasir": 100,
    "TestDataset/CVC-ClinicDB": 62,
    "TestDataset/CVC-ColonDB": 380,
    "TestDataset/CVC-300": 60,
    "TestDataset/ETIS-LaribPolypDB": 196,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="./dataset", type=Path)
    ap.add_argument("--out", default="manifests/pranet_protocol.json", type=Path)
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="do not fail when an image and its mask differ in size")
    args = ap.parse_args()

    manifest = build_manifest(args.root, args.splits, strict_sizes=not args.allow_size_mismatch)
    sha_path = args.out.with_suffix(".sha256")
    write_manifest(manifest, args.out, sha_path)

    print(f"wrote {args.out} and {sha_path}")
    print(f"{'split':<34} {'pairs':>6}  {'expected':>8}  status")
    bad = False
    for split in args.splits:
        n = manifest["splits"][split]["n_pairs"]
        exp = EXPECTED_COUNTS.get(split)
        if exp is None:
            status = "-"
        elif n == exp:
            status = "ok"
        else:
            status = f"MISMATCH ({n - exp:+d})"
            bad = True
        print(f"{split:<34} {n:>6}  {str(exp):>8}  {status}")
    print(f"{'TOTAL':<34} {manifest['total_pairs']:>6}")

    if bad:
        print("\nWARNING: at least one split does not match the PraNet-distributed counts.")
        print("Do not report results against published numbers until you can explain why.")
        print("See docs/PROTOCOL.md section 'Dataset provenance warnings'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
