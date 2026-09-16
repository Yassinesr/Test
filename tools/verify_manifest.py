#!/usr/bin/env python3
"""Re-verify a dataset against its frozen manifest.  Exits non-zero on drift.

    python tools/verify_manifest.py --root ./dataset --manifest manifests/pranet_protocol.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.data.manifest import load_manifest, verify_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="./dataset", type=Path)
    ap.add_argument("--manifest", default="manifests/pranet_protocol.json", type=Path)
    ap.add_argument("--skip-pixels", action="store_true",
                    help="only check file bytes; skips the re-encode detector")
    args = ap.parse_args()

    rep = verify_manifest(args.root, load_manifest(args.manifest), check_pixels=not args.skip_pixels)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
