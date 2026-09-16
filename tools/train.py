#!/usr/bin/env python3
"""Train one run.

    python tools/train.py --config configs/a1_pottc.yaml run.seed=0 run.name=a1_s0

Positional arguments after the flags are ``dotted.key=value`` overrides, so a
sweep needs no new config files.  The resolved config, the environment report
and the results land in ``<run.out_dir>/<run.name>/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.config import apply_overrides, load_config  # noqa: E402
from polyptail.engine.train import train  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.overrides)
    summary = train(cfg)
    print(json.dumps(summary["per_split"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
