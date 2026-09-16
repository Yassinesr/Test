#!/usr/bin/env python3
"""Run an ablation arm across seeds, sequentially, on one GPU.

    python tools/run_ablation.py --configs configs/a0_baseline.yaml configs/a1_pottc.yaml \
        --seeds 0 1 2 --out-dir runs/main

Every (config, seed) pair becomes ``<out-dir>/<config stem>/seed<k>/``.  The
runner is deliberately dumb and sequential: on a single 12 GB card the runs do
not fit side by side, and interleaving them would make wall-clock -- which A5
depends on -- meaningless.  It skips a pair whose ``results.json`` already
exists, so an interrupted sweep resumes.

``--auto-budget FROM TO`` closes the loop on ablation A5: it reads the measured
wall-clock of the two named arms and prints the epoch count that equalises
them, instead of leaving the placeholder 105 in the config.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_one(config: Path, seed: int, out_dir: Path, extra: list[str], dry: bool) -> bool:
    name = f"{config.stem}/seed{seed}"
    target = out_dir / name / "results.json"
    if target.is_file():
        print(f"[skip] {name} already has results.json")
        return True
    cmd = [sys.executable, str(ROOT / "tools" / "train.py"), "--config", str(config),
           f"run.seed={seed}", f"run.name={name}", f"run.out_dir={out_dir}", *extra]
    print(f"[run ] {' '.join(cmd)}", flush=True)
    if dry:
        return True
    t0 = time.time()
    proc = subprocess.run(cmd)
    ok = proc.returncode == 0
    print(f"[{'done' if ok else 'FAIL'}] {name} in {time.time() - t0:.0f}s", flush=True)
    return ok


def auto_budget(out_dir: Path, base_arm: str, ref_arm: str, seeds: list[int]) -> None:
    def mean_seconds(arm: str) -> float:
        vals = []
        for s in seeds:
            f = out_dir / arm / f"seed{s}" / "results.json"
            if f.is_file():
                blob = json.loads(f.read_text())
                vals.append(blob["wall_clock_seconds"])
        return sum(vals) / len(vals) if vals else float("nan")

    def epochs(arm: str) -> int:
        import yaml
        f = out_dir / arm / f"seed{seeds[0]}" / "config.yaml"
        return int(yaml.safe_load(f.read_text())["optim"]["epochs"])

    t_base, t_ref = mean_seconds(base_arm), mean_seconds(ref_arm)
    if not (t_base == t_base and t_ref == t_ref):  # NaN check
        print("auto-budget: need completed runs for both arms")
        return
    e_base = epochs(base_arm)
    matched = round(e_base * t_ref / t_base)
    print(f"\nA5 equal-compute budget")
    print(f"  {base_arm}: {t_base:.0f}s over {e_base} epochs ({t_base / e_base:.1f}s/epoch)")
    print(f"  {ref_arm}:  {t_ref:.0f}s  (overhead {100 * (t_ref / t_base - 1):+.1f}%)")
    print(f"  -> run A5 as: --config configs/a5_equal_budget.yaml optim.epochs={matched}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+", type=Path, required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", type=Path, default=Path("runs/main"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
    ap.add_argument("--auto-budget", nargs=2, metavar=("BASE_ARM", "REF_ARM"), default=None,
                    help="after running, print the A5 epoch count that equalises wall-clock")
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    failures: list[str] = []
    for config in args.configs:
        for seed in args.seeds:
            if not run_one(config, seed, args.out_dir, args.overrides, args.dry_run):
                failures.append(f"{config.stem}/seed{seed}")
                if not args.keep_going:
                    print(f"\nstopping after failure in {failures[-1]}")
                    return 1
    if args.auto_budget and not args.dry_run:
        auto_budget(args.out_dir, args.auto_budget[0], args.auto_budget[1], args.seeds)
    if failures:
        print(f"\n{len(failures)} runs failed: {failures}")
        return 1
    print(f"\nall {len(args.configs) * len(args.seeds)} runs complete under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
