#!/usr/bin/env python3
"""Turn a finished ablation into paired statistics and a pre-registered verdict.

    python tools/analyze.py --runs runs/main --baseline a0_baseline \
        --method a1_pottc --a2 a2_cvar --seeds 0 1 2 --out reports/main.md

Produces:
  * a per-dataset table with per-seed values, the baseline's own seed sd
    (blocking item 6), and three paired 95% intervals (blocking item 8);
  * Holm-adjusted p-values across the five datasets, because a claim that
    ranges over all five is a family of five tests;
  * the left-tail diagnostics the candidate is actually about -- the fraction
    of images with Dice <= 0.5, the 5th percentile of per-image Dice, HD95;
  * the §9 falsification verdict, evaluated mechanically from the numbers.

The verdict is computed, not narrated.  If it says REJECT, the candidate is
rejected; editing thresholds afterwards converts a pre-registered test into a
post-hoc one and the result stops meaning anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polyptail.stats.paired import (  # noqa: E402
    DatasetComparison, compare_runs, falsification_verdict,
)


def load_arm(runs: Path, arm: str, seeds: list[int], metric: str = "dice") -> dict:
    """Assemble ``{dataset: {"per_seed": [...], "per_image": [{stem: v}, ...]}}``."""
    out: dict[str, dict] = {}
    meta: dict = {"wall_clock": [], "params": None, "missing": []}
    for seed in seeds:
        run_dir = runs / arm / f"seed{seed}"
        res_f, img_f = run_dir / "results.json", run_dir / "per_image_metrics.json"
        if not res_f.is_file():
            meta["missing"].append(str(run_dir))
            continue
        res = json.loads(res_f.read_text())
        meta["wall_clock"].append(res.get("wall_clock_seconds", float("nan")))
        meta["params"] = res.get("params")
        per_image = json.loads(img_f.read_text()) if img_f.is_file() else {}
        for split, agg in res["per_split"].items():
            node = out.setdefault(split, {"per_seed": [], "per_image": [], "aux": {}})
            node["per_seed"].append(agg[metric])
            for key in ("dice_le_05", "dice_p05", "hd95", "iou", "recall", "dice_sweep"):
                node["aux"].setdefault(key, []).append(agg.get(key, float("nan")))
            if split in per_image:
                node["per_image"].append({r["stem"]: r[metric] for r in per_image[split]})
        for split in list(out):
            if not out[split]["per_image"]:
                out[split]["per_image"] = []
    return {"data": out, "meta": meta}


def fmt_ci(c: Optional[object]) -> str:
    if c is None:
        return "n/a"
    return f"{c.mean:+.4f} [{c.lo:+.4f}, {c.hi:+.4f}]"  # type: ignore[attr-defined]


def table(cmps: dict[str, DatasetComparison], a_name: str, b_name: str) -> list[str]:
    lines = [
        f"| dataset | {a_name} mDice (sd) | {b_name} mDice (sd) | delta | seed-t 95% CI | "
        "hierarchical 95% CI | p (Holm) |",
        "|---|---|---|---|---|---|---|",
    ]
    for ds in sorted(cmps):
        c = cmps[ds]
        lines.append(
            f"| {ds.split('/')[-1]} | {c.a_mean:.4f} ({c.a_sd:.4f}) | {c.b_mean:.4f} ({c.b_sd:.4f}) "
            f"| {c.b_mean - c.a_mean:+.4f} | {fmt_ci(c.seed_t)} | {fmt_ci(c.hierarchical)} "
            f"| {c.p_holm:.3f} |"
        )
    return lines


def aux_table(a: dict, b: dict, key: str, label: str, a_name: str, b_name: str) -> list[str]:
    lines = [f"| dataset | {a_name} {label} | {b_name} {label} | delta |", "|---|---|---|---|"]
    for ds in sorted(set(a) & set(b)):
        av = np.mean(a[ds]["aux"].get(key, [np.nan]))
        bv = np.mean(b[ds]["aux"].get(key, [np.nan]))
        lines.append(f"| {ds.split('/')[-1]} | {av:.4f} | {bv:.4f} | {bv - av:+.4f} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs/main"))
    ap.add_argument("--baseline", default="a0_baseline")
    ap.add_argument("--method", default="a1_pottc")
    ap.add_argument("--a2", default=None, help="arm implementing ablation A2 (empirical quantile)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--metric", default="dice", choices=["dice", "dice_sweep", "iou", "recall"])
    ap.add_argument("--ci", default="seed_t", choices=["seed_t", "image_bootstrap", "hierarchical"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a = load_arm(args.runs, args.baseline, args.seeds, args.metric)
    b = load_arm(args.runs, args.method, args.seeds, args.metric)
    for arm, blob in ((args.baseline, a), (args.method, b)):
        if blob["meta"]["missing"]:
            print(f"ERROR: {arm} is missing runs: {blob['meta']['missing']}")
            return 1

    cmps = compare_runs(a["data"], b["data"], metric=args.metric, n_boot=args.n_boot)

    a2_cmps = None
    if args.a2:
        c = load_arm(args.runs, args.a2, args.seeds, args.metric)
        if c["meta"]["missing"]:
            print(f"WARNING: A2 arm {args.a2} incomplete; the C2 criterion cannot be evaluated.")
        else:
            a2_cmps = compare_runs(c["data"], b["data"], metric=args.metric, n_boot=args.n_boot)

    tail_frac = {
        ds: {"a": a["data"][ds]["aux"]["dice_le_05"], "b": b["data"][ds]["aux"]["dice_le_05"]}
        for ds in set(a["data"]) & set(b["data"])
    }
    verdict = falsification_verdict(cmps, tail_frac, a1_vs_a2=a2_cmps, ci=args.ci)

    lines: list[str] = []
    lines.append(f"# {args.method} vs {args.baseline}")
    lines.append("")
    lines.append(f"- seeds: {args.seeds}   metric: `{args.metric}`   verdict read at `{args.ci}`")
    lines.append(f"- parameters: {args.baseline} {a['meta']['params']:,} | "
                 f"{args.method} {b['meta']['params']:,}"
                 + ("  **(equal -- the loss adds none)**"
                    if a["meta"]["params"] == b["meta"]["params"] else "  **(NOT equal)**"))
    wa, wb = np.mean(a["meta"]["wall_clock"]), np.mean(b["meta"]["wall_clock"])
    lines.append(f"- wall clock: {wa:.0f}s vs {wb:.0f}s ({100 * (wb / wa - 1):+.1f}%). "
                 "A5 must match this, or the comparison is not equal-compute.")
    lines.append("")
    lines.append("## Per-dataset mDice")
    lines.append("")
    lines += table(cmps, args.baseline, args.method)
    lines.append("")
    lines.append("The seed-t interval is the one that licenses a claim about methods; with three "
                 "seeds it has 2 degrees of freedom and is wide by construction. The hierarchical "
                 "interval resamples seeds and images. §1.3 W5 of the brief puts the cross-paper "
                 "noise floor at 0.3-0.8 mDice, so read any delta below ~0.01 as uninterpretable.")
    lines.append("")
    lines.append("## Left tail -- the quantity POT-TC actually targets")
    lines.append("")
    lines.append("### Fraction of images with Dice <= 0.5 (lower is better)")
    lines.append("")
    lines += aux_table(a["data"], b["data"], "dice_le_05", "P(Dice<=0.5)", args.baseline, args.method)
    lines.append("")
    lines.append("### 5th percentile of per-image Dice (higher is better)")
    lines.append("")
    lines += aux_table(a["data"], b["data"], "dice_p05", "Dice p05", args.baseline, args.method)
    lines.append("")
    lines.append("### HD95 in pixels of the native grid (lower is better)")
    lines.append("")
    lines += aux_table(a["data"], b["data"], "hd95", "HD95", args.baseline, args.method)
    lines.append("")
    if a2_cmps:
        lines.append(f"## Decisive ablation: {args.method} vs {args.a2} (GPD vs empirical quantile)")
        lines.append("")
        lines += table(a2_cmps, args.a2, args.method)
        lines.append("")
    lines.append("## Pre-registered falsification verdict (candidate card §9)")
    lines.append("")
    lines.append("```")
    lines.append(verdict.summary())
    lines.append("```")
    lines.append("")
    lines.append("## Per-seed values")
    lines.append("")
    lines.append("| dataset | " + " | ".join(
        f"{n} s{s}" for n in (args.baseline, args.method) for s in args.seeds) + " |")
    lines.append("|---" * (1 + 2 * len(args.seeds)) + "|")
    for ds in sorted(cmps):
        c = cmps[ds]
        lines.append(f"| {ds.split('/')[-1]} | "
                     + " | ".join(f"{v:.4f}" for v in c.a_per_seed + c.b_per_seed) + " |")
    lines.append("")
    lines.append("---")
    lines.append("No superiority claim is licensed by this table alone. See `docs/PROTOCOL.md` "
                 "for the blocking items in §6.2 of the brief that must also be discharged.")

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        json_out = args.out.with_suffix(".json")
        json_out.write_text(json.dumps({
            "baseline": args.baseline, "method": args.method, "seeds": args.seeds,
            "metric": args.metric, "ci": args.ci,
            "comparisons": {k: v.to_json() for k, v in cmps.items()},
            "a1_vs_a2": None if not a2_cmps else {k: v.to_json() for k, v in a2_cmps.items()},
            "verdict": {"rejected": verdict.rejected, "criteria": verdict.criteria,
                        "notes": verdict.notes},
        }, indent=2, default=float) + "\n")
        print(f"\nwrote {args.out} and {json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
