#!/usr/bin/env python3
"""Score a saved checkpoint against the frozen test splits.

    python tools/evaluate.py --run runs/a1_s0 --checkpoint runs/a1_s0/last.pth

Re-scoring a checkpoint must give the same numbers as the run that produced
it; if it does not, the evaluation path is not deterministic and nothing
downstream is trustworthy.  Use ``--compare`` to assert that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from polyptail.config import apply_overrides, load_config  # noqa: E402
from polyptail.data.manifest import load_manifest  # noqa: E402
from polyptail.eval.evaluator import evaluate_all  # noqa: E402
from polyptail.models import build_model  # noqa: E402
from polyptail.utils.env import amp_dtype, pick_device, seed_everything  # noqa: E402
from polyptail.utils.io import safe_torch_load  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a run directory containing config.yaml")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="assert the recomputed mDice matches the run's results.json")
    ap.add_argument("--tolerance", type=float, default=1e-6)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.run / "config.yaml"), args.overrides)
    ckpt_path = args.checkpoint or (args.run / "last.pth")
    device = pick_device()
    seed_everything(cfg.run.seed)

    model_kwargs: dict = {"channel": cfg.model.channel}
    if cfg.model.name != "tiny_unet":
        # Weights come from the checkpoint; do not re-download the ImageNet init.
        model_kwargs["pretrained"] = None
        model_kwargs["strict_pretrained"] = False
    if cfg.model.name == "polyp_pvt":
        model_kwargs["variant"] = cfg.model.variant
    model, spec = build_model(cfg.model.name, **model_kwargs)
    state = safe_torch_load(ckpt_path)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()

    results = evaluate_all(
        model, Path(cfg.data.root), load_manifest(Path(cfg.data.manifest)), device,
        head=cfg.pot.head or spec.head, splits=cfg.data.test_splits, size=cfg.data.size,
        threshold=cfg.eval.threshold, num_workers=cfg.eval.num_workers,
        batch_size=cfg.eval.batch_size, compute_hd95=cfg.eval.hd95, compute_sweep=cfg.eval.sweep,
        amp_dtype=amp_dtype(cfg.optim.amp, device),
        save_root=(args.run / "predictions_recheck") if args.save_masks else None,
    )
    out = args.out or (args.run / "results_recheck.json")
    out.write_text(json.dumps(
        {"per_split": results["per_split"], "pooled_external": results["pooled_external"],
         "checkpoint": str(ckpt_path)}, indent=2) + "\n")
    for split, agg in results["per_split"].items():
        print(f"{split:<34} mDice {agg['dice']:.4f}  mIoU {agg['iou']:.4f}  "
              f"HD95 {agg['hd95']:.1f}  Dice<=0.5 {agg['dice_le_05']:.3f}")

    if args.compare:
        ref = json.loads((args.run / "results.json").read_text())["per_split"]
        bad = []
        for split, agg in results["per_split"].items():
            if split in ref and abs(agg["dice"] - ref[split]["dice"]) > args.tolerance:
                bad.append((split, ref[split]["dice"], agg["dice"]))
        if bad:
            print("\nEVALUATION IS NOT REPRODUCIBLE:")
            for s, a, b in bad:
                print(f"  {s}: stored {a:.6f} vs recomputed {b:.6f}")
            return 1
        print("\nrecomputed metrics match the stored results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
