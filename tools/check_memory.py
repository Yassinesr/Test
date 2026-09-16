#!/usr/bin/env python3
"""Measure peak GPU memory for a config before committing to a 100-epoch run.

    python tools/check_memory.py --config configs/a1_pottc.yaml

Runs a handful of real training steps at every scale in ``optim.scales`` and
reports ``torch.cuda.max_memory_allocated`` for each, because the answer to
"does batch 16 fit in 12 GB?" depends on the scale schedule: at ``1.25`` the
input is 448x448, which is 1.6x the activation volume of 352x352, and that is
the step that decides whether a 3080 Ti OOMs on epoch 1 or on epoch 40.

If a scale does not fit, the script prints the next configuration down the
ladder rather than just failing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polyptail.config import apply_overrides, load_config  # noqa: E402
from polyptail.losses.deficit import combine_heads, per_image_deficit  # noqa: E402
from polyptail.losses.structure import structure_loss  # noqa: E402
from polyptail.losses.tail import TailConfig, TailRiskLoss  # noqa: E402
from polyptail.models import build_model  # noqa: E402
from polyptail.utils.env import amp_dtype, environment_report, pick_device  # noqa: E402
from polyptail.utils.io import make_grad_scaler  # noqa: E402

from dataclasses import asdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "a1_pottc.yaml")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("overrides", nargs="*", default=[])
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    device = pick_device()
    if device.type != "cuda":
        print("No CUDA device visible; this tool measures GPU memory and has nothing to do.")
        return 0

    env = environment_report(ROOT)
    total = env.get("gpu_total_mem_gb", float("nan"))
    print(f"GPU: {env.get('gpu_name')} ({total} GiB), torch {env['torch']}, CUDA {env['cuda_version']}")
    print(f"config: {args.config.name}  batch={cfg.optim.batch_size} "
          f"grad_accum={cfg.optim.grad_accum} amp={cfg.optim.amp} size={cfg.data.size}")
    print()

    model_kwargs: dict = {"channel": cfg.model.channel}
    if cfg.model.name != "tiny_unet":
        model_kwargs["pretrained"] = None       # weights are irrelevant to memory
        model_kwargs["strict_pretrained"] = False
    if cfg.model.name == "polyp_pvt":
        model_kwargs["variant"] = cfg.model.variant
    model, spec = build_model(cfg.model.name, **model_kwargs)
    model.to(device).train()
    head = cfg.pot.head or spec.head

    optimizer = torch.optim.AdamW(model.parameters(), cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    adtype = amp_dtype(cfg.optim.amp, device)
    scaler = make_grad_scaler(enabled=(adtype == torch.float16))
    tail = TailRiskLoss(TailConfig(**asdict(cfg.pot.tail)), total_steps=1000).to(device)

    micro = max(1, cfg.optim.batch_size // cfg.optim.grad_accum)
    worst = 0.0
    fits = True
    for rate in sorted(float(s) for s in cfg.optim.scales):
        size = int(round(cfg.data.size * rate / 32) * 32)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            for _ in range(args.steps):
                x = torch.randn(micro, 3, size, size, device=device)
                y = (torch.rand(micro, 1, size, size, device=device) > 0.7).float()
                optimizer.zero_grad(set_to_none=True)
                if adtype is not None:
                    with torch.autocast("cuda", dtype=adtype):
                        outs = model(x)
                else:
                    outs = model(x)
                outs = tuple(o.float() for o in outs)
                loss = sum(structure_loss(o, y, variant=cfg.optim.structure_variant) for o in outs)
                d = per_image_deficit(combine_heads(outs, head=head), y, kind=cfg.pot.deficit_kind)
                loss = loss + tail(d)[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            worst = max(worst, peak)
            print(f"  scale {rate:<5} -> {size}x{size}, micro-batch {micro}: peak {peak:.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            fits = False
            print(f"  scale {rate:<5} -> {size}x{size}, micro-batch {micro}: OUT OF MEMORY")
            torch.cuda.empty_cache()

    print()
    if fits:
        headroom = total - worst
        print(f"Peak {worst:.2f} GiB, headroom {headroom:.2f} GiB.")
        if headroom < 1.0:
            print("Under 1 GiB of headroom: fragmentation or a longer sequence can still OOM "
                  "mid-run. Consider the next rung down.")
        else:
            print("This configuration fits.")
    else:
        print("This configuration does NOT fit. Ladder, in order of preference "
              "(each rung leaves the effective batch at 16, so the optimisation is unchanged):")
        print("  1. optim.amp=fp16                      (if you had it off)")
        print("  2. optim.batch_size=8 optim.grad_accum=2")
        print("  3. optim.batch_size=4 optim.grad_accum=4")
        print("  4. optim.scales=[0.75,1.0]             (changes the recipe -- say so in the write-up)")
        print()
        print("Do NOT simply lower batch_size without raising grad_accum: that changes the")
        print("effective batch and makes the run incomparable to the reproduced baseline.")
    return 0 if fits else 1


if __name__ == "__main__":
    raise SystemExit(main())
