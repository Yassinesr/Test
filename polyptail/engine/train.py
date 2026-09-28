"""Training loop.

Reproduces the released Polyp-PVT recipe exactly where it is well defined, and
documents each place where the released code and the paper disagree:

* **Epoch count.**  ``for epoch in range(1, opt.epoch)`` with ``--epoch 100``
  runs 99 epochs.  Here ``optim.epochs`` means what it says; set it to 99 to
  match the released artefact bit-for-bit.
* **Learning-rate schedule.**  The released ``Train.py`` exposes
  ``--decay_epoch 50`` but calls ``adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)``,
  so with 100 epochs the decay never fires: the published recipe is a
  **constant** 1e-4.  Its ``adjust_lr`` also multiplies the *current* lr
  instead of rescaling the initial one, which compounds whenever it does
  fire.  Defaults here (``decay_epoch: 200``) reproduce the constant-lr
  behaviour; the implementation computes ``lr = init_lr * rate ** (epoch //
  decay_epoch)`` without compounding.
* **Gradient clipping** is by value to +/-0.5, as released, not by norm.
* **Model selection.**  The released ``Train.py`` evaluates the five test sets
  every epoch and checkpoints on the best test mDice.  That is selection on
  the test set.  This trainer cannot express it: the only selection criteria
  are ``last``, or validation Dice on data the model never trains on --
  either a held-out directory (``data.val_split``, e.g. ``ValidationDataset``)
  or a fold carved out of the training split (``data.val_frac``).  A
  ``val_split`` is checked against the training split for shared stems before
  the first epoch, because a partition that overlaps selects on what it
  trained on while looking exactly like one that does not.  The test splits
  are read once, after training ends; ``eval.every`` may sample them for
  curves, and that number never reaches the checkpoint.

Multi-scale training keeps the reference structure: each scale in
``optim.scales`` is a separate optimiser step over the same batch.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import Config, dump_config
from ..data.dataset import PolypTrainDataset, items_for_split, worker_init_fn
from ..data.manifest import load_manifest, verify_manifest
from ..eval.evaluator import evaluate_all, evaluate_split
from ..losses.deficit import combine_heads, per_image_deficit
from ..losses.structure import structure_loss
from ..losses.tail import TailConfig, TailRiskLoss
from ..models import MODELS, build_model
from ..utils.io import make_grad_scaler, safe_torch_load
from ..utils.env import amp_dtype, environment_report, pick_device, seed_everything
from ..utils.logging import AvgMeter, JsonlWriter, setup_logging

logger = logging.getLogger(__name__)

__all__ = ["train"]


def _split_train_val(items, val_frac: float, seed: int):
    if val_frac <= 0:
        return list(items), []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    n_val = max(1, int(round(val_frac * len(items))))
    val_idx = set(int(i) for i in idx[:n_val])
    train = [it for i, it in enumerate(items) if i not in val_idx]
    val = [it for i, it in enumerate(items) if i in val_idx]
    return train, val


def _lr_at(cfg: Config, epoch: int) -> float:
    return cfg.optim.lr * (cfg.optim.decay_rate ** (epoch // cfg.optim.decay_epoch))


def _set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _clip_by_value(optimizer, limit: float) -> None:
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                p.grad.data.clamp_(-limit, limit)


def _resize_pair(images, gts, size: int):
    if images.shape[-1] == size:
        return images, gts
    images = F.interpolate(images, size=(size, size), mode="bilinear", align_corners=True)
    gts = F.interpolate(gts, size=(size, size), mode="bilinear", align_corners=True)
    return images, gts


def train(cfg: Config) -> dict:
    cfg.validate()
    out_dir = Path(cfg.run.out_dir) / cfg.run.name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)
    seed_everything(cfg.run.seed, deterministic=cfg.run.deterministic)
    device = pick_device()
    adtype = amp_dtype(cfg.optim.amp, device)

    root = Path(cfg.data.root)
    manifest = load_manifest(Path(cfg.data.manifest))
    if cfg.data.verify == "hash":
        rep = verify_manifest(root, manifest, check_pixels=True)
        logger.info("manifest verification: %s", rep.summary())
        if not rep.ok:
            raise RuntimeError("dataset does not match the frozen manifest:\n" + rep.summary())
    elif cfg.data.verify == "exists":
        missing = [
            it["image"] for s in manifest["splits"].values() for it in s["items"]
            if not (root / it["image"]).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} manifest files missing, e.g. {missing[:3]}")

    all_items = items_for_split(manifest, cfg.data.train_split)
    for extra in cfg.data.extra_train_splits:
        if extra not in manifest["splits"]:
            raise KeyError(
                f"data.extra_train_splits names {extra!r}, which is not in "
                f"{cfg.data.manifest}. It holds {sorted(manifest['splits'])}."
            )
        all_items = all_items + items_for_split(manifest, extra)
        logger.info("folded %s into training (%d pairs)", extra,
                    len(items_for_split(manifest, extra)))
    if cfg.data.val_split:
        if cfg.data.val_split not in manifest["splits"]:
            raise KeyError(
                f"data.val_split={cfg.data.val_split!r} is not in {cfg.data.manifest}. "
                f"The manifest holds {sorted(manifest['splits'])}. Re-freeze it with that "
                f"directory included: python tools/freeze_manifest.py --root "
                f"{cfg.data.root} --out {cfg.data.manifest}"
            )
        train_items = list(all_items)
        val_items = items_for_split(manifest, cfg.data.val_split)
        overlap = {it.stem for it in train_items} & {it.stem for it in val_items}
        if overlap:
            raise ValueError(
                f"{len(overlap)} stem(s) appear in both {cfg.data.train_split} and "
                f"{cfg.data.val_split}, e.g. {sorted(overlap)[:4]}. Selection would be on "
                "images the model trained on. Fix the split before running."
            )
        val_source = cfg.data.val_split
    else:
        train_items, val_items = _split_train_val(all_items, cfg.data.val_frac, cfg.data.val_seed)
        val_source = f"{cfg.data.val_frac:.0%} of {cfg.data.train_split} at seed {cfg.data.val_seed}"
    logger.info("train pairs: %d   val pairs: %d%s", len(train_items), len(val_items),
                f"   (val from {val_source})" if val_items else "")

    train_ds = PolypTrainDataset(
        root, train_items, size=cfg.data.size, augment=cfg.data.augment,
        mask_interpolation=cfg.data.mask_interpolation,
    )
    loader = DataLoader(
        train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=False, worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(cfg.run.seed),
        persistent_workers=cfg.data.num_workers > 0,
    )

    model_kwargs: dict = {"channel": cfg.model.channel}
    if cfg.model.name != "tiny_unet":
        model_kwargs["strict_pretrained"] = cfg.model.strict_pretrained
        model_kwargs["pretrained"] = (
            cfg.model.pretrained if cfg.model.pretrained is not None
            else MODELS[cfg.model.name].default_pretrained
        )
    if cfg.model.name == "polyp_pvt":
        model_kwargs["variant"] = cfg.model.variant
    model, spec = build_model(cfg.model.name, **model_kwargs)
    model.to(device)
    head = cfg.pot.head or spec.head
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model %s: %.3fM params, eval head=%s", cfg.model.name, n_params / 1e6, head)

    params = model.parameters()
    if cfg.optim.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(params, cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    elif cfg.optim.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params, cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            params, cfg.optim.lr, weight_decay=cfg.optim.weight_decay, momentum=0.9
        )
    scaler = make_grad_scaler(enabled=(adtype == torch.float16))

    scales = [float(s) for s in cfg.optim.scales]
    tail_scales = [1.0] if cfg.pot.apply_scales == "unit" else scales
    steps_per_epoch = len(loader)
    tail_calls_per_epoch = steps_per_epoch * len(tail_scales) * cfg.optim.grad_accum
    tail = TailRiskLoss(
        TailConfig(**asdict(cfg.pot.tail)), total_steps=tail_calls_per_epoch * cfg.optim.epochs
    ).to(device)
    logger.info(
        "tail term: mode=%s p=%.3f alpha=%.3f lam=%.3f warmup=%.0f calls (%d calls/epoch)",
        cfg.pot.tail.mode, cfg.pot.tail.p, cfg.pot.tail.alpha, cfg.pot.tail.lam,
        cfg.pot.tail.warmup_frac * tail_calls_per_epoch * cfg.optim.epochs, tail_calls_per_epoch,
    )

    dump_config(cfg, out_dir / "config.yaml")
    (out_dir / "environment.json").write_text(json.dumps(environment_report("."), indent=2) + "\n")
    (out_dir / "model_info.json").write_text(json.dumps(
        {"name": cfg.model.name, "params": n_params, "head": head,
         "n_heads": int(getattr(model, "n_heads", 1))}, indent=2) + "\n")
    jsonl = JsonlWriter(out_dir / "train_metrics.jsonl")
    deficit_csv = None
    if cfg.pot.log_per_image:
        deficit_csv = open(out_dir / "per_image_deficits.csv", "w", newline="")
        deficit_writer = csv.writer(deficit_csv)
        deficit_writer.writerow(["epoch", "dataset_index", "stem", "deficit"])

    best_val = -1.0
    t_start = time.time()

    for epoch in range(1, cfg.optim.epochs + 1):
        lr = _lr_at(cfg, epoch)
        _set_lr(optimizer, lr)
        model.train()
        base_meter, tail_meter = AvgMeter(), AvgMeter()
        tail_meters = {k: AvgMeter() for k in
                       ("u", "xi", "xi_raw", "beta", "q_hat", "n_pool", "n_live",
                        "n_live_active", "w_sum", "degenerate", "xi_clamped",
                        "deficit_mean")}
        active_meter = AvgMeter()
        epoch_deficits: dict[int, float] = {}
        last_stats: dict = {}
        t_epoch = time.time()

        for step, (images, gts, index) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            gts = gts.to(device, non_blocking=True)
            for rate in scales:
                size = int(round(cfg.data.size * rate / 32) * 32)
                use_tail = rate in tail_scales
                optimizer.zero_grad(set_to_none=True)
                chunks = torch.chunk(torch.arange(images.size(0)), cfg.optim.grad_accum)
                for chunk in chunks:
                    if chunk.numel() == 0:
                        continue
                    img_c, gt_c = _resize_pair(images[chunk], gts[chunk], size)
                    if adtype is not None:
                        with torch.autocast("cuda", dtype=adtype):
                            outs = model(img_c)
                    else:
                        outs = model(img_c)
                    outs = tuple(o.float() for o in outs)
                    base = sum(
                        structure_loss(o, gt_c, variant=cfg.optim.structure_variant) for o in outs
                    ) if spec.deep_supervision else structure_loss(
                        combine_heads(outs, head=head), gt_c, variant=cfg.optim.structure_variant
                    )
                    loss = base / cfg.optim.grad_accum
                    tail_val = 0.0
                    if use_tail:
                        merged = combine_heads(outs, head=head)
                        deficits = per_image_deficit(
                            merged, gt_c, kind=cfg.pot.deficit_kind,
                            structure_variant=cfg.optim.structure_variant,
                        )
                        t_loss, tstats = tail(deficits, advance=True)
                        active_meter.update(tstats.get("tail/active", 0.0))
                        if tstats.get("tail/active", 0.0):
                            last_stats = tstats
                            for k, m in tail_meters.items():
                                v = tstats.get(f"tail/{k}")
                                if v is not None and np.isfinite(v):
                                    m.update(v)
                        loss = loss + t_loss / cfg.optim.grad_accum
                        tail_val = float(t_loss.detach())
                        if rate == 1.0 and cfg.pot.log_per_image:
                            for j, d in zip(chunk.tolist(), deficits.detach().cpu().tolist()):
                                epoch_deficits[int(index[j])] = float(d)
                    scaler.scale(loss).backward()
                    base_meter.update(float(base.detach()), chunk.numel())
                    if use_tail:
                        tail_meter.update(tail_val, chunk.numel())
                if cfg.optim.clip and cfg.optim.clip > 0:
                    scaler.unscale_(optimizer)
                    _clip_by_value(optimizer, cfg.optim.clip)
                scaler.step(optimizer)
                scaler.update()

            if step % cfg.run.log_every == 0 or step == steps_per_epoch:
                logger.info(
                    "epoch %3d/%d step %4d/%d lr %.2e base %.4f tail %.5f | u=%.4f xi=%s n_pool=%.0f lam=%.3f",
                    epoch, cfg.optim.epochs, step, steps_per_epoch, lr,
                    base_meter.avg, tail_meter.avg,
                    last_stats.get("tail/u", float("nan")),
                    ("%.3f" % last_stats["tail/xi"]) if "tail/xi" in last_stats else "-",
                    last_stats.get("tail/n_pool", 0.0), last_stats.get("tail/lam", 0.0),
                )

        if deficit_csv is not None and epoch_deficits:
            for i, d in sorted(epoch_deficits.items()):
                deficit_writer.writerow([epoch, i, train_items[i].stem, f"{d:.6f}"])
            deficit_csv.flush()

        rec = {
            "epoch": epoch, "lr": lr, "base_loss": base_meter.avg, "tail_loss": tail_meter.avg,
            "epoch_seconds": time.time() - t_epoch, "elapsed_seconds": time.time() - t_start,
            "tail_active_frac": active_meter.avg,
            **{f"tail_{k}": m.avg for k, m in tail_meters.items()},
        }
        # Deliberately outside the `did it ever fit` guard below: the case
        # worth warning about is the one where the tail term never fired, and
        # that is exactly the case where there are no fit statistics to print.
        # lam is nominal -- the weight a run actually applies is lam * active.
        # Early on a low fraction is a transient, because u is estimated from a
        # buffer of older, larger deficits and a fast-improving model rarely
        # exceeds it, so the check waits for warm-up to finish.
        if use_tail and cfg.pot.tail.mode != "none":
            warmup_epochs = math.ceil(cfg.pot.tail.warmup_frac * cfg.optim.epochs)
            expected = 100 * (1 - (1 - cfg.pot.tail.p) ** cfg.optim.batch_size)
            if epoch > max(1, warmup_epochs) and active_meter.avg < 0.5:
                logger.warning(
                    "the tail term fired on %.0f%% of calls this epoch, past the %d-epoch "
                    "warm-up. The effective weight is lam*active = %.3f, not the %.3f in the "
                    "config: on the other %.0f%% of steps no image in the batch reached the "
                    "pooled threshold, or the pool was too small to fit. Expect ~%.0f%% at "
                    "p=%.2f with batch %d. If it stays here, raise pot.tail.p, and do not "
                    "report lam as configured -- this arm is not the one the config describes.",
                    100 * active_meter.avg, max(1, warmup_epochs),
                    cfg.pot.tail.lam * active_meter.avg, cfg.pot.tail.lam,
                    100 * (1 - active_meter.avg), expected,
                    cfg.pot.tail.p, cfg.optim.batch_size,
                )

        if tail_meters["xi"].count:
            logger.info(
                "epoch %3d summary: base %.4f tail %.5f | active %.2f of calls, "
                "u=%.4f xi=%.3f (raw %.3f) beta=%.4f q_hat=%.4f degenerate=%.2f clamped=%.2f",
                epoch, base_meter.avg, tail_meter.avg, active_meter.avg,
                tail_meters["u"].avg, tail_meters["xi"].avg, tail_meters["xi_raw"].avg,
                tail_meters["beta"].avg, tail_meters["q_hat"].avg, tail_meters["degenerate"].avg,
                tail_meters["xi_clamped"].avg,
            )
            if tail_meters["xi_clamped"].avg > 0.2:
                logger.warning(
                    "the GPD shape clamp bound on %.0f%% of tail calls this epoch "
                    "(mean raw xi %.3f, bounds [%.2f, %.2f]). A binding clamp zeroes the "
                    "shape gradient and inverts the weight profile: widen pot.tail.xi_min / "
                    "xi_max before reporting this run.",
                    100 * tail_meters["xi_clamped"].avg, tail_meters["xi_raw"].avg,
                    cfg.pot.tail.xi_min, cfg.pot.tail.xi_max,
                )

        if val_items:
            _, val_agg = evaluate_split(
                model, root, val_items, device, head=head, size=cfg.data.size,
                threshold=cfg.eval.threshold, num_workers=cfg.eval.num_workers,
                compute_hd95=False, compute_sweep=False, amp_dtype=adtype,
            )
            rec["val_dice"] = val_agg["dice"]
            if cfg.run.select == "val_dice" and val_agg["dice"] > best_val:
                best_val = val_agg["dice"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_dice": best_val},
                           out_dir / "best.pth")
                rec["selected"] = True

        if cfg.eval.every and (epoch % cfg.eval.every == 0):
            probe = evaluate_all(
                model, root, manifest, device, head=head, splits=cfg.data.test_splits,
                size=cfg.data.size, threshold=cfg.eval.threshold,
                num_workers=cfg.eval.num_workers, batch_size=cfg.eval.batch_size,
                compute_hd95=False, compute_sweep=False, amp_dtype=adtype, save_root=None,
            )
            rec["probe"] = {k: v["dice"] for k, v in probe["per_split"].items()}

        jsonl.write(rec)
        if cfg.run.save_every and epoch % cfg.run.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": epoch}, out_dir / f"epoch{epoch}.pth")

    torch.save({"model": model.state_dict(), "epoch": cfg.optim.epochs,
                "tail": tail.state_dict()}, out_dir / "last.pth")

    if cfg.run.select == "val_dice" and (out_dir / "best.pth").is_file():
        state = safe_torch_load(out_dir / "best.pth")
        model.load_state_dict(state["model"])
        logger.info("selected checkpoint from epoch %d (val dice %.4f)", state["epoch"], state["val_dice"])
        selected = {"source": "val_dice", "epoch": state["epoch"], "val_dice": state["val_dice"],
                    "val_set": val_source, "val_pairs": len(val_items)}
    else:
        selected = {"source": "last", "epoch": cfg.optim.epochs}

    results = evaluate_all(
        model, root, manifest, device, head=head, splits=cfg.data.test_splits,
        size=cfg.data.size, threshold=cfg.eval.threshold,
        num_workers=cfg.eval.num_workers, batch_size=cfg.eval.batch_size,
        compute_hd95=cfg.eval.hd95, compute_sweep=cfg.eval.sweep, amp_dtype=adtype,
        save_root=(out_dir / "predictions") if cfg.eval.save_masks else None,
    )
    summary = {
        "run": cfg.run.name, "seed": cfg.run.seed, "model": cfg.model.name,
        "params": n_params, "selected": selected,
        "wall_clock_seconds": time.time() - t_start,
        "per_split": results["per_split"], "pooled_external": results["pooled_external"],
        "manifest": str(cfg.data.manifest),
        "tail_mode": cfg.pot.tail.mode, "tail_lam": cfg.pot.tail.lam,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out_dir / "per_image_metrics.json", "w") as fh:
        json.dump(results["per_image"], fh)
    jsonl.close()
    if deficit_csv is not None:
        deficit_csv.close()

    for split, agg in results["per_split"].items():
        logger.info(
            "%-32s mDice %.4f  mIoU %.4f  Recall %.4f  HD95 %.1f  Dice<=0.5 %.3f  (sweep mDice %.4f)",
            split, agg["dice"], agg["iou"], agg["recall"], agg["hd95"],
            agg["dice_le_05"], agg["dice_sweep"],
        )
    return summary


