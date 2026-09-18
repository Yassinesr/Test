"""Run a model over the frozen test splits and score it.

Resize direction, stated once and enforced here: the network runs at
``size x size``; its logits are bilinearly resized **up to the ground-truth
grid**; the sigmoid is taken there.  The ground truth is never resized.  The
opposite convention (downsampling the ground truth to 352) is common,
undocumented, and inflates scores on the high-resolution sets -- which are
exactly ETIS and CVC-ColonDB, the two that matter for this candidate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..data.dataset import PolypTestDataset, collate_eval, items_for_split
from ..losses.deficit import combine_heads
from .metrics import aggregate, image_metrics

__all__ = ["DEFAULT_TEST_SPLITS", "evaluate_split", "evaluate_all", "pooled_external"]

DEFAULT_TEST_SPLITS = (
    "TestDataset/Kvasir",
    "TestDataset/CVC-ClinicDB",
    "TestDataset/CVC-ColonDB",
    "TestDataset/CVC-300",
    "TestDataset/ETIS-LaribPolypDB",
)

#: Splits whose training data the model never saw.  "In-domain" here means
#: "drawn from the same source collection as the training split", which is the
#: strongest claim the PraNet manifest supports -- it is not a patient-level
#: guarantee, and §1.2.1 of the brief flags that CVC-300 may overlap
#: CVC-ColonDB, so the pooled external average is reported *with* that caveat.
EXTERNAL_SPLITS = (
    "TestDataset/CVC-ColonDB",
    "TestDataset/CVC-300",
    "TestDataset/ETIS-LaribPolypDB",
)


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    root: Path,
    items,
    device: torch.device,
    head: str = "sum",
    size: int = 352,
    threshold: float = 0.5,
    batch_size: int = 1,
    num_workers: int = 2,
    save_dir: Optional[Path] = None,
    compute_hd95: bool = True,
    compute_sweep: bool = True,
    amp_dtype: Optional[torch.dtype] = None,
) -> tuple[list[dict], dict]:
    """Score one split.  Returns ``(per_image_records, aggregate)``."""
    ds = PolypTestDataset(root, items, size=size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_eval, pin_memory=(device.type == "cuda"),
    )
    was_training = model.training
    model.eval()
    records: list[dict] = []
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        if amp_dtype is not None and device.type == "cuda":
            with torch.autocast("cuda", dtype=amp_dtype):
                outs = model(x)
            outs = tuple(o.float() for o in outs)
        else:
            outs = model(x)
        logits = combine_heads(outs, head=head)
        for i, gt_t in enumerate(batch["gt"]):
            gt = gt_t.numpy().astype(np.uint8)
            up = F.interpolate(
                logits[i: i + 1].float(), size=gt.shape, mode="bilinear", align_corners=False
            )
            prob = torch.sigmoid(up)[0, 0].cpu().numpy().astype(np.float32)
            rec = image_metrics(
                prob, gt, threshold=threshold,
                compute_hd95=compute_hd95, compute_sweep=compute_sweep,
            )
            rec["stem"] = batch["stem"][i]
            rec["name"] = batch["name"][i]
            records.append(rec)
            if save_dir is not None:
                # Retained prediction maps (blocking item 9): 8-bit probability
                # maps, *not* min-max normalised, so a third party can recompute
                # either scoring mode from what is on disk.
                Image.fromarray(np.rint(prob * 255.0).astype(np.uint8)).save(
                    Path(save_dir) / batch["name"][i]
                )
    if was_training:
        model.train()
    return records, aggregate(records)


def evaluate_all(
    model: torch.nn.Module,
    root: Path,
    manifest: dict,
    device: torch.device,
    head: str = "sum",
    splits: Sequence[str] = DEFAULT_TEST_SPLITS,
    save_root: Optional[Path] = None,
    **kwargs,
) -> dict:
    """Score every split and add the pooled external average."""
    out: dict = {"per_split": {}, "per_image": {}}
    for split in splits:
        items = items_for_split(manifest, split)
        save_dir = None if save_root is None else Path(save_root) / Path(split).name
        recs, agg = evaluate_split(
            model, root, items, device, head=head, save_dir=save_dir, **kwargs
        )
        out["per_split"][split] = agg
        out["per_image"][split] = recs
    out["pooled_external"] = pooled_external(out["per_image"], splits)
    return out


def pooled_external(per_image: dict, splits: Iterable[str]) -> dict:
    """Macro-average over the external splits, plus the pooled image-level tail.

    Two numbers, because they answer different questions: ``macro_*`` weights
    each dataset equally (what tables report), ``*`` weights each image equally
    (what a per-image tail statement is about).  §1.2.1's unresolved
    CVC-300 / CVC-ColonDB overlap means the pooled figures may double-count
    frames; ``tools/hash_collisions.py`` settles that before either is used.
    """
    present = [s for s in splits if s in EXTERNAL_SPLITS and s in per_image]
    if not present:
        return {"n_splits": 0}
    flat = [r for s in present for r in per_image[s]]
    agg = aggregate(flat)
    agg["n_splits"] = len(present)
    agg["splits"] = list(present)
    for key in ("dice", "iou", "recall", "hd95", "dice_sweep"):
        vals = [aggregate(per_image[s])[key] for s in present]
        vals = [v for v in vals if np.isfinite(v)]
        agg[f"macro_{key}"] = float(np.mean(vals)) if vals else float("nan")
    return agg
