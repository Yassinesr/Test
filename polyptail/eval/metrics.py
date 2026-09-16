"""The one evaluation implementation.

§1.5 item 3 of the brief requires exactly this: a single file, stating the
threshold, the resize direction, the aggregation, and the metric set.  Two
scoring semantics are implemented because two are genuinely needed:

``fixed`` (PRIMARY, and what the locked protocol reports)
    sigmoid -> resize to the **ground-truth** grid -> threshold at 0.5.
    No min-max normalisation.  This is the number a clinician-facing system
    would actually get, and it is the only one for which "Recall" means what
    it says.

``sweep_minmax`` (COMPATIBILITY, and what published tables contain)
    The PraNet MATLAB toolbox semantics that Polyp-PVT, PraNet, M2SNet and
    almost every table in §2 of the brief were produced with: the saved
    probability map is min-max normalised to [0, 1], quantised to 8 bits (it
    is written to a PNG), scored at all 256 thresholds ``t = k/255``, and the
    per-image Dice/IoU is the **average over thresholds**.  Without this mode
    you cannot check a reproduction against any published number, and §2.1
    makes that check blocking.

Reporting only ``fixed`` would make the work incomparable to the literature;
reporting only ``sweep_minmax`` would report a threshold-free summary as if
it were an operating point.  Both are computed in one pass and both are
written out.  Min-max normalisation is not a neutral rescaling: it forces
every map to span [0, 1], so a confidently-empty prediction is stretched into
a confident one.  That is one reason external-set numbers look better under
the compatibility mode than under the fixed one.

Aggregation is **per-image mean** throughout ("m" as in mDice), stated
explicitly because §1.3 W8 shows most papers do not say.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

__all__ = [
    "METRIC_KEYS",
    "confusion",
    "binary_scores",
    "hd95",
    "sweep_scores",
    "image_metrics",
    "aggregate",
]

METRIC_KEYS = (
    "dice", "iou", "recall", "precision", "specificity", "hd95",
    "dice_sweep", "iou_sweep",
)

_STRUCT = ndimage.generate_binary_structure(2, 1)


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    tn = int(p.size - tp - fp - fn)
    return tp, fp, fn, tn


def binary_scores(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Dice/IoU/Recall/Precision/Specificity at a fixed binarisation.

    Empty-mask convention, stated rather than smoothed away:
      * both empty  -> dice = iou = 1.0 (a correct "nothing here");
      * gt empty, pred not -> dice = iou = 0.0, recall = nan;
      * pred empty, gt not -> dice = iou = recall = 0.0, precision = nan.
    No epsilon is added to the numerator: a smoothing constant would turn a
    total failure into a small positive score and quietly lift the very left
    tail this project is about.
    """
    tp, fp, fn, tn = confusion(pred, gt)
    p_sum, g_sum = tp + fp, tp + fn
    if p_sum == 0 and g_sum == 0:
        dice = iou = 1.0
    else:
        dice = 2.0 * tp / (p_sum + g_sum) if (p_sum + g_sum) else 0.0
        union = p_sum + g_sum - tp
        iou = tp / union if union else 0.0
    return {
        "dice": float(dice),
        "iou": float(iou),
        "recall": float(tp / g_sum) if g_sum else float("nan"),
        "precision": float(tp / p_sum) if p_sum else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pred_empty": bool(p_sum == 0), "gt_empty": bool(g_sum == 0),
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    return m ^ ndimage.binary_erosion(m, structure=_STRUCT, border_value=1)


def hd95(pred: np.ndarray, gt: np.ndarray, empty_value: Optional[float] = None) -> float:
    """95th-percentile symmetric surface distance, in pixels of the GT grid.

    Follows the ``medpy.metric.binary.hd95`` convention: pool the directed
    surface distances both ways and take the 95th percentile of the pooled
    set (not the max of the two directed 95th percentiles).  Stated because
    the two differ and papers rarely say which they used.

    If either mask is empty the distance is undefined.  Rather than dropping
    the image (which would silently make the metric optimistic on exactly the
    failures that matter) it returns ``empty_value``, defaulting to the image
    diagonal -- the largest distance the grid admits.  Callers get the count
    of such images alongside the mean.
    """
    p, g = pred.astype(bool), gt.astype(bool)
    if empty_value is None:
        empty_value = float(math.hypot(*pred.shape))
    if not p.any() or not g.any():
        return 0.0 if (not p.any() and not g.any()) else float(empty_value)
    sp, sg = _surface(p), _surface(g)
    dt_to_g = ndimage.distance_transform_edt(~sg)
    dt_to_p = ndimage.distance_transform_edt(~sp)
    pooled = np.hstack((dt_to_g[sp], dt_to_p[sg]))
    return float(np.percentile(pooled, 95))


def sweep_scores(prob: np.ndarray, gt: np.ndarray, normalize: bool = True) -> dict:
    """Threshold-averaged Dice/IoU, reproducing the PraNet MATLAB pipeline.

    The reference pipeline min-max normalises the probability map, writes it
    to an 8-bit PNG, and averages Dice/IoU over the 256 thresholds
    ``t = k/255, k = 0..255`` under the predicate ``map >= t``.  Because the
    map really is 8-bit by then, the sweep is *exact* via two 256-bin
    histograms and a reverse cumulative sum -- O(N + 256) instead of
    O(256 N), which is what makes native-resolution ETIS evaluation cheap.
    """
    p = prob.astype(np.float64)
    if normalize:
        lo, hi = float(p.min()), float(p.max())
        p = (p - lo) / (hi - lo + 1e-8)
    q = np.rint(np.clip(p, 0.0, 1.0) * 255.0).astype(np.int64)
    g = gt.astype(bool)

    h_pos = np.bincount(q[g], minlength=256).astype(np.float64)
    h_neg = np.bincount(q[~g], minlength=256).astype(np.float64)
    # counts at "quantised value >= k", for k = 0..255
    tp = np.cumsum(h_pos[::-1])[::-1]
    fp = np.cumsum(h_neg[::-1])[::-1]
    n_pos = float(h_pos.sum())
    fn = n_pos - tp

    denom_d = 2.0 * tp + fp + fn
    dice = np.where(denom_d > 0, 2.0 * tp / np.maximum(denom_d, 1e-12), 1.0)
    denom_i = tp + fp + fn
    iou = np.where(denom_i > 0, tp / np.maximum(denom_i, 1e-12), 1.0)
    return {"dice_sweep": float(dice.mean()), "iou_sweep": float(iou.mean())}


def image_metrics(
    prob: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.5,
    compute_hd95: bool = True,
    compute_sweep: bool = True,
) -> dict:
    """All metrics for one image.  ``prob`` and ``gt`` must share a shape."""
    if prob.shape != gt.shape:
        raise ValueError(f"prob shape {prob.shape} != gt shape {gt.shape}")
    gt_b = gt.astype(bool)
    pred_b = prob >= threshold
    out = binary_scores(pred_b, gt_b)
    out["hd95"] = hd95(pred_b, gt_b) if compute_hd95 else float("nan")
    if compute_sweep:
        out.update(sweep_scores(prob, gt_b, normalize=True))
    else:
        out["dice_sweep"] = float("nan")
        out["iou_sweep"] = float("nan")
    out["height"], out["width"] = int(gt.shape[0]), int(gt.shape[1])
    return out


def aggregate(records: Sequence[dict], tail_levels: Sequence[float] = (0.5, 0.75)) -> dict:
    """Per-image means plus the left-tail diagnostics the candidate needs.

    ``dice_le_XX`` is the fraction of images whose fixed-threshold Dice is at
    or below ``XX`` -- the quantity F1 of the brief measures (21-31% of frames
    at DSC <= 0.5) and the quantity the falsification criterion in §9 of the
    candidate card is written against.  ``dice_p05``/``dice_p10`` are the 5th
    and 10th percentiles of the per-image Dice, i.e. the tail itself rather
    than a count.
    """
    if not records:
        return {"n": 0}
    out: dict = {"n": len(records)}
    for key in METRIC_KEYS:
        vals = np.array([r.get(key, np.nan) for r in records], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        out[key] = float(finite.mean()) if finite.size else float("nan")
        out[f"{key}_n_defined"] = int(finite.size)
    dice = np.array([r["dice"] for r in records], dtype=np.float64)
    for lvl in tail_levels:
        out[f"dice_le_{lvl:g}".replace(".", "")] = float((dice <= lvl).mean())
    out["dice_p05"] = float(np.percentile(dice, 5))
    out["dice_p10"] = float(np.percentile(dice, 10))
    out["dice_sd"] = float(dice.std(ddof=1)) if len(dice) > 1 else 0.0
    out["n_pred_empty"] = int(sum(bool(r.get("pred_empty")) for r in records))
    out["n_hd95_undefined"] = int(
        sum(1 for r in records if r.get("pred_empty") or r.get("gt_empty"))
    )
    return out
