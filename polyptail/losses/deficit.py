"""Per-image scalar deficits -- the only thing POT-TC reads.

The tail objective is *field-agnostic by construction*: it never sees the
image, the colour statistics, the polyp geometry or any metadata.  It sees one
non-negative scalar per training image, and that is the whole interface.

``soft_dice_deficit`` is the default: ``d_i = 1 - softDice_i`` on the
sigmoid of the logits, so ``d_i`` lies in ``[0, 1]`` and is a direct
differentiable proxy for the quantity the benchmark reports.  The prediction
used is the one that inference actually uses (see ``head`` below), so the
tail term shapes what is ultimately evaluated rather than an auxiliary head.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .structure import structure_loss_per_image

__all__ = ["soft_dice_deficit", "per_image_deficit", "combine_heads"]


def combine_heads(outputs: tuple[Tensor, ...] | list[Tensor], head: str = "sum") -> Tensor:
    """Reduce a model's multi-head logits to the single map used downstream.

    ``"sum"`` matches Polyp-PVT inference, which evaluates ``sigmoid(P1 + P2)``.
    ``"last"`` uses the final head only.  ``"mean"`` averages the logits.
    """
    outs = list(outputs)
    if head == "sum":
        return torch.stack(outs, 0).sum(0)
    if head == "mean":
        return torch.stack(outs, 0).mean(0)
    if head == "last":
        return outs[-1]
    raise ValueError(f"unknown head reduction {head!r}")


def soft_dice_deficit(logits: Tensor, mask: Tensor, smooth: float = 1.0) -> Tensor:
    """``1 - softDice`` per image.  Returns ``[B]``, values in ``[0, 1]``.

    ``logits``/``mask`` are ``[B, 1, H, W]``.  The smoothing constant matches
    the reference evaluation code (``smooth = 1``); at 352x352 it shifts Dice
    by O(1e-5) and exists only to define the empty-mask case.
    """
    pred = torch.sigmoid(logits.float())
    m = mask.float()
    inter = (pred * m).sum(dim=(1, 2, 3))
    total = pred.sum(dim=(1, 2, 3)) + m.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + smooth) / (total + smooth)
    return (1.0 - dice).clamp(0.0, 1.0)


def per_image_deficit(
    logits: Tensor,
    mask: Tensor,
    kind: str = "soft_dice",
    structure_variant: str = "legacy",
) -> Tensor:
    """Dispatch for the deficit definition.

    ``"soft_dice"`` (default) is bounded in ``[0, 1]`` and is what the
    candidate card specifies.  ``"structure"`` uses the per-image base loss
    instead; it is unbounded above, which makes the GPD shape estimate
    noticeably less stable, so it is offered only for ablation.
    """
    if kind == "soft_dice":
        return soft_dice_deficit(logits, mask)
    if kind == "structure":
        return structure_loss_per_image(logits, mask, variant=structure_variant)
    raise ValueError(f"unknown deficit kind {kind!r}")
