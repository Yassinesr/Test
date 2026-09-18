"""Base segmentation objective: boundary-weighted BCE + boundary-weighted IoU.

This is the loss PraNet introduced and Polyp-PVT inherited.  The weight map
``1 + 5 * |avgpool_31(y) - y|`` is large exactly on mask boundaries, so both
terms are up-weighted where the annotation changes.

A NOTE ON REPRODUCIBILITY THAT MATTERS
--------------------------------------
The released PraNet/Polyp-PVT code calls

    F.binary_cross_entropy_with_logits(pred, mask, reduce='none')

``reduce`` is a deprecated *boolean* argument.  The non-empty string
``'none'`` is truthy, so PyTorch resolves it to ``reduction='mean'`` and
``wbce`` comes back as a **scalar**.  The subsequent
``(weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))`` then reduces to
``wbce`` exactly, because the scalar factors straight out of the ratio.  In
other words: in the code that produced the published numbers, the BCE term is
**unweighted mean BCE**, not the weighted BCE the paper describes.  Modern
PyTorch raises on that call rather than silently accepting it.

Both behaviours are implemented, selected by ``variant``:

    "legacy"  -- reproduces the released artefact (plain mean BCE).  This is
                 the default, because §2.1 of the brief requires reproducing
                 Polyp-PVT to within +/-0.5 mDice of its published table, and
                 that table was produced by the released code.
    "weighted"-- the weighted BCE the paper text describes.

Anyone comparing against published Polyp-PVT numbers should keep "legacy";
anyone reporting a new baseline of their own should say which they used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["structure_loss", "structure_loss_per_image", "boundary_weight"]


def boundary_weight(mask: Tensor, kernel_size: int = 31, gain: float = 5.0) -> Tensor:
    """``1 + gain * |avgpool_k(y) - y|`` -- large on mask boundaries."""
    pad = kernel_size // 2
    smooth = F.avg_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    return 1.0 + gain * torch.abs(smooth - mask)


def structure_loss_per_image(
    logits: Tensor,
    mask: Tensor,
    variant: str = "legacy",
    eps: float = 1.0,
) -> Tensor:
    """Per-image structure loss.  Returns shape ``[B]``.

    ``logits`` and ``mask`` are ``[B, 1, H, W]``; ``mask`` in ``[0, 1]``.
    """
    if variant not in ("legacy", "weighted"):
        raise ValueError(f"unknown structure-loss variant {variant!r}")
    weit = boundary_weight(mask)

    bce_map = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    if variant == "weighted":
        wbce = (weit * bce_map).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    else:
        # Reproduces the released code's effective behaviour exactly: a plain
        # mean over all pixels, per image.
        wbce = bce_map.mean(dim=(2, 3))

    pred = torch.sigmoid(logits)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1.0 - (inter + eps) / (union - inter + eps)
    # wbce and wiou are [B, C]; masks here are always single-channel, so the
    # channel mean is an identity that keeps the return shape [B] either way.
    return (wbce + wiou).mean(dim=1)


def structure_loss(logits: Tensor, mask: Tensor, variant: str = "legacy") -> Tensor:
    """Batch-mean structure loss (scalar)."""
    return structure_loss_per_image(logits, mask, variant=variant).mean()
