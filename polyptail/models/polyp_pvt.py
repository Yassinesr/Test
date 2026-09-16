"""Polyp-PVT (Dong et al., CAAI AIR 2023) -- the primary reference baseline.

Adapted from the official release (https://github.com/DengPingFan/Polyp-PVT,
``lib/pvt.py``).  The computation is unchanged; what changed is the parts that
make a *reproduction* auditable:

* the pretrained-backbone path is a constructor argument, not a hard-coded
  ``'./pretrained_pth/pvt_v2_b2.pth'``;
* loading is **strict and reported**.  The reference does

      state_dict = {k: v for k, v in save_model.items() if k in model_dict}

  which silently drops every key that does not match.  Point it at the wrong
  checkpoint and it loads *nothing*, trains from scratch, and says nothing.
  ``load_pretrained_backbone`` raises on any missing or unexpected key unless
  explicitly told not to, and always logs the counts.
* ``F.upsample`` (removed-in-spirit alias) is ``F.interpolate``.

The model returns ``(P1, P2)``: the CFM head and the SAM head.  Inference and
the deficit both use ``P1 + P2``, matching the released ``Test.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.io import safe_torch_load
from .pvtv2 import pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5

logger = logging.getLogger(__name__)

_PVT_VARIANTS = {
    "pvt_v2_b0": (pvt_v2_b0, [32, 64, 160, 256]),
    "pvt_v2_b1": (pvt_v2_b1, [64, 128, 320, 512]),
    "pvt_v2_b2": (pvt_v2_b2, [64, 128, 320, 512]),
    "pvt_v2_b3": (pvt_v2_b3, [64, 128, 320, 512]),
    "pvt_v2_b4": (pvt_v2_b4, [64, 128, 320, 512]),
    "pvt_v2_b5": (pvt_v2_b5, [64, 128, 320, 512]),
}


def load_pretrained_backbone(
    backbone: nn.Module, path: Optional[str | Path], strict: bool = True
) -> dict:
    """Load an ImageNet checkpoint into ``backbone``, loudly.

    Returns a dict with ``loaded``/``missing``/``unexpected`` key counts.
    Raises ``RuntimeError`` under ``strict`` if anything did not line up --
    a silently-empty load is the single easiest way to produce a "reproduction
    gap" that has nothing to do with the method.
    """
    if path is None:
        logger.warning("No pretrained backbone given; training the encoder from scratch.")
        return {"loaded": 0, "missing": [], "unexpected": []}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"pretrained backbone not found: {path}\n"
            "Download pvt_v2_b2.pth from the Polyp-PVT release and place it there "
            "(see docs/REPRODUCIBILITY.md)."
        )
    ckpt = safe_torch_load(path)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}

    own = backbone.state_dict()
    # The classification head is legitimately absent from a segmentation
    # backbone; everything else must match.
    ignorable = {k for k in ckpt if k.startswith("head.")}
    missing = sorted(k for k in own if k not in ckpt)
    unexpected = sorted(k for k in ckpt if k not in own and k not in ignorable)
    shape_bad = sorted(k for k in own if k in ckpt and tuple(own[k].shape) != tuple(ckpt[k].shape))

    msg = (
        f"backbone checkpoint {path.name}: {len(own) - len(missing)}/{len(own)} keys loaded, "
        f"{len(missing)} missing, {len(unexpected)} unexpected, {len(shape_bad)} shape-mismatched"
    )
    if missing or unexpected or shape_bad:
        detail = (
            f"{msg}\n  missing (first 5): {missing[:5]}"
            f"\n  unexpected (first 5): {unexpected[:5]}"
            f"\n  shape-mismatched (first 5): {shape_bad[:5]}"
        )
        if strict:
            raise RuntimeError(detail)
        logger.warning(detail)
    else:
        logger.info(msg)
    backbone.load_state_dict({k: v for k, v in ckpt.items() if k in own and k not in shape_bad}, strict=False)
    return {"loaded": len(own) - len(missing), "missing": missing, "unexpected": unexpected}


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # NOTE: the released code applies conv+bn and *not* the ReLU here.
        # Kept as-is; changing it changes the reproduced baseline.
        return self.bn(self.conv(x))


class CFM(nn.Module):
    """Cascaded fusion module."""

    def __init__(self, channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
            * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv4(x3_2)


class GCN(nn.Module):
    def __init__(self, num_state, num_node, bias=False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SAM(nn.Module):
    """Similarity aggregation module."""

    def __init__(self, num_in=32, plane_mid=16, mids=4, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))
        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x, edge):
        edge = F.interpolate(edge, size=(x.size(-2), x.size(-1)), mode="nearest")
        n, c, h, w = x.size()
        # As released: a softmax over the 32 translayer channels, channel 1 taken
        # as the "edge" gate.  Idiosyncratic, but it is what the published
        # weights and numbers correspond to.
        edge = torch.softmax(edge, dim=1)[:, 1, :, :].unsqueeze(1)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)
        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.softmax(x_proj_reshaped, dim=1)

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1.0 / x_state_reshaped.size(2))
        x_n_rel = self.gcn(x_n_state)
        x_state = torch.matmul(x_n_rel, x_proj_reshaped).view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class PolypPVT(nn.Module):
    """PVTv2 encoder + CIM / CFM / SAM decoder.  Returns ``(P1, P2)`` logits."""

    n_heads = 2

    def __init__(
        self,
        channel: int = 32,
        variant: str = "pvt_v2_b2",
        pretrained: Optional[str | Path] = "./pretrained_pth/pvt_v2_b2.pth",
        strict_pretrained: bool = True,
    ) -> None:
        super().__init__()
        if variant not in _PVT_VARIANTS:
            raise ValueError(f"unknown PVT variant {variant!r}")
        ctor, dims = _PVT_VARIANTS[variant]
        self.backbone = ctor()
        self.pretrained_report = load_pretrained_backbone(
            self.backbone, pretrained, strict=strict_pretrained
        )

        c1, c2, c3, c4 = dims
        self.Translayer2_0 = BasicConv2d(c1, channel, 1)
        self.Translayer2_1 = BasicConv2d(c2, channel, 1)
        self.Translayer3_1 = BasicConv2d(c3, channel, 1)
        self.Translayer4_1 = BasicConv2d(c4, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(c1)
        self.sa = SpatialAttention()
        self.SAM = SAM(num_in=channel)

        self.down05 = nn.Upsample(scale_factor=0.5, mode="bilinear", align_corners=True)
        self.out_SAM = nn.Conv2d(channel, 1, 1)
        self.out_CFM = nn.Conv2d(channel, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        # CIM: channel then spatial attention on the shallowest stage.
        x1 = self.ca(x1) * x1
        cim_feature = self.sa(x1) * x1

        # CFM over the three deeper stages.
        cfm_feature = self.CFM(self.Translayer4_1(x4), self.Translayer3_1(x3), self.Translayer2_1(x2))

        # SAM couples the two.
        sam_feature = self.SAM(cfm_feature, self.down05(self.Translayer2_0(cim_feature)))

        p1 = F.interpolate(self.out_CFM(cfm_feature), scale_factor=8, mode="bilinear", align_corners=False)
        p2 = F.interpolate(self.out_SAM(sam_feature), scale_factor=8, mode="bilinear", align_corners=False)
        return p1, p2
