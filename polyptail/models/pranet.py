"""PraNet (Fan et al., MICCAI 2020) -- the secondary, low-capacity control.

The brief nominates PraNet as the secondary baseline (§2.1) because its
protocol numbers have four independent readings in the literature, which makes
its reproduction tolerance measurable.  It is also the A7 transfer check: if
POT-TC helps on a PVT encoder but not on a Res2Net one, the effect is
backbone-specific and the claim is much weaker.

Faithful to ``lib/PraNet_Res2Net.py`` in https://github.com/DengPingFan/PraNet.
Returns the four deep-supervision maps ``(map5, map4, map3, map2)``; inference
uses ``map2``, so the POT-TC deficit head for this model is ``"last"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .res2net import res2net50_v1b_26w_4s

__all__ = ["PraNet"]


class BasicConv2d(nn.Module):
    """conv + BN (no activation) -- as in the released PraNet."""

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class RFB_modified(nn.Module):
    """Receptive-field block: four dilated branches plus a 1x1 residual."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        cat = torch.cat((self.branch0(x), self.branch1(x), self.branch2(x), self.branch3(x)), 1)
        return self.relu(self.conv_cat(cat) + self.conv_res(x))


class Aggregation(nn.Module):
    """Partial decoder (Wu et al. CVPR 2019) over the three RFB outputs."""

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
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
            * self.conv_upsample3(self.upsample(x2)) * x3
        x2_2 = self.conv_concat2(torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1))
        x3_2 = self.conv_concat3(torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1))
        return self.conv5(self.conv4(x3_2))


class PraNet(nn.Module):
    n_heads = 4

    def __init__(self, channel: int = 32, pretrained: Optional[str | Path] = None,
                 strict_pretrained: bool = True):
        super().__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=pretrained, strict=strict_pretrained)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.agg1 = Aggregation(channel)

        self.ra4_conv1 = BasicConv2d(2048, 256, 1)
        self.ra4_conv2 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, 5, padding=2)
        self.ra4_conv5 = nn.Conv2d(256, 1, 1)

        self.ra3_conv1 = BasicConv2d(1024, 64, 1)
        self.ra3_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra3_conv4 = nn.Conv2d(64, 1, 1)

        self.ra2_conv1 = BasicConv2d(512, 64, 1)
        self.ra2_conv2 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, 3, padding=1)
        self.ra2_conv4 = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        r = self.resnet
        x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
        x1 = r.layer1(x)          # 256,  1/4
        x2 = r.layer2(x1)         # 512,  1/8
        x3 = r.layer3(x2)         # 1024, 1/16
        x4 = r.layer4(x3)         # 2048, 1/32

        ra5_feat = self.agg1(self.rfb4_1(x4), self.rfb3_1(x3), self.rfb2_1(x2))  # 1/8
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode="bilinear", align_corners=False)

        crop_4 = F.interpolate(ra5_feat, size=x4.shape[-2:], mode="bilinear", align_corners=False)
        ra = (1.0 - torch.sigmoid(crop_4)).expand(-1, 2048, -1, -1).mul(x4)
        ra = self.ra4_conv1(ra)
        ra = F.relu(self.ra4_conv2(ra))
        ra = F.relu(self.ra4_conv3(ra))
        ra = F.relu(self.ra4_conv4(ra))
        x = self.ra4_conv5(ra) + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode="bilinear", align_corners=False)

        crop_3 = F.interpolate(x, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        ra = (1.0 - torch.sigmoid(crop_3)).expand(-1, 1024, -1, -1).mul(x3)
        ra = self.ra3_conv1(ra)
        ra = F.relu(self.ra3_conv2(ra))
        ra = F.relu(self.ra3_conv3(ra))
        x = self.ra3_conv4(ra) + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)

        crop_2 = F.interpolate(x, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        ra = (1.0 - torch.sigmoid(crop_2)).expand(-1, 512, -1, -1).mul(x2)
        ra = self.ra2_conv1(ra)
        ra = F.relu(self.ra2_conv2(ra))
        ra = F.relu(self.ra2_conv3(ra))
        x = self.ra2_conv4(ra) + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode="bilinear", align_corners=False)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
