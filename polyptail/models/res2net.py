"""Res2Net-50 v1b (26w x 4s) -- encoder for the PraNet control.

Faithful re-implementation of the official
``Res2Net-PretrainedModels/res2net_v1b.py`` so that the released ImageNet
checkpoint ``res2net50_v1b_26w_4s-3cf99910.pth`` loads with zero missing and
zero unexpected keys.  ``tests/test_models.py`` pins the parameter count, and
``load_res2net_checkpoint`` refuses a checkpoint that does not line up rather
than quietly training from scratch.

Checkpoint URL (download once, pass the local path):
https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from ..utils.io import safe_torch_load

logger = logging.getLogger(__name__)

__all__ = ["Bottle2neck", "Res2Net", "res2net50_v1b_26w_4s", "load_res2net_checkpoint"]


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype="normal"):
        super().__init__()
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
             for _ in range(self.nums)]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        spx = torch.split(out, self.width, 1)
        sp = None
        for i in range(self.nums):
            sp = spx[i] if (i == 0 or self.stype == "stage") else sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)
        if self.scale != 1:
            tail = spx[self.nums]
            out = torch.cat((out, self.pool(tail) if self.stype == "stage" else tail), 1)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class Res2Net(nn.Module):
    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample, stype="stage",
                        baseWidth=self.baseWidth, scale=self.scale)]
        self.inplanes = planes * block.expansion
        layers += [block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale)
                   for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def res2net50_v1b_26w_4s(pretrained: Optional[str | Path] = None, strict: bool = True) -> Res2Net:
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)
    if pretrained is not None:
        load_res2net_checkpoint(model, pretrained, strict=strict)
    return model


def load_res2net_checkpoint(model: nn.Module, path: str | Path, strict: bool = True) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Res2Net checkpoint not found: {path}\n"
            "Download res2net50_v1b_26w_4s-3cf99910.pth (see docs/REPRODUCIBILITY.md)."
        )
    ckpt = safe_torch_load(path)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    own = model.state_dict()
    missing = sorted(k for k in own if k not in ckpt)
    unexpected = sorted(k for k in ckpt if k not in own)
    shape_bad = sorted(k for k in own if k in ckpt and tuple(own[k].shape) != tuple(ckpt[k].shape))
    msg = (f"res2net checkpoint {path.name}: {len(own) - len(missing)}/{len(own)} keys loaded, "
           f"{len(missing)} missing, {len(unexpected)} unexpected, {len(shape_bad)} shape-mismatched")
    if missing or unexpected or shape_bad:
        detail = (f"{msg}\n  missing: {missing[:5]}\n  unexpected: {unexpected[:5]}"
                  f"\n  shape-mismatched: {shape_bad[:5]}")
        if strict:
            raise RuntimeError(detail)
        logger.warning(detail)
    else:
        logger.info(msg)
    model.load_state_dict({k: v for k, v in ckpt.items() if k in own and k not in shape_bad}, strict=False)
    return {"missing": missing, "unexpected": unexpected}
