"""A ~0.5M-parameter U-Net used only by the test suite and the CPU smoke run.

It exists so that the whole training/eval/statistics pipeline can be exercised
end to end without a GPU, a 25M-parameter transformer or a pretrained
checkpoint.  It is never a baseline; ``tools/run_ablation.py`` refuses to use
it for anything reported.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TinyUNet"]


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    n_heads = 2

    def __init__(self, width: int = 16, **_ignored):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = _block(3, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.d2, self.d1 = _block(4 * w + 2 * w, 2 * w), _block(2 * w + w, w)
        self.head_main = nn.Conv2d(w, 1, 1)
        self.head_aux = nn.Conv2d(2 * w, 1, 1)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(F.max_pool2d(s1, 2))
        s3 = self.e3(F.max_pool2d(s2, 2))
        u2 = self.d2(torch.cat([F.interpolate(s3, size=s2.shape[-2:], mode="bilinear", align_corners=False), s2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, size=s1.shape[-2:], mode="bilinear", align_corners=False), s1], 1))
        aux = F.interpolate(self.head_aux(u2), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.head_main(u1), aux
