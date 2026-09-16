"""Model registry.

Every entry declares which head reduction the *deficit* and the *evaluation*
must use, so that the tail term shapes exactly the prediction the benchmark
scores.  Getting that wrong -- shaping an auxiliary head while evaluating an
ensemble -- is a silent way to make an ablation meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch.nn as nn

from .polyp_pvt import PolypPVT, load_pretrained_backbone
from .pranet import PraNet
from .res2net import res2net50_v1b_26w_4s
from .tiny_unet import TinyUNet

__all__ = ["ModelSpec", "MODELS", "build_model", "available_models"]


@dataclass(frozen=True)
class ModelSpec:
    ctor: Callable[..., nn.Module]
    head: str
    """How to reduce the model's outputs into the single evaluated map."""
    default_pretrained: Optional[str]
    deep_supervision: bool
    """Whether the base loss is applied to every head (PraNet/Polyp-PVT do)."""
    reportable: bool = True


MODELS: dict[str, ModelSpec] = {
    "polyp_pvt": ModelSpec(
        ctor=PolypPVT, head="sum",
        default_pretrained="./pretrained_pth/pvt_v2_b2.pth", deep_supervision=True,
    ),
    "pranet": ModelSpec(
        ctor=PraNet, head="last",
        default_pretrained="./pretrained_pth/res2net50_v1b_26w_4s-3cf99910.pth",
        deep_supervision=True,
    ),
    "tiny_unet": ModelSpec(
        ctor=TinyUNet, head="sum", default_pretrained=None,
        deep_supervision=True, reportable=False,
    ),
}


def available_models() -> list[str]:
    return sorted(MODELS)


def build_model(name: str, **kwargs) -> tuple[nn.Module, ModelSpec]:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; available: {available_models()}")
    spec = MODELS[name]
    return spec.ctor(**kwargs), spec
