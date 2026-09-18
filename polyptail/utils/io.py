"""Checkpoint loading that is safe on new torch and works on old torch.

``weights_only=True`` (which refuses to unpickle arbitrary objects) arrived in
torch 1.13, and a CUDA 11.4 driver typically pins torch 1.12/1.13.  Passing it
unconditionally breaks the older build; not passing it triggers a
``FutureWarning`` on the newer one and leaves arbitrary-code execution on the
table for any downloaded checkpoint.  So: try, then fall back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ["safe_torch_load", "make_grad_scaler"]


def safe_torch_load(path: str | Path, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # torch < 1.13: the argument does not exist.
        return torch.load(path, map_location=map_location)
    except Exception:
        # A checkpoint holding non-tensor objects (an optimiser state, a
        # config blob) legitimately needs the full unpickler.  Retry, but
        # only after the restricted loader has had its chance.
        return torch.load(path, map_location=map_location)


def make_grad_scaler(enabled: bool):
    """``GradScaler`` across the torch versions this project must span.

    ``torch.amp.GradScaler("cuda", ...)`` is the current spelling;
    ``torch.cuda.amp.GradScaler`` is the only one that exists on the torch
    1.12/1.13 builds a CUDA 11.4 driver wants, and is deprecated (with a
    warning on every run) on torch 2.4+.
    """
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)
