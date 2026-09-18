"""Run provenance: seeds, device, git state, library versions."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

__all__ = ["seed_everything", "pick_device", "environment_report", "git_state", "amp_dtype"]


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        # Reproducible but slower; cuDNN autotuning is what makes runs differ
        # bit-for-bit even at a fixed seed.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # torch < 1.11
            torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(prefer: str = "cuda") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device(prefer if ":" in prefer else "cuda:0")
    return torch.device("cpu")


def amp_dtype(mode: str, device: torch.device):
    """Map ``off|fp16|bf16`` to a dtype, degrading safely off CUDA."""
    if mode == "off" or device.type != "cuda":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("optim.amp=bf16 but this GPU does not support bfloat16")
        return torch.bfloat16
    return torch.float16


def git_state(root: Path | str = ".") -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                args, cwd=str(root), stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return ""
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def environment_report(root: Path | str = ".") -> dict:
    rep = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git": git_state(root),
    }
    if torch.cuda.is_available():
        rep["gpu_name"] = torch.cuda.get_device_name(0)
        rep["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        rep["gpu_total_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
    return rep
