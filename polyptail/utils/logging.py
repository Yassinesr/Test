"""Minimal run logging: a console logger plus an append-only JSONL record."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

__all__ = ["setup_logging", "JsonlWriter", "AvgMeter"]


def setup_logging(out_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("[%(asctime)s %(levelname)s %(name)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(out_dir) / "train.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    return root


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", buffering=1)

    def write(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, default=float) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AvgMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
