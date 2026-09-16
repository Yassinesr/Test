"""Typed configuration: YAML file + dotted command-line overrides.

Every run writes its fully-resolved config, the git commit, the manifest
hash and the library versions into its output directory, so a result can
always be traced back to the exact settings that produced it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from .eval.evaluator import DEFAULT_TEST_SPLITS
from .losses.tail import TailConfig

__all__ = ["Config", "DataConfig", "ModelConfig", "OptimConfig", "EvalConfig",
           "RunConfig", "TrainTailConfig", "load_config", "apply_overrides", "config_to_dict"]


@dataclass
class DataConfig:
    root: str = "./dataset"
    manifest: str = "manifests/pranet_protocol.json"
    train_split: str = "TrainDataset"
    test_splits: list[str] = field(default_factory=lambda: list(DEFAULT_TEST_SPLITS))
    size: int = 352
    augment: str = "none"
    """none (the released Polyp-PVT recipe) | flip | flip_rotate."""
    mask_interpolation: str = "bilinear"
    num_workers: int = 4
    verify: str = "exists"
    """off | exists | hash.  ``hash`` re-verifies every SHA-256 before training
    (~1 min for 2248 files) and is what a reportable run should use."""
    val_frac: float = 0.0
    """Fraction of the *training* split held out for model/lambda selection.
    0 disables it.  Selection on the test sets -- which the released
    ``Train.py`` does, checkpointing on test mDice every epoch -- is a
    protocol violation and is deliberately not implementable here."""
    val_seed: int = 12345


@dataclass
class ModelConfig:
    name: str = "polyp_pvt"
    channel: int = 32
    variant: str = "pvt_v2_b2"
    pretrained: Optional[str] = None
    """None means "the registry default path for this model"."""
    strict_pretrained: bool = True


@dataclass
class OptimConfig:
    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    batch_size: int = 16
    grad_accum: int = 1
    """Micro-batching for 12 GB cards: effective batch = batch_size * grad_accum."""
    clip: float = 0.5
    decay_rate: float = 0.1
    decay_epoch: int = 50
    scales: list[float] = field(default_factory=lambda: [0.75, 1.0, 1.25])
    """Multi-scale training, as in PraNet/Polyp-PVT.  Each scale is a separate
    optimiser step in the reference; that is reproduced exactly."""
    amp: str = "fp16"
    """off | fp16 | bf16.  fp16 roughly halves activation memory, which is what
    makes batch 16 at scale 1.25 fit on a 12 GB card."""
    structure_variant: str = "legacy"
    """legacy | weighted -- see polyptail/losses/structure.py."""


@dataclass
class TrainTailConfig:
    """The tail term plus how it attaches to the training loop."""

    tail: TailConfig = field(default_factory=TailConfig)
    deficit_kind: str = "soft_dice"
    head: Optional[str] = None
    """Which prediction the deficit is computed on.  None = the model
    registry's evaluation head, so the tail shapes what is actually scored."""
    apply_scales: str = "unit"
    """unit | all.  ``unit`` applies the tail term only at scale 1.0 so the
    deficit population stays homogeneous; at 0.75 small polyps are
    systematically harder, which would let the scale schedule -- not the data
    -- decide what lands in the tail."""
    log_per_image: bool = True
    """Write per-epoch per-image deficits to CSV.  This is how you tell
    "the tail is hard frames" from "the tail is label noise"."""


@dataclass
class EvalConfig:
    threshold: float = 0.5
    hd95: bool = True
    sweep: bool = True
    num_workers: int = 2
    batch_size: int = 1
    save_masks: bool = True
    every: int = 0
    """Epochs between mid-training evaluations (0 = final only).  Mid-training
    test evaluation is for curves only and must never drive selection."""


@dataclass
class RunConfig:
    name: str = "run"
    seed: int = 0
    out_dir: str = "./runs"
    deterministic: bool = False
    log_every: int = 20
    save_every: int = 0
    select: str = "last"
    """last | val_dice.  ``val_dice`` requires ``data.val_frac > 0``."""
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    pot: TrainTailConfig = field(default_factory=TrainTailConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    run: RunConfig = field(default_factory=RunConfig)

    def validate(self) -> None:
        if self.run.select == "val_dice" and self.data.val_frac <= 0:
            raise ValueError("run.select='val_dice' requires data.val_frac > 0")
        if self.data.verify not in ("off", "exists", "hash"):
            raise ValueError(f"data.verify must be off|exists|hash, got {self.data.verify!r}")
        if self.optim.amp not in ("off", "fp16", "bf16"):
            raise ValueError(f"optim.amp must be off|fp16|bf16, got {self.optim.amp!r}")
        if self.pot.apply_scales not in ("unit", "all"):
            raise ValueError(f"pot.apply_scales must be unit|all, got {self.pot.apply_scales!r}")
        if 1.0 not in [float(s) for s in self.optim.scales] and self.pot.apply_scales == "unit":
            raise ValueError("pot.apply_scales='unit' needs 1.0 in optim.scales")
        if self.optim.grad_accum < 1:
            raise ValueError("optim.grad_accum must be >= 1")
        # Re-run the dataclass validator after any override touched it.
        TailConfig(**asdict(self.pot.tail))


def _build(cls, data: dict):
    """Recursively construct nested dataclasses from plain dicts."""
    if not isinstance(data, dict):
        return data
    kwargs: dict[str, Any] = {}
    type_hints = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key not in type_hints:
            raise KeyError(f"unknown config key {key!r} for {cls.__name__}")
        f = type_hints[key]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), {**asdict(default), **value})
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: Optional[str | Path] = None, base: Optional[Config] = None) -> Config:
    """Load a YAML config.  ``_base:`` in the file pulls in another config first."""
    cfg = base or Config()
    if path is None:
        return cfg
    path = Path(path)
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("_base", None)
    if parent:
        cfg = load_config((path.parent / parent).resolve(), base=cfg)
    merged = _deep_merge(config_to_dict(cfg), raw)
    return _build(Config, merged)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(current: Any, text: str) -> Any:
    if isinstance(current, bool):
        low = text.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot read {text!r} as bool")
    if current is None:
        if text.lower() in ("none", "null"):
            return None
        try:
            return yaml.safe_load(text)
        except Exception:
            return text
    if isinstance(current, int) and not isinstance(current, bool):
        return int(text)
    if isinstance(current, float):
        return float(text)
    if isinstance(current, list):
        parsed = yaml.safe_load(text)
        return parsed if isinstance(parsed, list) else [parsed]
    return text


def apply_overrides(cfg: Config, overrides: Sequence[str]) -> Config:
    """Apply ``a.b.c=value`` strings in place and return the config."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        key, value = item.split("=", 1)
        node: Any = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if not hasattr(node, p):
                raise KeyError(f"unknown config path {key!r} (no {p!r})")
            node = getattr(node, p)
        leaf = parts[-1]
        if not hasattr(node, leaf):
            raise KeyError(f"unknown config path {key!r} (no {leaf!r})")
        setattr(node, leaf, _coerce(getattr(node, leaf), value))
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def dump_config(cfg: Config, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=True, default_flow_style=False)
