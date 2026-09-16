"""Manifest-driven datasets.

Two deliberate departures from the reference implementation, both of which
are correctness fixes rather than preferences:

1.  **Pairing is by stem, from the manifest.**  The reference sorts the
    ``images/`` and ``masks/`` listings independently and zips them.  That is
    correct only while both directories contain exactly the same stems with
    the same sort order; any extension difference, stray file or case
    mismatch silently trains the model on mis-paired masks.

2.  **Augmentation is applied pairwise with explicit parameters.**  The
    reference draws a seed and calls ``random.seed`` / ``torch.manual_seed``
    inside ``__getitem__`` to make the image and mask transforms agree.  In a
    ``num_workers > 0`` loader that reseeds the worker's *global* RNG on every
    sample, which destroys the intended per-worker stream and makes runs
    irreproducible in a way that no seed setting can fix.  Here the transform
    parameters are drawn once and applied to both tensors.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .manifest import ManifestItem, binarize_mask

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PolypTrainDataset",
    "PolypTestDataset",
    "items_for_split",
    "collate_eval",
    "worker_init_fn",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def items_for_split(manifest: dict, split: str) -> list[ManifestItem]:
    if split not in manifest["splits"]:
        raise KeyError(f"split {split!r} not in manifest (have {sorted(manifest['splits'])})")
    return [ManifestItem.from_json(d) for d in manifest["splits"][split]["items"]]


def _to_tensor_rgb(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    t = torch.from_numpy(arr.copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def _to_tensor_mask(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.uint8)
    return torch.from_numpy(arr.copy()).unsqueeze(0).float().div_(255.0)


class PolypTrainDataset(data.Dataset):
    """Training pairs at a fixed square resolution.

    ``__getitem__`` returns ``(image, mask, index)``.  The index is not
    decoration: it is what lets the trainer write out *which* training images
    populate the tail over time, which is the only cheap way to tell the
    difference between "the tail is genuinely hard frames" and "the tail is
    label noise" -- the failure mode the candidate card lists first.
    """

    def __init__(
        self,
        root: Path,
        items: Sequence[ManifestItem],
        size: int = 352,
        augment: str = "none",
        mask_interpolation: str = "bilinear",
    ) -> None:
        if augment not in ("none", "flip", "flip_rotate"):
            raise ValueError(f"unknown augment {augment!r}")
        if mask_interpolation not in ("bilinear", "nearest"):
            raise ValueError(f"unknown mask_interpolation {mask_interpolation!r}")
        self.root = Path(root)
        self.items = list(items)
        self.size = int(size)
        self.augment = augment
        self._mask_resample = Image.BILINEAR if mask_interpolation == "bilinear" else Image.NEAREST

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        it = self.items[index]
        img = Image.open(self.root / it.image).convert("RGB")
        msk = Image.open(self.root / it.mask).convert("L")

        if self.augment != "none":
            img, msk = self._augment(img, msk)

        img = img.resize((self.size, self.size), Image.BILINEAR)
        msk = msk.resize((self.size, self.size), self._mask_resample)
        return _to_tensor_rgb(img), _to_tensor_mask(msk), index

    def _augment(self, img: Image.Image, msk: Image.Image):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            msk = msk.transpose(Image.FLIP_TOP_BOTTOM)
        if self.augment == "flip_rotate":
            angle = random.uniform(-90.0, 90.0)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
            msk = msk.rotate(angle, resample=Image.NEAREST, expand=False)
        return img, msk


class PolypTestDataset(data.Dataset):
    """Test images at ``size``; ground truth kept at **native** resolution.

    The locked protocol resizes *predictions up to the ground-truth grid*,
    never the ground truth down to the network grid, so the ground truth is
    returned unresized as a ``uint8`` array in ``{0, 1}``.
    """

    def __init__(self, root: Path, items: Sequence[ManifestItem], size: int = 352) -> None:
        self.root = Path(root)
        self.items = list(items)
        self.size = int(size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        it = self.items[index]
        img = Image.open(self.root / it.image).convert("RGB")
        gt = np.asarray(Image.open(self.root / it.mask).convert("L"))
        gt = binarize_mask(gt)
        x = _to_tensor_rgb(img.resize((self.size, self.size), Image.BILINEAR))
        return {"image": x, "gt": torch.from_numpy(gt), "stem": it.stem, "name": Path(it.mask).name}


def collate_eval(batch: list[dict]) -> dict:
    """Keeps ground truths as a list because their native sizes differ."""
    return {
        "image": torch.stack([b["image"] for b in batch], 0),
        "gt": [b["gt"] for b in batch],
        "stem": [b["stem"] for b in batch],
        "name": [b["name"] for b in batch],
    }


def worker_init_fn(worker_id: int) -> None:
    """Give each worker a distinct, run-reproducible RNG stream."""
    info = torch.utils.data.get_worker_info()
    base = int(torch.initial_seed()) % (2 ** 31)
    seed = (base + worker_id) % (2 ** 31)
    random.seed(seed)
    np.random.seed(seed)
