"""Frozen dataset manifests -- blocking item 1 of the locked protocol.

The brief's §1.1 shows that the two primary sources for the "canonical"
polyp protocol disagree with each other: the PraNet paper says 80/10/10, the
PraNet repository ships 900+550 train and 100/62/380/60/196 test.  The
repository manifest is what the field actually downloads, so the only way to
make a run auditable is to freeze *the files you actually used*, by content
hash, and publish that.

A manifest records, per image/mask pair:

  * the relative paths and the pairing stem (pairs are matched by stem, never
    by ``sorted(listdir)`` position -- the reference code sorts the two
    directories independently, which silently mis-pairs whenever the two
    directories disagree on extension or case);
  * the SHA-256 of the **file bytes** (detects any change at all);
  * the SHA-256 of the **decoded pixels** (detects re-encodes that leave the
    image identical -- the same picture saved as a different JPEG quality has
    a different file hash but the same pixel hash);
  * the image size and mode, so a loader can fail loudly instead of silently
    resizing something unexpected;
  * the positive fraction of the binarised mask, which is the cheapest
    possible smoke test that masks were not inverted or mis-paired.

``verify`` recomputes everything and reports differences by category.  It is
wired into CI-style use via ``tools/verify_manifest.py``, which exits non-zero
on any mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "SCHEMA",
    "IMAGE_EXTS",
    "MASK_EXTS",
    "ManifestItem",
    "build_split",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "verify_manifest",
    "binarize_mask",
    "sha256_file",
]

SCHEMA = "polyptail/manifest/1"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
MASK_EXTS = IMAGE_EXTS
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def binarize_mask(arr: np.ndarray) -> np.ndarray:
    """Binarise a grayscale mask array robustly across the five datasets.

    Kvasir-SEG masks are 1-bit (PIL gives 0/255 after ``convert('L')``); the
    CVC sets are 0/255 8-bit; a few third-party redistributions ship 0/1.
    Thresholding at ``>127`` is correct for the first two and wrong for the
    third, so the 0/1 case is detected by its maximum and handled separately.
    """
    a = np.asarray(arr)
    if a.max() > 1:
        return (a > 127).astype(np.uint8)
    return (a > 0).astype(np.uint8)


@dataclass
class ManifestItem:
    stem: str
    image: str
    mask: str
    image_sha256: str
    mask_sha256: str
    image_pixel_sha256: str
    mask_pixel_sha256: str
    image_size: tuple[int, int]
    mask_size: tuple[int, int]
    image_mode: str
    mask_mode: str
    mask_positive_frac: float

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        d["image_size"] = list(self.image_size)
        d["mask_size"] = list(self.mask_size)
        return d

    @staticmethod
    def from_json(d: dict) -> "ManifestItem":
        d = dict(d)
        d["image_size"] = tuple(d["image_size"])
        d["mask_size"] = tuple(d["mask_size"])
        return ManifestItem(**d)


def _index_dir(d: Path, exts: Sequence[str]) -> dict[str, Path]:
    if not d.is_dir():
        raise FileNotFoundError(f"missing directory: {d}")
    out: dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in exts or not p.is_file():
            continue
        stem = p.stem
        if stem in out:
            raise ValueError(f"duplicate stem {stem!r} in {d} ({out[stem].name} and {p.name})")
        out[stem] = p
    return out


def _describe(path: Path, as_mask: bool) -> tuple[dict, np.ndarray]:
    with Image.open(path) as im:
        mode = im.mode
        size = im.size  # (w, h)
        conv = im.convert("L" if as_mask else "RGB")
        arr = np.asarray(conv)
    info = {
        "sha256": sha256_file(path),
        "pixel_sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest(),
        "size": (int(size[0]), int(size[1])),
        "mode": mode,
    }
    return info, arr


def build_split(root: Path, split_rel: str, strict_sizes: bool = True) -> list[ManifestItem]:
    """Hash one ``<split>/images`` + ``<split>/masks`` directory pair."""
    base = root / split_rel
    imgs = _index_dir(base / "images", IMAGE_EXTS)
    msks = _index_dir(base / "masks", MASK_EXTS)
    only_img = sorted(set(imgs) - set(msks))
    only_msk = sorted(set(msks) - set(imgs))
    if only_img or only_msk:
        raise ValueError(
            f"{split_rel}: unpaired stems -- images only: {only_img[:5]}"
            f" ({len(only_img)}), masks only: {only_msk[:5]} ({len(only_msk)})"
        )
    items: list[ManifestItem] = []
    for stem in sorted(imgs):
        ip, mp = imgs[stem], msks[stem]
        i_info, _ = _describe(ip, as_mask=False)
        m_info, m_arr = _describe(mp, as_mask=True)
        if strict_sizes and i_info["size"] != m_info["size"]:
            raise ValueError(
                f"{split_rel}/{stem}: image size {i_info['size']} != mask size {m_info['size']}"
            )
        binm = binarize_mask(m_arr)
        items.append(
            ManifestItem(
                stem=stem,
                image=str(ip.relative_to(root)).replace(os.sep, "/"),
                mask=str(mp.relative_to(root)).replace(os.sep, "/"),
                image_sha256=i_info["sha256"],
                mask_sha256=m_info["sha256"],
                image_pixel_sha256=i_info["pixel_sha256"],
                mask_pixel_sha256=m_info["pixel_sha256"],
                image_size=i_info["size"],
                mask_size=m_info["size"],
                image_mode=i_info["mode"],
                mask_mode=m_info["mode"],
                mask_positive_frac=float(binm.mean()),
            )
        )
    return items


def build_manifest(root: Path, splits: Iterable[str], strict_sizes: bool = True) -> dict:
    root = Path(root)
    out: dict = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root_name": root.name,
        "splits": {},
    }
    for s in splits:
        items = build_split(root, s, strict_sizes=strict_sizes)
        out["splits"][s] = {"n_pairs": len(items), "items": [i.to_json() for i in items]}
    out["total_pairs"] = sum(v["n_pairs"] for v in out["splits"].values())
    return out


def write_manifest(manifest: dict, json_path: Path, sha_path: Optional[Path] = None) -> None:
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    if sha_path is not None:
        lines: list[str] = []
        for split in sorted(manifest["splits"]):
            for it in manifest["splits"][split]["items"]:
                lines.append(f"{it['image_sha256']}  {it['image']}")
                lines.append(f"{it['mask_sha256']}  {it['mask']}")
        Path(sha_path).write_text("\n".join(lines) + "\n")


def load_manifest(json_path: Path) -> dict:
    with open(json_path) as fh:
        m = json.load(fh)
    if m.get("schema") != SCHEMA:
        raise ValueError(f"unexpected manifest schema {m.get('schema')!r}, expected {SCHEMA!r}")
    return m


@dataclass
class VerifyReport:
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    file_hash_mismatch: list[str] = field(default_factory=list)
    pixel_hash_mismatch: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.file_hash_mismatch or self.pixel_hash_mismatch or self.extra)

    def summary(self) -> str:
        if self.ok:
            return f"OK: {self.checked} files match the manifest."
        parts = [f"FAILED ({self.checked} files checked)"]
        for name, lst in (
            ("missing", self.missing),
            ("file-hash mismatch", self.file_hash_mismatch),
            ("pixel-hash mismatch (re-encoded)", self.pixel_hash_mismatch),
            ("present but not in manifest", self.extra),
        ):
            if lst:
                parts.append(f"  {name}: {len(lst)} e.g. {lst[:5]}")
        return "\n".join(parts)


def verify_manifest(root: Path, manifest: dict, check_pixels: bool = True) -> VerifyReport:
    root = Path(root)
    rep = VerifyReport()
    listed: set[str] = set()
    for split, blob in sorted(manifest["splits"].items()):
        for raw in blob["items"]:
            it = ManifestItem.from_json(raw)
            for rel, fsha, psha, as_mask in (
                (it.image, it.image_sha256, it.image_pixel_sha256, False),
                (it.mask, it.mask_sha256, it.mask_pixel_sha256, True),
            ):
                listed.add(rel)
                p = root / rel
                rep.checked += 1
                if not p.is_file():
                    rep.missing.append(rel)
                    continue
                if sha256_file(p) != fsha:
                    rep.file_hash_mismatch.append(rel)
                    if not check_pixels:
                        continue
                if check_pixels:
                    info, _ = _describe(p, as_mask=as_mask)
                    if info["pixel_sha256"] != psha:
                        rep.pixel_hash_mismatch.append(rel)
    for split in manifest["splits"]:
        for sub in ("images", "masks"):
            d = root / split / sub
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMAGE_EXTS or not p.is_file():
                    continue
                rel = str(p.relative_to(root)).replace(os.sep, "/")
                if rel not in listed:
                    rep.extra.append(rel)
    return rep
