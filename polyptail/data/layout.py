"""Validate a ./dataset tree against the PraNet / Polyp-PVT distribution.

This runs *before* ``tools/freeze_manifest.py``. Freezing answers "what
exactly do I have"; this answers "is what I have the thing everyone else
means", and it says so in terms you can act on rather than by raising on the
first surprise.

Expectations are read off the reference implementation, not assumed:

* ``Train.py`` and ``Test.py`` hard-code ``./dataset/TrainDataset/`` and
  ``./dataset/TestDataset/<name>/`` with ``images/`` and ``masks/``
  subdirectories, and the five test names
  ``CVC-300, CVC-ClinicDB, Kvasir, CVC-ColonDB, ETIS-LaribPolypDB``.
* ``utils/dataloader.py`` filters **training** masks to ``.png`` only, and
  **test** masks to ``.tif`` or ``.png``; images to ``.jpg`` or ``.png`` in
  both. A file outside those sets is silently dropped by the reference code --
  it does not error, it just trains on fewer images than you think. This
  module reports such files, because a copy of the data that this project
  reads and the reference does not is a copy that cannot be compared.

One further quirk is checked because it wastes an afternoon otherwise:
``Train.py`` line 116 calls ``test(model, test_path, 'test')``, i.e. it
requires ``./dataset/TestDataset/test/`` -- a directory the distributed
archive does not contain. The released training script therefore cannot
finish its first epoch on the released data unless you create that directory,
and whatever you put in it silently becomes the checkpoint-selection
criterion. Nothing here needs it; it is reported only so its absence is not
mistaken for a broken download.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

__all__ = [
    "EXPECTED_COUNTS", "TEST_SPLITS", "REFERENCE_IMAGE_EXTS",
    "REFERENCE_TRAIN_MASK_EXTS", "REFERENCE_TEST_MASK_EXTS",
    "LayoutReport", "check_layout",
]

#: Counts in the PraNet-distributed archive, as documented in the brief's §1.2.
EXPECTED_COUNTS = {
    "TrainDataset": 1450,
    "TestDataset/Kvasir": 100,
    "TestDataset/CVC-ClinicDB": 62,
    "TestDataset/CVC-ColonDB": 380,
    "TestDataset/CVC-300": 60,
    "TestDataset/ETIS-LaribPolypDB": 196,
}
TEST_SPLITS = [s for s in EXPECTED_COUNTS if s.startswith("TestDataset/")]

REFERENCE_IMAGE_EXTS = {".jpg", ".png"}
REFERENCE_TRAIN_MASK_EXTS = {".png"}
REFERENCE_TEST_MASK_EXTS = {".tif", ".png"}

#: Directory names people actually end up with, and what they should be.
_ALIASES = {
    "cvc-t": "TestDataset/CVC-300",
    "cvct": "TestDataset/CVC-300",
    "endoscene": "TestDataset/CVC-300",
    "cvc300": "TestDataset/CVC-300",
    "cvc_300": "TestDataset/CVC-300",
    "cvc-clinicdb": "TestDataset/CVC-ClinicDB",
    "cvc612": "TestDataset/CVC-ClinicDB",
    "cvc-612": "TestDataset/CVC-ClinicDB",
    "cvc-colondb": "TestDataset/CVC-ColonDB",
    "etis": "TestDataset/ETIS-LaribPolypDB",
    "etis-larib": "TestDataset/ETIS-LaribPolypDB",
    "etislaribpolypdb": "TestDataset/ETIS-LaribPolypDB",
    "kvasir-seg": "TestDataset/Kvasir",
    "kvasir_seg": "TestDataset/Kvasir",
}


@dataclass
class LayoutReport:
    root: Path
    errors: list[str] = field(default_factory=list)
    """Things that will stop the pipeline working."""
    warnings: list[str] = field(default_factory=list)
    """Things that are suspicious but may be legitimate."""
    notes: list[str] = field(default_factory=list)
    """Informational; includes the count table."""
    counts: dict[str, int] = field(default_factory=dict)
    count_mismatches: list[str] = field(default_factory=list)
    """Splits whose pair count differs from the distributed archive. Collected
    rather than warned per-split: six near-identical warnings bury the one
    error that actually needs acting on."""

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines: list[str] = []
        if self.counts:
            lines.append(f"{'split':<34} {'pairs':>6}  {'expected':>8}  status")
            for split, expected in EXPECTED_COUNTS.items():
                n = self.counts.get(split)
                if n is None:
                    lines.append(f"{split:<34} {'-':>6}  {expected:>8}  MISSING")
                elif n == expected:
                    lines.append(f"{split:<34} {n:>6}  {expected:>8}  ok")
                else:
                    lines.append(f"{split:<34} {n:>6}  {expected:>8}  differs ({n - expected:+d})")
            lines.append("")
        for label, items in (("ERROR", self.errors), ("WARNING", self.warnings), ("note", self.notes)):
            for item in items:
                lines.append(f"{label}: {item}")
        if self.ok and not self.warnings:
            lines.append("")
            lines.append("Layout matches the PraNet/Polyp-PVT distribution. "
                         "Next: python tools/freeze_manifest.py")
        elif self.ok:
            lines.append("")
            lines.append("Usable, with the warnings above. Read them before reporting any number.")
        return "\n".join(lines)


def _files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))


def _check_split(root: Path, split: str, rep: LayoutReport) -> None:
    base = root / split
    is_train = split == "TrainDataset"
    if not base.is_dir():
        rep.errors.append(f"missing directory {split}/")
        _suggest_alias(root, split, rep)
        return

    img_dir, msk_dir = base / "images", base / "masks"
    for name, d in (("images", img_dir), ("masks", msk_dir)):
        if not d.is_dir():
            nested = base / split.split("/")[-1]
            if nested.is_dir():
                rep.errors.append(
                    f"{split}/ has no {name}/ but does contain {nested.name}/ -- the archive was "
                    f"unzipped one level too deep. Move {nested}/* up into {base}/ ."
                )
            else:
                present = sorted(p.name for p in base.iterdir())[:6] if base.is_dir() else []
                rep.errors.append(f"{split}/ has no {name}/ subdirectory (found: {present})")
            return

    images, masks = _files(img_dir), _files(msk_dir)
    if not images or not masks:
        rep.errors.append(f"{split}: images/ or masks/ is empty")
        return

    img_by_stem = {p.stem: p for p in images}
    msk_by_stem = {p.stem: p for p in masks}
    paired = sorted(set(img_by_stem) & set(msk_by_stem))
    rep.counts[split] = len(paired)

    only_img = sorted(set(img_by_stem) - set(msk_by_stem))
    only_msk = sorted(set(msk_by_stem) - set(img_by_stem))
    if only_img:
        rep.errors.append(
            f"{split}: {len(only_img)} image(s) have no mask with a matching name, "
            f"e.g. {only_img[:4]}"
        )
    if only_msk:
        rep.errors.append(
            f"{split}: {len(only_msk)} mask(s) have no image with a matching name, "
            f"e.g. {only_msk[:4]}"
        )

    # Extensions the *reference* implementation would accept. Files outside
    # these sets are silently skipped by it, not rejected.
    bad_img = sorted(p.name for p in images if p.suffix.lower() not in REFERENCE_IMAGE_EXTS)
    mask_exts = REFERENCE_TRAIN_MASK_EXTS if is_train else REFERENCE_TEST_MASK_EXTS
    bad_msk = sorted(p.name for p in masks if p.suffix.lower() not in mask_exts)
    if bad_img:
        rep.warnings.append(
            f"{split}: {len(bad_img)} image(s) have an extension the reference dataloader "
            f"filters out ({sorted(REFERENCE_IMAGE_EXTS)} only), e.g. {bad_img[:4]}. "
            "This project reads them; the reference would silently skip them, so the two "
            "are not comparable."
        )
    if bad_msk:
        rep.warnings.append(
            f"{split}: {len(bad_msk)} mask(s) have an extension the reference dataloader "
            f"filters out ({sorted(mask_exts)} only), e.g. {bad_msk[:4]}. Same caveat."
        )

    # Sizes, and a cheap sanity check that masks are masks.
    mismatched, degenerate = [], []
    for stem in paired:
        try:
            with Image.open(img_by_stem[stem]) as im:
                isize = im.size
            with Image.open(msk_by_stem[stem]) as mm:
                msize = mm.size
        except Exception as exc:  # unreadable file
            rep.errors.append(f"{split}/{stem}: cannot be opened ({exc})")
            continue
        if isize != msize:
            mismatched.append(stem)
    if mismatched:
        rep.errors.append(
            f"{split}: {len(mismatched)} pair(s) where image and mask differ in size, "
            f"e.g. {mismatched[:4]}. The archive did not unpack cleanly; re-download."
        )

    if split in EXPECTED_COUNTS and len(paired) != EXPECTED_COUNTS[split]:
        rep.count_mismatches.append(split)


def _suggest_alias(root: Path, split: str, rep: LayoutReport) -> None:
    """If a differently-named directory looks like the missing one, say so."""
    parent = root / Path(split).parent if "/" in split else root
    if not parent.is_dir():
        return
    for candidate in sorted(p for p in parent.iterdir() if p.is_dir()):
        key = candidate.name.lower().replace(" ", "")
        target = _ALIASES.get(key)
        if target == split:
            rep.errors.append(
                f"  -> {candidate.relative_to(root)}/ looks like it: rename it to "
                f"{Path(split).name}/ (the reference code hard-codes that spelling)"
            )


def check_layout(root: Path, splits: Optional[list[str]] = None) -> LayoutReport:
    """Validate ``root`` against the distributed layout. Never writes anything."""
    root = Path(root)
    rep = LayoutReport(root=root)
    if not root.is_dir():
        rep.errors.append(f"{root}/ does not exist. Unpack the archive so that "
                          f"{root}/TrainDataset/images/ exists.")
        return rep

    for split in (splits or list(EXPECTED_COUNTS)):
        _check_split(root, split, rep)

    # The reference Train.py needs this and the archive does not ship it.
    if not (root / "TestDataset" / "test").is_dir():
        rep.notes.append(
            "TestDataset/test/ is absent. That is correct for the distributed archive and "
            "nothing here needs it -- but the reference Train.py calls "
            "test(model, test_path, 'test') and will crash at the end of epoch 1 without it. "
            "If you also want to run the original code, point it at a copy of one test split, "
            "and note that whatever you choose becomes its checkpoint-selection criterion."
        )

    if rep.count_mismatches:
        detail = ", ".join(
            f"{s} ({rep.counts.get(s, 0)} vs {EXPECTED_COUNTS[s]})"
            for s in rep.count_mismatches
        )
        rep.warnings.append(
            f"{len(rep.count_mismatches)} split(s) differ from the counts the PraNet archive "
            f"distributes: {detail}. Not automatically wrong -- §1.2 of the brief documents an "
            "unresolved 300-vs-380 discrepancy for CVC-ColonDB -- but you must be able to "
            "explain each one before quoting a number from that split."
        )

    stray = []
    td = root / "TestDataset"
    if td.is_dir():
        known = {Path(s).name for s in TEST_SPLITS} | {"test"}
        stray = sorted(p.name for p in td.iterdir() if p.is_dir() and p.name not in known)
    if stray:
        rep.warnings.append(
            f"TestDataset/ contains unrecognised directories: {stray}. "
            "If one of these is a renamed split, rename it back -- the five names are "
            "hard-coded in the reference code and in configs/base.yaml."
        )
    return rep
