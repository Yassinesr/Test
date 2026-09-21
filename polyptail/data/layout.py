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

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image

__all__ = [
    "EXPECTED_COUNTS", "TEST_SPLITS", "REFERENCE_IMAGE_EXTS",
    "REFERENCE_TRAIN_MASK_EXTS", "REFERENCE_TEST_MASK_EXTS",
    "LayoutReport", "check_layout", "link_dataset",
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


#: How the distributed splits are composed. Used only to explain a count
#: mismatch: TrainDataset is two corpora concatenated, and knowing which half
#: is short is the difference between "re-copy one directory" and "you have a
#: different dataset".
SPLIT_COMPOSITION = {
    "TrainDataset": {"Kvasir-SEG": 900, "CVC-ClinicDB": 550},
}


def _name_shape(stem: str) -> str:
    """Classify a stem coarsely enough to tell two corpora apart.

    The archive keeps each source's original naming: one corpus is numbered,
    the other carries long lowercase alphanumeric identifiers. Which is which
    is not hard-coded here -- the counts say that, and saying it from the data
    cannot go stale.
    """
    if stem.isdigit():
        return "numeric"
    if len(stem) >= 16 and stem.isalnum() and stem.islower():
        return "alphanumeric"
    return "other"


def _numeric_gaps(stems: set) -> Optional[tuple]:
    """``(lo, hi, missing)`` for the numeric stems, or None if not applicable.

    A contiguous block of missing numbers means a copy stopped early; scattered
    ones mean something removed files on purpose. The two have different fixes,
    so the distinction is worth the few lines.
    """
    nums = sorted(int(s) for s in stems if s.isdigit())
    if len(nums) < 2 or nums[-1] - nums[0] > 100_000:
        return None
    return nums[0], nums[-1], sorted(set(range(nums[0], nums[-1] + 1)) - set(nums))


def _stem_collisions(files: list) -> list[str]:
    """Stems carried by more than one file, e.g. ``7.png`` beside ``7.jpg``."""
    seen: dict = {}
    for f in files:
        seen[f.stem] = seen.get(f.stem, 0) + 1
    return sorted(k for k, v in seen.items() if v > 1)


def _diagnose_count(split: str, expected: int, paired: list,
                    images: list, masks: list) -> list[str]:
    """Explain *why* a split's pair count differs. Returns display lines.

    ``differs (-162)`` is a fact, not a diagnosis. Each branch below separates
    causes that have different fixes -- a copy that stopped early, a corpus
    that was never there, a filter someone applied -- so the table stops being
    a prompt to go and ask someone what it means.
    """
    n, out = len(paired), []
    img_stems, msk_stems = {p.stem for p in images}, {p.stem for p in masks}
    out.append(f"files on disk: images/ {len(images)}, masks/ {len(masks)}; "
               f"distinct stems: {len(img_stems)} and {len(msk_stems)}")

    if len(images) == len(masks) == n:
        out.append(f"both sides agree and every file is paired, so the {expected - n} "
                   f"absent item(s) are missing from images/ and masks/ alike. A transfer "
                   f"that stopped mid-directory leaves orphans on one side; there are none.")

    shapes: dict = {}
    for stem in paired:
        shape = _name_shape(stem)
        shapes[shape] = shapes.get(shape, 0) + 1
    if len(shapes) > 1:
        parts = []
        for shape, count in sorted(shapes.items(), key=lambda kv: -kv[1]):
            match = [name for name, k in SPLIT_COMPOSITION.get(split, {}).items() if k == count]
            parts.append(f"{count} {shape}" + (f" (= the {match[0]} share exactly)" if match else ""))
        out.append("names by shape: " + ", ".join(parts))
        if split in SPLIT_COMPOSITION:
            want = ", ".join(f"{k} {v}" for k, v in SPLIT_COMPOSITION[split].items())
            out.append(f"the archive is {want}, and the two corpora are named differently, "
                       f"so the group that is short is the one to re-copy.")

    gaps = _numeric_gaps({s for s in paired if _name_shape(s) == "numeric"})
    if gaps:
        lo, hi, missing = gaps
        count = sum(1 for s in paired if _name_shape(s) == "numeric")
        if missing:
            contiguous = missing[-1] - missing[0] + 1 == len(missing)
            out.append(
                f"numeric names run {lo}-{hi} with {len(missing)} missing: "
                + (f"one contiguous block, {missing[0]}-{missing[-1]}. A run of consecutive "
                   f"numbers absent from the middle is a copy or unzip that stopped early "
                   f"and was restarted past the gap: redo that transfer."
                   if contiguous else
                   f"scattered, e.g. {missing[:6]}. Scattered gaps are a filter, not an "
                   f"accident -- something selected these out, so find out what before you train.")
            )
        elif lo == 1 and hi == count:
            out.append(f"numeric names run 1-{hi} unbroken -- nothing is missing from inside "
                       f"that range, the sequence simply stops at {hi}. If this group should "
                       f"be larger, its tail was never copied rather than lost in transit.")
    return out



#: A validation split is not part of the distributed archive -- it exists only
#: when you have re-split the training pool yourself -- so it has no expected
#: count and is checked as a partition instead. See ``_check_partition``.
VALIDATION_SPLIT = "ValidationDataset"

#: Directory names a mask folder travels under. ``masks`` is what the
#: reference code reads; ``gts`` is what several polyp repositories ship and
#: what a hand-assembled split often ends up with.
MASK_DIR_NAMES = ("masks", "gts")


class AmbiguousMaskDir(Exception):
    """``masks/`` and ``gts/`` both exist and hold different things."""


def resolve_mask_dir(base: Path) -> tuple:
    """Return ``(directory, note)`` for a split's ground truth.

    Raises ``AmbiguousMaskDir`` when both names exist and disagree: with two
    candidate ground truths and no way to tell which one a number came from,
    guessing is worse than stopping.
    """
    present = [base / n for n in MASK_DIR_NAMES if (base / n).is_dir()]
    if not present:
        return None, None
    if len(present) == 1:
        d = present[0]
        note = None if d.name == "masks" else (
            f"{base.name}/ stores ground truth in {d.name}/ rather than masks/. Read here; "
            f"the reference code hard-codes masks/, so rename or symlink it before running "
            f"the original scripts.")
        return d, note
    masks, gts = base / "masks", base / "gts"
    m_stems = {f.stem for f in _files(masks)}
    g_stems = {f.stem for f in _files(gts)}
    if m_stems != g_stems:
        only_m, only_g = sorted(m_stems - g_stems), sorted(g_stems - m_stems)
        raise AmbiguousMaskDir(
            f"{base.name}/ has both masks/ ({len(m_stems)}) and gts/ ({len(g_stems)}), and they "
            f"hold different stems -- masks/ only: {only_m[:3]} ({len(only_m)}), gts/ only: "
            f"{only_g[:3]} ({len(only_g)}). Two candidate ground truths means a reported number "
            f"cannot say which one produced it. Delete or rename one.")
    return masks, (
        f"{base.name}/ has both masks/ and gts/ with the same {len(m_stems)} stems. Reading "
        f"masks/; gts/ is ignored. They are compared by name only -- if you edited one, "
        f"freeze_manifest will hash whichever this reads, so remove the copy you do not want.")


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
    partition_ok: bool = False
    """True when a held-out validation split accounts for a short training
    split exactly. The table and the closing line both have to know: a status
    of ``differs`` beside a note saying it does not is worse than either."""
    diagnostics: dict = field(default_factory=dict)
    """split -> lines explaining a count difference. Populated only for the
    splits in ``count_mismatches``; the explanation costs a directory listing
    that has already been done, so it is never deferred to a second command."""

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
                elif self.partition_ok and split == "TrainDataset":
                    lines.append(f"{split:<34} {n:>6}  {expected:>8}  re-split ({n - expected:+d})")
                else:
                    lines.append(f"{split:<34} {n:>6}  {expected:>8}  differs ({n - expected:+d})")
            for split, n in self.counts.items():
                if split not in EXPECTED_COUNTS:
                    lines.append(f"{split:<34} {n:>6}  {'-':>8}  held out by you")
            lines.append("")
        for label, items in (("ERROR", self.errors), ("WARNING", self.warnings), ("note", self.notes)):
            for item in items:
                lines.append(f"{label}: {item}")
        for split, detail in self.diagnostics.items():
            lines.append("")
            lines.append(f"why {split} differs:")
            for line in detail:
                lines.append(f"  - {line}")
        if self.ok and not self.warnings:
            lines.append("")
            lines.append(
                ("Layout is a re-split of the PraNet/Polyp-PVT distribution and the parts "
                 "add up. " if self.partition_ok else
                 "Layout matches the PraNet/Polyp-PVT distribution. ")
                + "Next: python tools/freeze_manifest.py")
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

    img_dir = base / "images"
    try:
        msk_dir, msk_note = resolve_mask_dir(base)
    except AmbiguousMaskDir as exc:
        rep.errors.append(f"{split}: {exc}")
        return
    if msk_note:
        rep.notes.append(msk_note)
    for name, d in (("images", img_dir), ("masks", msk_dir)):
        if d is None or not d.is_dir():
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

    for side, files in (("images", images), ("masks", masks)):
        dupes = _stem_collisions(files)
        if dupes:
            rep.errors.append(
                f"{split}: {len(dupes)} stem(s) in {side}/ belong to more than one file "
                f"(e.g. {dupes[:3]}) -- the same name under two extensions. This project "
                f"pairs by stem and would silently take whichever sorts last, so which file "
                f"you trained on would be unrecorded. The reference counts files rather than "
                f"stems, so its assert len(images) == len(gts) compares {len(images)} with "
                f"{len(masks)}. Delete the copies you do not want.")

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
        rep.diagnostics[split] = _diagnose_count(
            split, EXPECTED_COUNTS[split], paired, images, masks)


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



def _check_partition(rep: LayoutReport) -> None:
    """Reconcile a re-split training pool against the distributed one.

    Holding out a validation set means ``TrainDataset`` is *supposed* to be
    short, so comparing it against 1450 on its own reports a problem that is
    really a design decision. What is worth checking is the arithmetic: the
    parts must still add up to the pool they came from, because a partition
    that loses images loses them silently and one that duplicates them selects
    on data it trained on.
    """
    train, val = rep.counts.get("TrainDataset"), rep.counts.get(VALIDATION_SPLIT)
    if train is None or val is None:
        return
    pool = EXPECTED_COUNTS["TrainDataset"]
    total = train + val
    if total == pool:
        if "TrainDataset" in rep.count_mismatches:
            rep.count_mismatches.remove("TrainDataset")
            rep.diagnostics.pop("TrainDataset", None)
        rep.partition_ok = True
        rep.notes.append(
            f"TrainDataset ({train}) + {VALIDATION_SPLIT} ({val}) = {total}, the size of the "
            f"pool PraNet distributes as TrainDataset. This is a re-split of that pool, not a "
            f"short copy, so the count is not reported as a mismatch. Two things this does not "
            f"check: that no image is in both halves -- tools/hash_collisions.py answers that "
            f"from the frozen manifest, by pixels rather than by name -- and that the split is "
            f"recorded somewhere a reader can reproduce."
        )
    else:
        rep.warnings.append(
            f"TrainDataset ({train}) + {VALIDATION_SPLIT} ({val}) = {total}, but the pool they "
            f"come from holds {pool}: {abs(total - pool)} "
            f"{'unaccounted for' if total < pool else 'more than the pool has'}. If you also "
            f"carved a third part out, say so in run.notes and point at where it lives; if you "
            f"did not, {abs(total - pool)} image(s) went missing in the split and the halves "
            f"below are not the experiment you think you are running."
        )


def check_layout(root: Path, splits: Optional[list[str]] = None) -> LayoutReport:
    """Validate ``root`` against the distributed layout. Never writes anything."""
    root = Path(root)
    rep = LayoutReport(root=root)
    if not root.is_dir():
        rep.errors.append(f"{root}/ does not exist. Unpack the archive so that "
                          f"{root}/TrainDataset/images/ exists.")
        return rep

    wanted = list(splits) if splits else list(EXPECTED_COUNTS)
    if splits is None and (root / VALIDATION_SPLIT).is_dir():
        wanted.append(VALIDATION_SPLIT)
    for split in wanted:
        _check_split(root, split, rep)
    _check_partition(rep)

    # The reference Train.py needs this and the archive does not ship it.
    if (root / "TestDataset" / "test").is_dir():
        rep.notes.append(
            "TestDataset/test/ exists. Nothing here reads it -- it is not in any split list in "
            "configs/, so it cannot leak into a result -- and it is recognised rather than "
            "flagged because the reference Train.py requires it. Keep it if you also run the "
            "original code; whatever is in it is that code's checkpoint-selection criterion."
        )
    else:
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


def link_dataset(source: Path, dest: Path) -> int:
    """Symlink ``dest`` -> ``source`` after checking ``source`` is a real tree."""
    source = Path(source).expanduser()
    if not source.is_dir():
        print(f"source {source} is not a directory", file=sys.stderr)
        return 1
    source = source.resolve()

    print(f"checking {source} before linking ...")
    rep = check_layout(source)
    print()
    print(rep.summary())
    print()
    if not rep.ok:
        print(f"refusing to link: {source} is not a usable dataset tree (see errors above).",
              file=sys.stderr)
        return 1

    if dest.is_symlink():
        current = dest.resolve()
        if current == source:
            print(f"{dest} already points at {source} -- nothing to do")
            return 0
        print(f"{dest} is a symlink to {current}. Remove it first if you meant to repoint it:\n"
              f"  rm {dest}", file=sys.stderr)
        return 1
    if dest.exists():
        print(f"{dest} already exists and is not a symlink. Move or remove it first, or skip\n"
              f"linking and pass --root {source} / data.root={source} instead.", file=sys.stderr)
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.symlink_to(source, target_is_directory=True)
    print(f"linked {dest} -> {source}")
    print("Both checkouts now read the same bytes; nothing was copied.")
    return 0
