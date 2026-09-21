"""Perceptual hashing and a cross-split duplicate audit.

Blocking item 2 of §6.2: "a published perceptual-hash collision matrix across
the five test sets".  §1.2.1 of the brief states the concern precisely -- the
60-image CVC-300 is documented as a partition of a 300-image CVC-ColonDB
component, while the released CVC-ColonDB test directory holds 380 images, so
the two "independent external" sets may overlap.  If they do, the pooled
external average double-counts frames and CVC-300 is not an independent
result.

This module additionally audits what the brief does not ask for and what
matters just as much: **train/test leakage**.  The 900/550 split is a PraNet
artefact, not a dataset-owner artefact, and neither Kvasir-SEG nor
CVC-ClinicDB publishes patient-level groupings, so consecutive frames of one
sequence can sit on both sides of the split.  A near-duplicate of a test frame
inside the training set inflates the in-domain numbers and nothing in the
protocol would reveal it.

Three hashes are computed, because they fail differently:
  aHash  -- mean threshold; crude, robust to blur, many false positives.
  dHash  -- horizontal gradient; good at near-duplicate frames.
  pHash  -- DCT low-frequency; the standard choice, robust to scale and
            compression, which is what a redistribution would change.
A pair is flagged when **at least two of the three** distances fall within
their thresholds.  Requiring all three would miss rescaled re-encodes (pHash
degrades under resampling noise); requiring only one floods the report,
because colonoscopy frames share a dark circular vignette and collide easily
under aHash alone.

Thresholds are a judgement call, so the report does not rest on them: it also
publishes, for every ordered pair of splits, the distribution of each image's
**nearest-neighbour distance** into the other split, plus the closest pairs
regardless of threshold.  An overlapping pair of splits shows a spike of
near-zero nearest-neighbour distances that no threshold choice can hide, and a
genuinely disjoint pair shows a clean unimodal bulk around 32 (the expectation
for independent 64-bit hashes).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
from scipy.fft import dct

__all__ = ["HASH_BITS", "ahash", "dhash", "phash", "hash_image", "pack_bits",
           "hamming_matrix", "CollisionReport", "audit"]

HASH_BITS = 64
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _gray(img: Image.Image, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(img.convert("L").resize(size, Image.LANCZOS), dtype=np.float64)


def ahash(img: Image.Image) -> np.ndarray:
    g = _gray(img, (8, 8))
    return (g > g.mean()).flatten()


def dhash(img: Image.Image) -> np.ndarray:
    g = _gray(img, (9, 8))
    return (g[:, 1:] > g[:, :-1]).flatten()


def phash(img: Image.Image) -> np.ndarray:
    g = _gray(img, (32, 32))
    d = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    flat = d.flatten()
    med = np.median(flat[1:])  # drop DC before taking the median
    bits = flat > med
    bits[0] = False           # DC carries brightness only
    return bits


def hash_image(path: Path) -> dict[str, np.ndarray]:
    with Image.open(path) as im:
        im.load()
        return {"ahash": ahash(im), "dhash": dhash(im), "phash": phash(im)}


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """``[N, 64]`` booleans -> ``[N, 8]`` uint8, for fast XOR popcount."""
    return np.packbits(np.asarray(bits, dtype=bool), axis=-1)


def hamming_matrix(a_packed: np.ndarray, b_packed: np.ndarray, block: int = 512) -> np.ndarray:
    """Pairwise Hamming distances between two packed hash sets.

    Uses ``np.bitwise_count`` when available (numpy >= 2.0) and an 8-bit
    lookup table otherwise, so this runs on the numpy that ships with the
    torch 1.12/CUDA 11.3 builds a CUDA 11.4 driver wants.
    """
    n, m = a_packed.shape[0], b_packed.shape[0]
    out = np.empty((n, m), dtype=np.uint8)
    counter = getattr(np, "bitwise_count", None)
    for i in range(0, n, block):
        chunk = a_packed[i: i + block]
        x = np.bitwise_xor(chunk[:, None, :], b_packed[None, :, :])
        pc = counter(x) if counter is not None else _POPCOUNT[x]
        out[i: i + block] = pc.sum(axis=-1).astype(np.uint8)
    return out


@dataclass
class CollisionReport:
    splits: list[str]
    counts: dict[str, int]
    matrix: dict[str, dict[str, int]]
    """``matrix[A][B]`` = number of images in A with at least one flagged
    near-duplicate in B (asymmetric on purpose: it answers "how much of A is
    compromised by B", which is the question you act on)."""
    pairs: list[dict]
    exact_pixel_duplicates: list[dict]
    nn_stats: dict[str, dict[str, dict]]
    """``nn_stats[A][B]`` summarises, over images of A, the distance to the
    closest image of B.  Threshold-free evidence of overlap."""
    closest: dict[str, list[dict]]
    """``closest["A|B"]`` = the closest pairs between A and B regardless of
    threshold, for eyeballing."""

    def to_json(self) -> dict:
        return {
            "splits": self.splits, "counts": self.counts, "matrix": self.matrix,
            "n_flagged_pairs": len(self.pairs), "pairs": self.pairs,
            "exact_pixel_duplicates": self.exact_pixel_duplicates,
            "exact_duplicates_by_location": self.exact_duplicates_by_location(),
            "nearest_neighbour_stats": self.nn_stats, "closest_pairs": self.closest,
            "findings": self.findings(),
        }

    def summary(self) -> str:
        w = max(len(s) for s in self.splits) + 2
        head = "".join(f"{Path(s).name[:14]:>16}" for s in self.splits)
        label = "from / to"
        lines = [f"{label:<{w}}{head}"]
        for a in self.splits:
            row = "".join(f"{self.matrix[a][b]:>16}" for b in self.splits)
            lines.append(f"{a:<{w}}{row}")
        lines.append("")
        lines.append("Median nearest-neighbour pHash distance (row -> column); "
                     "~32 means independent, near 0 means overlap")
        lines.append(f"{label:<{w}}{head}")
        for a in self.splits:
            row = "".join(
                f"{self.nn_stats[a][b]['phash_median']:>16.1f}" if a != b else f"{'-':>16}"
                for b in self.splits
            )
            lines.append(f"{a:<{w}}{row}")
        lines.append("")
        lines.append(f"total flagged near-duplicate pairs: {len(self.pairs)}")
        lines.append(f"exact decoded-pixel duplicates:     {len(self.exact_pixel_duplicates)}")
        found = self.findings()
        if found:
            lines.append("")
            lines.append("Findings (threshold-free evidence only):")
            lines.extend(f"  * {f}" for f in found)
        return "\n".join(lines)

    def exact_duplicates_by_location(self) -> dict:
        """``{"within A": n, "A <-> B": n}`` for the decoded-pixel collisions.

        A count on its own is not actionable: 76 duplicates inside one split
        and 76 spanning train and test are different findings with different
        consequences, and the number is identical.
        """
        loc: dict = {}
        for group in self.exact_pixel_duplicates:
            splits = sorted({m["split"] for m in group["members"]})
            key = f"within {splits[0]}" if len(splits) == 1 else " <-> ".join(splits)
            loc[key] = loc.get(key, 0) + 1
        return dict(sorted(loc.items(), key=lambda kv: -kv[1]))

    def findings(self) -> list[str]:
        """Statements to act on, derived only from evidence without a threshold.

        The flag matrix depends on three cut-offs chosen by hand, and
        colonoscopy frames collide more than natural images do, so it screens
        rather than concludes.  A median nearest-neighbour distance and a
        decoded-pixel SHA-256 collision do not depend on any cut-off: the first
        is a property of the two sets, the second is not a similarity judgement
        at all.
        """
        out: list[str] = []
        for key, n in self.exact_duplicates_by_location().items():
            if key.startswith("within "):
                split = key[len("within "):]
                out.append(
                    f"{n} group(s) of byte-identical decoded images {key}. The split is "
                    f"smaller than its count: {split} has fewer distinct images than pairs.")
            else:
                out.append(
                    f"{n} group(s) of byte-identical decoded images span {key}. These are the "
                    f"same picture, not a similarity call -- no threshold argument applies.")

        for a in self.splits:
            for b in self.splits:
                if a == b:
                    continue
                st = self.nn_stats.get(a, {}).get(b, {})
                med = st.get("phash_median")
                if med is None or med != med:
                    continue
                n_a = self.counts.get(a, 0)
                near = st.get("phash", {}).get("n_le_2")
                if med <= 2.0:
                    out.append(
                        f"{a} -> {b}: median nearest-neighbour pHash distance {med:.1f}. At "
                        f"least half of {a} has a near-exact twin in {b}"
                        + (f" ({near} of {n_a} within 2 bits)" if near is not None else "")
                        + f". Treat {a} as a subset of {b}, not as an independent set: a "
                          f"pooled average over both counts those images twice.")
                elif med <= 10.0:
                    out.append(
                        f"{a} -> {b}: median nearest-neighbour pHash distance {med:.1f}, "
                        f"against ~32 for independent sets"
                        + (f" ({near} of {n_a} within 2 bits)" if near is not None else "")
                        + f". Substantial overlap; say so wherever a {a} number is reported.")
        return out


def _nn_summary(d: np.ndarray, axis: int) -> dict:
    """Percentiles of the per-image minimum distance along ``axis``."""
    if d.size == 0:
        return {}
    mins = d.min(axis=axis).astype(float)
    return {
        "min": float(mins.min()), "p01": float(np.percentile(mins, 1)),
        "p05": float(np.percentile(mins, 5)), "median": float(np.median(mins)),
        "n_le_2": int((mins <= 2).sum()), "n_le_6": int((mins <= 6).sum()),
        "n_le_10": int((mins <= 10).sum()),
    }


def audit(
    root: Path,
    manifest: dict,
    splits: Optional[Sequence[str]] = None,
    ahash_max: int = 8,
    dhash_max: int = 8,
    phash_max: int = 8,
    n_closest: int = 50,
    progress: bool = True,
) -> CollisionReport:
    """Hash every image in every split and audit for near-duplicates.

    Cost is O(sum_i n_i^2) 64-bit XORs; at ~2250 images that is ~5M
    comparisons per hash, i.e. seconds.  Hashing dominates, at one decode per
    image.
    """
    splits = list(splits or manifest["splits"].keys())
    entries: dict[str, list[dict]] = {}
    packed: dict[str, dict[str, np.ndarray]] = {}
    pixel_index: dict[str, list[tuple[str, str]]] = {}

    for split in splits:
        items = manifest["splits"][split]["items"]
        rows: list[dict] = []
        bits: dict[str, list[np.ndarray]] = {"ahash": [], "dhash": [], "phash": []}
        for k, it in enumerate(items):
            if progress and k % 200 == 0:
                print(f"  hashing {split}: {k}/{len(items)}", flush=True)
            h = hash_image(root / it["image"])
            for key in bits:
                bits[key].append(h[key])
            rows.append({"stem": it["stem"], "image": it["image"]})
            pixel_index.setdefault(it["image_pixel_sha256"], []).append((split, it["stem"]))
        entries[split] = rows
        packed[split] = {k: pack_bits(np.array(v)) for k, v in bits.items()}

    matrix = {a: {b: 0 for b in splits} for a in splits}
    nn_stats: dict[str, dict[str, dict]] = {a: {} for a in splits}
    closest: dict[str, list[dict]] = {}
    pairs: list[dict] = []

    for i, a in enumerate(splits):
        for b in splits[i:]:
            same = a == b
            da = hamming_matrix(packed[a]["ahash"], packed[b]["ahash"]).astype(np.int16)
            dd = hamming_matrix(packed[a]["dhash"], packed[b]["dhash"]).astype(np.int16)
            dp = hamming_matrix(packed[a]["phash"], packed[b]["phash"]).astype(np.int16)
            if same:
                np.fill_diagonal(da, 64)
                np.fill_diagonal(dd, 64)
                np.fill_diagonal(dp, 64)

            votes = ((da <= ahash_max).astype(np.int8) + (dd <= dhash_max).astype(np.int8)
                     + (dp <= phash_max).astype(np.int8))
            flag = votes >= 2
            ii, jj = np.nonzero(flag)
            for x, y in zip(ii.tolist(), jj.tolist()):
                if same and y <= x:
                    continue
                pairs.append({
                    "split_a": a, "image_a": entries[a][x]["image"],
                    "split_b": b, "image_b": entries[b][y]["image"],
                    "ahash": int(da[x, y]), "dhash": int(dd[x, y]), "phash": int(dp[x, y]),
                })
            matrix[a][b] += int(flag.any(axis=1).sum())
            if not same:
                matrix[b][a] += int(flag.any(axis=0).sum())

            nn_stats[a][b] = {
                "phash_median": float(np.median(dp.min(axis=1))),
                "phash": _nn_summary(dp, axis=1), "dhash": _nn_summary(dd, axis=1),
                "ahash": _nn_summary(da, axis=1),
            }
            if not same:
                nn_stats[b][a] = {
                    "phash_median": float(np.median(dp.min(axis=0))),
                    "phash": _nn_summary(dp, axis=0), "dhash": _nn_summary(dd, axis=0),
                    "ahash": _nn_summary(da, axis=0),
                }
                combined = da.astype(np.int32) + dd + dp
                k = min(n_closest, combined.size)
                flat = np.argpartition(combined.ravel(), k - 1)[:k]
                flat = flat[np.argsort(combined.ravel()[flat])]
                closest[f"{a}|{b}"] = [
                    {"image_a": entries[a][int(f // combined.shape[1])]["image"],
                     "image_b": entries[b][int(f % combined.shape[1])]["image"],
                     "ahash": int(da.ravel()[f]), "dhash": int(dd.ravel()[f]),
                     "phash": int(dp.ravel()[f])}
                    for f in flat.tolist()
                ]

    for a in splits:
        nn_stats[a].setdefault(a, {"phash_median": float("nan")})

    exact = [
        {"pixel_sha256": k, "members": [{"split": s, "stem": st} for s, st in v]}
        for k, v in pixel_index.items() if len(v) > 1
    ]
    return CollisionReport(
        splits=splits, counts={s: len(entries[s]) for s in splits},
        matrix=matrix, pairs=pairs, exact_pixel_duplicates=exact,
        nn_stats=nn_stats, closest=closest,
    )
