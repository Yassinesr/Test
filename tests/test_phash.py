"""Duplicate auditing: the tool must find a planted duplicate and not invent one."""

import numpy as np
import pytest
from PIL import Image

from polyptail.data.manifest import build_manifest
from polyptail.data.phash import (
    ahash, audit, dhash, hamming_matrix, hash_image, pack_bits, phash,
)


def textured(rng, h, w, seed_shift=0.0):
    """An image with independent low-frequency structure on every call.

    Deliberately *not* a phase-shifted sinusoid: a smooth periodic field makes
    aHash and dHash of two different images collide at distance 0-6, because
    those hashes only see coarse structure.  That produced a fixture whose
    'distinct' images were genuine near-duplicates -- the audit was right and
    the generator was wrong.
    """
    del seed_shift  # every call is already independent
    low = rng.normal(0, 1, (6, 8))
    field = np.asarray(
        Image.fromarray(low.astype(np.float32), mode="F").resize((w, h), Image.BICUBIC)
    )
    img = np.clip(128 + 55 * field[..., None].repeat(3, 2) + rng.normal(0, 6, (h, w, 3)), 0, 255)
    return Image.fromarray(img.astype(np.uint8))


class TestHashes:
    @pytest.mark.parametrize("fn", [ahash, dhash, phash])
    def test_hash_is_64_bits(self, fn):
        rng = np.random.default_rng(0)
        assert fn(textured(rng, 40, 60)).size == 64

    @pytest.mark.parametrize("fn", [ahash, dhash, phash])
    def test_identical_images_hash_identically(self, fn):
        rng = np.random.default_rng(1)
        img = textured(rng, 40, 60)
        assert np.array_equal(fn(img), fn(img.copy()))

    @pytest.mark.parametrize("fn", [ahash, dhash, phash])
    def test_rescaled_copy_is_much_closer_than_an_unrelated_image(self, fn):
        rng = np.random.default_rng(2)
        a = textured(rng, 120, 160)
        rescaled = a.resize((80, 60)).resize((160, 120))
        other = textured(rng, 120, 160, seed_shift=3.1)
        pa, pr, po = (pack_bits(np.array([fn(x)])) for x in (a, rescaled, other))
        assert int(hamming_matrix(pa, pr)[0, 0]) < int(hamming_matrix(pa, po)[0, 0])

    def test_unrelated_images_sit_near_the_random_baseline(self):
        rng = np.random.default_rng(3)
        imgs = [textured(rng, 90, 110, seed_shift=k * 2.7) for k in range(12)]
        packed = pack_bits(np.array([phash(i) for i in imgs]))
        d = hamming_matrix(packed, packed).astype(float)
        off = d[~np.eye(len(imgs), dtype=bool)]
        assert off.mean() > 12   # far from 0; 32 is the expectation for random hashes

    def test_popcount_fallback_matches_numpy_bitwise_count(self, monkeypatch):
        rng = np.random.default_rng(4)
        bits = rng.random((20, 64)) > 0.5
        packed = pack_bits(bits)
        fast = hamming_matrix(packed, packed)
        monkeypatch.delattr(np, "bitwise_count", raising=False)
        slow = hamming_matrix(packed, packed)
        assert np.array_equal(fast, slow)

    def test_hash_image_reads_from_disk(self, tmp_path):
        rng = np.random.default_rng(5)
        p = tmp_path / "x.png"
        textured(rng, 40, 40).save(p)
        h = hash_image(p)
        assert set(h) == {"ahash", "dhash", "phash"}


class TestAudit:
    def test_visually_distinct_splits_produce_an_empty_cross_matrix(self, tmp_path):
        rng = np.random.default_rng(11)
        for split, base in (("A", 0.0), ("B", 40.0)):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
            for i in range(6):
                img = textured(rng, 64, 80, seed_shift=base + i * 5.3)
                img.save(tmp_path / split / "images" / f"{i}.png")
                Image.new("L", img.size, 255).save(tmp_path / split / "masks" / f"{i}.png")
        rep = audit(tmp_path, build_manifest(tmp_path, ["A", "B"]), progress=False)
        assert rep.matrix["A"]["B"] == 0
        assert rep.exact_pixel_duplicates == []

    def test_self_similar_images_can_trip_the_thresholds(self, synthetic_dataset):
        """Honest limitation, and the reason the report is not threshold-only.

        Every image in the synthetic fixture is the same generative process --
        a bright ellipse near the centre of a noisy grey field -- exactly as
        colonoscopy frames share a dark circular vignette.  Flag counts are a
        screening signal; the nearest-neighbour distributions and the exact
        pixel-hash duplicates are what settle whether two splits overlap.
        """
        root, manifest = synthetic_dataset
        rep = audit(root, manifest, progress=False)
        assert rep.exact_pixel_duplicates == []
        st = rep.nn_stats["TrainDataset"]["TestDataset/Fake"]["phash"]
        assert st["min"] > 0    # similar, but no bit-identical perceptual hash

    def test_a_planted_cross_split_duplicate_is_found(self, tmp_path):
        """The CVC-300 / CVC-ColonDB question of brief §1.2.1, in miniature."""
        rng = np.random.default_rng(6)
        for split in ("A", "B"):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
            for i in range(5):
                shift = (0 if split == "A" else 50) + i * 1.7
                img = textured(rng, 60, 72, seed_shift=shift)
                img.save(tmp_path / split / "images" / f"{i}.png")
                Image.new("L", img.size, 255).save(tmp_path / split / "masks" / f"{i}.png")
        dup = Image.open(tmp_path / "A" / "images" / "0.png")
        dup.save(tmp_path / "B" / "images" / "0.png")
        Image.new("L", dup.size, 255).save(tmp_path / "B" / "masks" / "0.png")

        manifest = build_manifest(tmp_path, ["A", "B"])
        rep = audit(tmp_path, manifest, progress=False)
        assert rep.matrix["A"]["B"] >= 1
        assert rep.matrix["B"]["A"] >= 1
        assert any(p["split_a"] != p["split_b"] for p in rep.pairs)
        assert len(rep.exact_pixel_duplicates) == 1
        assert rep.nn_stats["B"]["A"]["phash"]["min"] == 0

    def test_nearest_neighbour_stats_are_threshold_free_evidence(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        rep = audit(root, manifest, progress=False)
        st = rep.nn_stats["TrainDataset"]["TestDataset/Fake"]["phash"]
        assert set(st) >= {"min", "p01", "p05", "median", "n_le_2", "n_le_6", "n_le_10"}

    def test_summary_renders(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        text = audit(root, manifest, progress=False).summary()
        assert "TrainDataset" in text and "near-duplicate" in text
