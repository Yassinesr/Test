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


class TestFindings:
    """A matrix is evidence; a finding is what you act on. These come only
    from the parts of the report that no threshold choice can move."""

    def _subset_tree(self, tmp_path, n_shared=6, n_extra=6):
        """`Small` is entirely contained in `Big`, as CVC-300 is in ColonDB."""
        rng = np.random.default_rng(3)
        for split in ("Small", "Big"):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
        for i in range(n_shared):
            img = textured(rng, 64, 80, seed_shift=i * 6.1)
            for split in ("Small", "Big"):
                img.save(tmp_path / split / "images" / f"s{i}.png")
                Image.new("L", img.size, 255).save(tmp_path / split / "masks" / f"s{i}.png")
        for i in range(n_extra):
            img = textured(rng, 64, 80, seed_shift=200 + i * 6.1)
            img.save(tmp_path / "Big" / "images" / f"b{i}.png")
            Image.new("L", img.size, 255).save(tmp_path / "Big" / "masks" / f"b{i}.png")
        return audit(tmp_path, build_manifest(tmp_path, ["Small", "Big"]), progress=False)

    def test_a_contained_split_is_named_as_a_subset(self, tmp_path):
        rep = self._subset_tree(tmp_path)
        found = "\n".join(rep.findings())
        assert "Small -> Big" in found
        assert "subset of Big" in found
        assert "counts those images twice" in found

    def test_the_containing_split_is_not_called_a_subset(self, tmp_path):
        """Containment is asymmetric, and reporting it symmetrically would
        tell you to drop the larger set, which is the wrong fix."""
        rep = self._subset_tree(tmp_path, n_shared=6, n_extra=30)
        found = "\n".join(rep.findings())
        assert "Small -> Big" in found
        assert "Big -> Small: median nearest-neighbour pHash distance 0.0" not in found

    def test_exact_duplicates_are_located_not_just_counted(self, tmp_path):
        rep = self._subset_tree(tmp_path)
        loc = rep.exact_duplicates_by_location()
        assert loc == {"Big <-> Small": 6}
        assert any("span Big <-> Small" in f for f in rep.findings())

    def test_within_split_duplicates_are_reported_separately(self, tmp_path):
        """A split that repeats itself is smaller than its count says, which
        matters for a method weighted by the tail of a per-image loss."""
        rng = np.random.default_rng(4)
        (tmp_path / "A" / "images").mkdir(parents=True)
        (tmp_path / "A" / "masks").mkdir(parents=True)
        for i in range(5):
            img = textured(rng, 64, 80, seed_shift=i * 9.3)
            img.save(tmp_path / "A" / "images" / f"{i}.png")
            Image.new("L", img.size, 255).save(tmp_path / "A" / "masks" / f"{i}.png")
        dup = Image.open(tmp_path / "A" / "images" / "0.png")
        dup.save(tmp_path / "A" / "images" / "copy.png")
        Image.new("L", dup.size, 255).save(tmp_path / "A" / "masks" / "copy.png")
        rep = audit(tmp_path, build_manifest(tmp_path, ["A"]), progress=False)
        assert rep.exact_duplicates_by_location() == {"within A": 1}
        assert any("fewer distinct images than pairs" in f for f in rep.findings())

    def test_independent_splits_produce_no_findings(self, tmp_path):
        rng = np.random.default_rng(11)
        for split, base in (("A", 0.0), ("B", 400.0)):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
            for i in range(6):
                img = textured(rng, 64, 80, seed_shift=base + i * 21.7)
                img.save(tmp_path / split / "images" / f"{i}.png")
                Image.new("L", img.size, 255).save(tmp_path / split / "masks" / f"{i}.png")
        rep = audit(tmp_path, build_manifest(tmp_path, ["A", "B"]), progress=False)
        assert rep.findings() == [], rep.summary()

    def test_the_summary_carries_the_findings(self, tmp_path):
        rep = self._subset_tree(tmp_path)
        assert "Findings (threshold-free evidence only):" in rep.summary()


class TestSelectionLeakage:
    """Only a decoded-pixel collision blocks a run. An earlier version blocked
    on the flag count, which is threshold-dependent -- the same signal this
    report tells you never to act on alone."""

    def _tree(self, tmp_path, planted_exact=False, val_shift=500.0):
        rng = np.random.default_rng(5)
        for split in ("TrainDataset", "ValidationDataset"):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
        for i in range(8):
            img = textured(rng, 64, 80, seed_shift=i * 17.3)
            img.save(tmp_path / "TrainDataset" / "images" / f"{i}.png")
            Image.new("L", img.size, 255).save(tmp_path / "TrainDataset" / "masks" / f"{i}.png")
        for i in range(6):
            img = textured(rng, 64, 80, seed_shift=val_shift + i * 17.3)
            img.save(tmp_path / "ValidationDataset" / "images" / f"v{i}.png")
            Image.new("L", img.size, 255).save(
                tmp_path / "ValidationDataset" / "masks" / f"v{i}.png")
        if planted_exact:
            dup = Image.open(tmp_path / "TrainDataset" / "images" / "0.png")
            dup.save(tmp_path / "ValidationDataset" / "images" / "v0.png")
            Image.new("L", dup.size, 255).save(
                tmp_path / "ValidationDataset" / "masks" / "v0.png")
        manifest = build_manifest(tmp_path, ["TrainDataset", "ValidationDataset"])
        return audit(tmp_path, manifest, progress=False)

    def test_the_same_image_on_both_sides_blocks(self, tmp_path):
        v = self._tree(tmp_path, planted_exact=True).selection_leakage()
        assert v["status"] == "blocking"
        text = " ".join(v["lines"])
        assert "byte-identical" in text and "blocks the run" in text

    def test_close_neighbours_alone_do_not_block(self, tmp_path):
        """Two halves of one corpus of video frames are always going to be
        close. Calling that a blocker stops a run that is fine."""
        v = self._tree(tmp_path, val_shift=1.0).selection_leakage()
        assert v["status"] == "report"
        text = " ".join(v["lines"])
        assert "no byte-identical images" in text
        assert "blocks the run" not in text
        assert "threshold-dependent and not a verdict" in text

    def test_it_says_where_the_number_sits_in_its_own_column(self, tmp_path):
        """A median means nothing against 32 in a self-similar corpus; it
        means something against the other splits' distances into training."""
        rep = self._tree(tmp_path)
        for split in ("TestDataset/Kvasir",):
            (tmp_path / split / "images").mkdir(parents=True)
            (tmp_path / split / "masks").mkdir(parents=True)
            rng = np.random.default_rng(9)
            for i in range(6):
                img = textured(rng, 64, 80, seed_shift=900 + i * 17.3)
                img.save(tmp_path / split / "images" / f"{i}.png")
                Image.new("L", img.size, 255).save(tmp_path / split / "masks" / f"{i}.png")
        rep = audit(tmp_path, build_manifest(
            tmp_path, ["TrainDataset", "ValidationDataset", "TestDataset/Kvasir"]),
            progress=False)
        text = " ".join(rep.selection_leakage()["lines"])
        assert "read down the same column" in text
        assert "TestDataset/Kvasir" in text

    def test_no_validation_split_is_reported_as_not_checked(self, tmp_path):
        rng = np.random.default_rng(7)
        (tmp_path / "TrainDataset" / "images").mkdir(parents=True)
        (tmp_path / "TrainDataset" / "masks").mkdir(parents=True)
        for i in range(5):
            img = textured(rng, 64, 80, seed_shift=i * 17.3)
            img.save(tmp_path / "TrainDataset" / "images" / f"{i}.png")
            Image.new("L", img.size, 255).save(tmp_path / "TrainDataset" / "masks" / f"{i}.png")
        rep = audit(tmp_path, build_manifest(tmp_path, ["TrainDataset"]), progress=False)
        v = rep.selection_leakage()
        assert v["status"] == "absent"
        assert "did not run" in " ".join(v["lines"])
