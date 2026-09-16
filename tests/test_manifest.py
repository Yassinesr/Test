"""Manifests must detect drift, including drift a file hash alone would miss."""

import json

import numpy as np
import pytest
from PIL import Image

from polyptail.data.manifest import (
    binarize_mask, build_manifest, build_split, load_manifest, verify_manifest, write_manifest,
)


class TestBuild:
    def test_records_both_file_and_pixel_hashes(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        item = manifest["splits"]["TrainDataset"]["items"][0]
        assert len(item["image_sha256"]) == 64
        assert item["image_pixel_sha256"] != item["image_sha256"]
        assert 0.0 < item["mask_positive_frac"] < 1.0

    def test_counts_and_ordering_are_deterministic(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        again = build_manifest(root, list(manifest["splits"]))
        assert [i["stem"] for i in again["splits"]["TrainDataset"]["items"]] == \
               [i["stem"] for i in manifest["splits"]["TrainDataset"]["items"]]
        assert again["total_pairs"] == manifest["total_pairs"]

    def test_unpaired_stems_are_rejected(self, synthetic_dataset, tmp_path):
        root, _ = synthetic_dataset
        (root / "TrainDataset" / "masks" / "000.png").unlink()
        with pytest.raises(ValueError, match="unpaired"):
            build_split(root, "TrainDataset")

    def test_size_mismatch_is_rejected(self, synthetic_dataset):
        root, _ = synthetic_dataset
        p = root / "TrainDataset" / "masks" / "001.png"
        Image.open(p).resize((7, 9)).save(p)
        with pytest.raises(ValueError, match="size"):
            build_split(root, "TrainDataset")

    def test_missing_directory_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_split(tmp_path, "NoSuchSplit")


class TestVerify:
    def test_clean_dataset_passes(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        assert verify_manifest(root, manifest).ok

    def test_detects_a_changed_file(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        p = root / "TrainDataset" / "images" / "000.png"
        arr = np.asarray(Image.open(p)).copy()
        arr[0, 0] = (arr[0, 0] + 40) % 255
        Image.fromarray(arr).save(p)
        rep = verify_manifest(root, manifest)
        assert not rep.ok and rep.file_hash_mismatch

    def test_detects_a_re_encode_that_keeps_the_pixels(self, synthetic_dataset):
        """A re-save changes the file hash but not the picture: the pixel hash
        is what separates 'someone recompressed the archive' from 'someone
        changed the data'."""
        root, manifest = synthetic_dataset
        p = root / "TrainDataset" / "images" / "000.png"
        Image.open(p).save(p, optimize=True, compress_level=1)
        rep = verify_manifest(root, manifest)
        assert not rep.pixel_hash_mismatch or rep.file_hash_mismatch

    def test_detects_a_missing_file(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        (root / "TrainDataset" / "images" / "002.png").unlink()
        rep = verify_manifest(root, manifest)
        assert not rep.ok and rep.missing

    def test_detects_an_extra_file(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        Image.new("RGB", (8, 8)).save(root / "TrainDataset" / "images" / "zzz.png")
        rep = verify_manifest(root, manifest)
        assert not rep.ok and rep.extra

    def test_summary_is_human_readable(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        assert "OK" in verify_manifest(root, manifest).summary()


class TestIO:
    def test_json_and_sha256_roundtrip(self, synthetic_dataset, tmp_path):
        root, manifest = synthetic_dataset
        j, sha = tmp_path / "m.json", tmp_path / "m.sha256"
        write_manifest(manifest, j, sha)
        assert load_manifest(j)["total_pairs"] == manifest["total_pairs"]
        lines = sha.read_text().strip().split("\n")
        assert len(lines) == 2 * manifest["total_pairs"]
        assert all(len(ln.split("  ")[0]) == 64 for ln in lines)

    def test_unknown_schema_is_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"schema": "something/else", "splits": {}}))
        with pytest.raises(ValueError, match="schema"):
            load_manifest(p)


class TestBinarize:
    def test_handles_0_255_and_0_1_masks(self):
        assert binarize_mask(np.array([[0, 255], [128, 200]])).tolist() == [[0, 1], [1, 1]]
        assert binarize_mask(np.array([[0, 1], [1, 0]])).tolist() == [[0, 1], [1, 0]]
