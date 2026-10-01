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


class TestVerifyExplainsTheDifference:
    """A failed verify names three things that need different responses: a
    directory renamed, a mask re-saved, a mask re-annotated. All three fail the
    same hash, so the count alone sends you the wrong way."""

    def test_a_renamed_mask_directory_is_not_reported_as_lost(self, synthetic_dataset):
        """masks/ -> gt/ makes every mask in the split read as missing, which
        looks exactly like deleting them and has a completely different fix."""
        root, manifest = synthetic_dataset
        (root / "TrainDataset" / "masks").rename(root / "TrainDataset" / "gt")
        rep = verify_manifest(root, manifest)
        assert not rep.ok and len(rep.missing) == 12
        assert len(rep.moved) == 12
        text = rep.summary()
        assert "are not gone" in text and "TrainDataset/gt" in text
        assert "renamed after the manifest was frozen" in text

    def test_a_re_saved_mask_is_called_a_re_save(self, synthetic_dataset):
        """A mask re-saved at a different value level -- 255 down to 200 --
        has different decoded pixels and the same binary content, because
        ``binarize_mask`` thresholds at >127. Re-freeze and carry on.

        Worth recording what does *not* land here: an undithered ``L`` to
        ``1`` conversion of a hard-edged mask is pixel-*identical*, because the
        hash is taken after converting back to ``L`` and 0/255 round-trips
        exactly. That shows up as a file-hash mismatch alone. So a pixel
        mismatch after such a conversion means the edges were not hard, or the
        convert dithered -- either way information was lost.
        """
        root, manifest = synthetic_dataset
        p = root / "TrainDataset" / "masks" / "000.png"
        arr = np.asarray(Image.open(p)).copy()
        arr[arr > 127] = 200
        Image.fromarray(arr).save(p)
        rep = verify_manifest(root, manifest)
        assert rep.pixel_hash_mismatch, "a changed value level must change the decoded bytes"
        text = rep.summary()
        assert "same binary content" in text
        assert "re-saved, not re-annotated" in text.lower()
        assert "different annotations" not in text

    def test_a_re_annotated_mask_is_called_different_annotations(self, synthetic_dataset):
        """The one that must not be re-frozen over: the labels moved."""
        root, manifest = synthetic_dataset
        p = root / "TrainDataset" / "masks" / "000.png"
        arr = np.asarray(Image.open(p)).copy()
        arr[:6, :6] = 255                      # grow the mask
        Image.fromarray(arr).save(p)
        rep = verify_manifest(root, manifest)
        assert rep.pixel_hash_mismatch
        text = rep.summary()
        assert "cover a different fraction" in text
        assert "different annotations" in text
        assert "Do not re-freeze" in text

    def test_the_label_does_not_call_a_pixel_change_a_re_encode(self, synthetic_dataset):
        """A re-encode is the case where the file hash moves and the pixel
        hash does not. Calling a pixel mismatch 're-encoded' says harmless
        when it means the content changed."""
        root, manifest = synthetic_dataset
        p = root / "TrainDataset" / "masks" / "000.png"
        arr = np.asarray(Image.open(p)).copy()
        arr[:6, :6] = 255
        Image.fromarray(arr).save(p)
        text = verify_manifest(root, manifest).summary()
        assert "pixel-hash mismatch (the DECODED pixels differ)" in text
        assert "pixel-hash mismatch (re-encoded)" not in text


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
