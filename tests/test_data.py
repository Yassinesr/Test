"""Datasets: pairing, resizing, resize direction, and reproducible workers."""

import numpy as np
import pytest

from polyptail.data.dataset import (
    PolypTestDataset, PolypTrainDataset, collate_eval, items_for_split,
)


class TestTrainDataset:
    def test_returns_normalised_image_mask_and_index(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        ds = PolypTrainDataset(root, items_for_split(manifest, "TrainDataset"), size=64)
        img, mask, idx = ds[3]
        assert img.shape == (3, 64, 64) and mask.shape == (1, 64, 64)
        assert idx == 3
        assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
        assert float(img.mean()) < 3.0     # normalised, not raw 0-255

    def test_index_identifies_the_manifest_item(self, synthetic_dataset):
        """Needed so per-image deficits can be traced back to a file, which is
        how label noise gets distinguished from genuine difficulty."""
        root, manifest = synthetic_dataset
        items = items_for_split(manifest, "TrainDataset")
        ds = PolypTrainDataset(root, items, size=32)
        for i in (0, 5, len(ds) - 1):
            assert ds[i][2] == i
            assert items[i].stem is not None

    def test_pairs_come_from_the_manifest_not_from_directory_order(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        items = items_for_split(manifest, "TrainDataset")
        ds = PolypTrainDataset(root, list(reversed(items)), size=32)
        _, mask, _ = ds[0]
        assert float(mask.sum()) > 0        # still the mask that belongs to that image

    @pytest.mark.parametrize("augment", ["none", "flip", "flip_rotate"])
    def test_augmentation_keeps_image_and_mask_aligned(self, synthetic_dataset, augment):
        root, manifest = synthetic_dataset
        ds = PolypTrainDataset(root, items_for_split(manifest, "TrainDataset"),
                               size=64, augment=augment)
        for i in range(6):
            img, mask, _ = ds[i]
            bright = img.mean(0)
            inside = float(bright[mask[0] > 0.5].mean())
            outside = float(bright[mask[0] <= 0.5].mean())
            assert inside > outside      # the synthetic polyp is brighter

    def test_unknown_options_are_rejected(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        items = items_for_split(manifest, "TrainDataset")
        with pytest.raises(ValueError):
            PolypTrainDataset(root, items, augment="rotate90")
        with pytest.raises(ValueError):
            PolypTrainDataset(root, items, mask_interpolation="bicubic")


class TestTestDataset:
    def test_ground_truth_stays_at_native_resolution(self, synthetic_dataset):
        """The locked protocol resizes predictions up to the GT grid, never the
        GT down to 352.  Downsampling the GT inflates scores on ETIS and
        ColonDB, the two sets this candidate is about."""
        root, manifest = synthetic_dataset
        items = items_for_split(manifest, "TestDataset/Fake")
        ds = PolypTestDataset(root, items, size=64)
        for i, item in enumerate(items):
            sample = ds[i]
            assert sample["image"].shape == (3, 64, 64)
            assert tuple(sample["gt"].shape) == (item.image_size[1], item.image_size[0])

    def test_ground_truth_is_binary(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        ds = PolypTestDataset(root, items_for_split(manifest, "TestDataset/Fake"), size=32)
        gt = ds[0]["gt"].numpy()
        assert set(np.unique(gt)).issubset({0, 1})

    def test_collate_keeps_variable_sized_ground_truths_as_a_list(self, synthetic_dataset):
        root, manifest = synthetic_dataset
        ds = PolypTestDataset(root, items_for_split(manifest, "TestDataset/Fake"), size=32)
        batch = collate_eval([ds[0], ds[1], ds[2]])
        assert batch["image"].shape[0] == 3
        assert isinstance(batch["gt"], list) and len(batch["gt"]) == 3
        assert len(batch["stem"]) == 3

    def test_missing_split_is_a_keyerror(self, synthetic_dataset):
        _, manifest = synthetic_dataset
        with pytest.raises(KeyError):
            items_for_split(manifest, "TestDataset/NoSuch")
