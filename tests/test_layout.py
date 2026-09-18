"""The ./dataset layout checker: it must name the actual problem."""

import pytest
from PIL import Image

from polyptail.data.layout import EXPECTED_COUNTS, check_layout, link_dataset


def build(root, splits, ext_image=".png", ext_mask=".png", size=(32, 24), n_override=None):
    """Create a miniature dataset tree. Counts are tiny; the checker warns
    about that, which the tests account for."""
    for split in splits:
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "masks").mkdir(parents=True, exist_ok=True)
        n = n_override if n_override is not None else 3
        for i in range(n):
            Image.new("RGB", size, (i * 20 % 255, 40, 60)).save(
                root / split / "images" / f"{i:03d}{ext_image}")
            m = Image.new("L", size, 0)
            m.paste(255, (4, 4, 12, 12))
            m.save(root / split / "masks" / f"{i:03d}{ext_mask}")
    return root


ALL = list(EXPECTED_COUNTS)


class TestHappyPath:
    def test_a_complete_tree_has_no_errors(self, tmp_path):
        build(tmp_path / "dataset", ALL)
        rep = check_layout(tmp_path / "dataset")
        assert rep.ok, rep.summary()
        assert set(rep.counts) == set(EXPECTED_COUNTS)

    def test_counts_below_the_distribution_are_a_warning_not_an_error(self, tmp_path):
        """§1.2 of the brief documents a real 300-vs-380 discrepancy, so a
        count mismatch must not be fatal -- only unexplained."""
        build(tmp_path / "dataset", ALL)
        rep = check_layout(tmp_path / "dataset")
        assert rep.ok
        assert set(rep.count_mismatches) == set(EXPECTED_COUNTS)
        assert any("differ from the counts" in w for w in rep.warnings)

    def test_count_mismatches_collapse_into_one_warning(self, tmp_path):
        """Six near-identical warnings bury the one error worth acting on."""
        build(tmp_path / "dataset", ALL)
        rep = check_layout(tmp_path / "dataset")
        assert sum("differ from the counts" in w for w in rep.warnings) == 1

    def test_tif_test_masks_are_accepted(self, tmp_path):
        """The reference test loader accepts .tif; ETIS originates as TIFF."""
        root = tmp_path / "dataset"
        build(root, ["TrainDataset"])
        build(root, [s for s in ALL if s != "TrainDataset"], ext_mask=".tif")
        rep = check_layout(root)
        assert rep.ok, rep.summary()
        assert not any("filters out" in w for w in rep.warnings)

    def test_jpg_images_are_accepted(self, tmp_path):
        build(tmp_path / "dataset", ALL, ext_image=".jpg")
        rep = check_layout(tmp_path / "dataset")
        assert rep.ok, rep.summary()


class TestRealMistakes:
    def test_missing_root(self, tmp_path):
        rep = check_layout(tmp_path / "nope")
        assert not rep.ok
        assert "does not exist" in rep.errors[0]

    def test_archive_unzipped_one_level_too_deep(self, tmp_path):
        root = tmp_path / "dataset"
        build(root / "TrainDataset", ["TrainDataset"])
        for s in ALL:
            if s != "TrainDataset":
                build(root, [s])
        rep = check_layout(root)
        assert not rep.ok
        assert any("one level too deep" in e for e in rep.errors)

    def test_a_renamed_split_is_identified_by_name(self, tmp_path):
        """CVC-300 is distributed under three different names in the
        literature; the reference code hard-codes one of them."""
        root = tmp_path / "dataset"
        build(root, [s for s in ALL if s != "TestDataset/CVC-300"])
        build(root, ["TestDataset/CVC-T"])
        rep = check_layout(root)
        assert not rep.ok
        assert any("missing directory TestDataset/CVC-300/" in e for e in rep.errors)
        assert any("CVC-T" in e and "looks like it" in e for e in rep.errors)

    def test_unpaired_mask_is_reported_with_examples(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        (root / "TrainDataset" / "masks" / "001.png").unlink()
        rep = check_layout(root)
        assert not rep.ok
        assert any("have no mask with a matching name" in e for e in rep.errors)
        assert any("001" in e for e in rep.errors)

    def test_size_mismatch_between_image_and_mask(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        Image.new("L", (8, 8), 255).save(root / "TrainDataset" / "masks" / "000.png")
        rep = check_layout(root)
        assert not rep.ok
        assert any("differ in size" in e for e in rep.errors)

    def test_missing_images_subdirectory(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        for f in (root / "TrainDataset" / "images").iterdir():
            f.unlink()
        (root / "TrainDataset" / "images").rmdir()
        rep = check_layout(root)
        assert not rep.ok
        assert any("no images/ subdirectory" in e for e in rep.errors)

    def test_empty_split(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        for f in (root / "TestDataset" / "Kvasir" / "images").iterdir():
            f.unlink()
        rep = check_layout(root)
        assert not rep.ok
        assert any("is empty" in e for e in rep.errors)


class TestReferenceCompatibility:
    def test_a_bmp_mask_warns_that_the_reference_would_skip_it(self, tmp_path):
        """Not an error here -- this project reads it -- but the reference
        dataloader filters it out silently, so the two are not comparable."""
        root = tmp_path / "dataset"
        build(root, ALL)
        img = root / "TrainDataset" / "images" / "000.png"
        msk = root / "TrainDataset" / "masks" / "000.png"
        Image.open(msk).save(msk.with_suffix(".bmp"))
        msk.unlink()
        rep = check_layout(root)
        assert rep.ok, rep.summary()
        assert any("filters out" in w and "mask" in w for w in rep.warnings)

    def test_tif_train_masks_warn_because_the_train_loader_only_takes_png(self, tmp_path):
        root = tmp_path / "dataset"
        build(root, ["TrainDataset"], ext_mask=".tif")
        build(root, [s for s in ALL if s != "TrainDataset"])
        rep = check_layout(root)
        assert any("filters out" in w for w in rep.warnings)

    def test_absent_test_directory_is_a_note_not_a_failure(self, tmp_path):
        """The released Train.py requires TestDataset/test/, which the
        distributed archive does not contain. Nothing here needs it."""
        rep = check_layout(build(tmp_path / "dataset", ALL))
        assert rep.ok
        assert any("TestDataset/test/" in n and "crash" in n for n in rep.notes)

    def test_unrecognised_test_directory_is_flagged(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        (root / "TestDataset" / "SomeOtherSet").mkdir()
        rep = check_layout(root)
        assert any("unrecognised directories" in w for w in rep.warnings)


class TestSummary:
    def test_summary_renders_a_count_table_and_next_step(self, tmp_path):
        root = build(tmp_path / "dataset", ALL,
                     n_override=None)
        text = check_layout(root).summary()
        assert "expected" in text and "TrainDataset" in text
        assert "1450" in text

    def test_summary_of_a_clean_tree_points_at_freeze_manifest(self, tmp_path):
        root = tmp_path / "dataset"
        for split, n in EXPECTED_COUNTS.items():
            build(root, [split], n_override=min(n, 2))
        rep = check_layout(root)
        # counts are deliberately small here, so warnings exist but no errors
        assert rep.ok
        assert "freeze_manifest" in rep.summary() or rep.warnings


class TestLinkExistingDataset:
    """The common case: the data is already on the machine, next to another
    checkout. Nothing should be downloaded or copied."""

    def _valid_tree(self, tmp_path):
        src = tmp_path / "Polyp-PVT" / "dataset"
        build(src, ALL)
        (tmp_path / "Test").mkdir(parents=True, exist_ok=True)
        return src, tmp_path / "Test" / "dataset"

    def test_links_a_valid_tree_without_copying(self, tmp_path):
        src, dest = self._valid_tree(tmp_path)
        assert link_dataset(src, dest) == 0
        assert dest.is_symlink()
        assert dest.resolve() == src.resolve()

    def test_is_idempotent(self, tmp_path):
        src, dest = self._valid_tree(tmp_path)
        assert link_dataset(src, dest) == 0
        assert link_dataset(src, dest) == 0

    def test_refuses_to_link_a_broken_tree(self, tmp_path):
        """Fail here, where the message is about the dataset, rather than at
        freeze time where it is about a hash."""
        src, dest = self._valid_tree(tmp_path)
        (src / "TrainDataset" / "masks" / "001.png").unlink()
        assert link_dataset(src, dest) == 1
        assert not dest.exists()

    def test_refuses_a_missing_source(self, tmp_path):
        _, dest = self._valid_tree(tmp_path)
        assert link_dataset(tmp_path / "nope", dest) == 1
        assert not dest.exists()

    def test_refuses_to_clobber_a_real_directory(self, tmp_path):
        src, dest = self._valid_tree(tmp_path)
        dest.mkdir()
        assert link_dataset(src, dest) == 1
        assert not dest.is_symlink()

    def test_refuses_to_silently_repoint_an_existing_link(self, tmp_path):
        src, dest = self._valid_tree(tmp_path)
        other = tmp_path / "other"
        build(other, ALL)
        assert link_dataset(src, dest) == 0
        assert link_dataset(other, dest) == 1
        assert dest.resolve() == src.resolve()

    def test_a_manifest_frozen_through_the_link_verifies_at_the_real_path(self, tmp_path):
        """Manifest paths are root-relative, so sharing one dataset between two
        checkouts -- or moving it -- does not invalidate a frozen protocol."""
        from polyptail.data.manifest import build_manifest, verify_manifest

        src, dest = self._valid_tree(tmp_path)
        assert link_dataset(src, dest) == 0
        manifest = build_manifest(dest, ALL)
        assert manifest["splits"]["TrainDataset"]["items"][0]["image"].startswith("TrainDataset/")
        assert verify_manifest(dest, manifest).ok
        assert verify_manifest(src, manifest).ok


class TestExistingTestDirectory:
    """A Polyp-PVT checkout often has a hand-made TestDataset/test/ so the
    original Train.py can run. It must be recognised, and must not be read."""

    def test_present_is_recognised_not_flagged(self, tmp_path):
        root = build(tmp_path / "dataset", ALL)
        build(root, ["TestDataset/test"])
        rep = check_layout(root)
        assert rep.ok, rep.summary()
        assert not any("unrecognised" in w for w in rep.warnings)
        assert any("TestDataset/test/ exists" in n for n in rep.notes)

    def test_present_is_not_scored(self, tmp_path):
        from polyptail.eval.evaluator import DEFAULT_TEST_SPLITS
        assert "TestDataset/test" not in DEFAULT_TEST_SPLITS
        assert "TestDataset/test" not in EXPECTED_COUNTS
