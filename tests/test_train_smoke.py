"""End-to-end: manifest -> train -> evaluate -> saved artefacts, on CPU."""

import json
from pathlib import Path

import pytest
import torch

from polyptail.config import apply_overrides, load_config
from polyptail.engine.train import train

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def smoke_cfg(dataset, tmp_path, **overrides):
    root, _ = dataset
    cfg = load_config(CONFIGS / "base.yaml")
    base = {
        "data.root": str(root), "data.manifest": str(root / "manifest.json"),
        "data.test_splits": '["TestDataset/Fake"]', "data.size": "32",
        "data.num_workers": "0", "data.verify": "hash",
        "model.name": "tiny_unet", "optim.epochs": "2", "optim.batch_size": "4",
        "optim.scales": "[1.0]", "optim.amp": "off",
        "eval.num_workers": "0", "run.out_dir": str(tmp_path / "runs"),
        "run.log_every": "100", "run.seed": "0",
        # base.yaml selects on ValidationDataset; each test here opts into a
        # selection rule explicitly, so none of them inherits one silently.
        "data.val_split": "null", "run.select": "last",
    }
    base.update({k: str(v) for k, v in overrides.items()})
    return apply_overrides(cfg, [f"{k}={v}" for k, v in base.items()])


@pytest.mark.parametrize("mode", ["none", "gpd", "cvar", "ohem"])
def test_every_tail_mode_trains_and_evaluates(synthetic_dataset, tmp_path, mode):
    cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
        "run.name": f"smoke_{mode}", "pot.tail.mode": mode,
        "pot.tail.buffer_size": 48, "pot.tail.min_buffer": 12,
        "pot.tail.min_exceedances": 4, "pot.tail.warmup_frac": 0.0,
    })
    summary = train(cfg)
    agg = summary["per_split"]["TestDataset/Fake"]
    assert 0.0 <= agg["dice"] <= 1.0
    assert 0.0 <= agg["dice_le_05"] <= 1.0
    assert agg["n"] == 6


class TestArtefacts:
    def test_a_run_writes_everything_needed_to_audit_it(self, synthetic_dataset, tmp_path):
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "artefacts", "pot.tail.mode": "gpd",
            "pot.tail.buffer_size": 48, "pot.tail.min_buffer": 12,
            "pot.tail.min_exceedances": 4, "pot.tail.warmup_frac": 0.0,
        })
        train(cfg)
        out = tmp_path / "runs" / "artefacts"
        for name in ("config.yaml", "environment.json", "model_info.json",
                     "results.json", "per_image_metrics.json", "train_metrics.jsonl",
                     "per_image_deficits.csv", "last.pth", "train.log"):
            assert (out / name).is_file(), f"missing artefact {name}"
        # retained prediction maps -- blocking item 9
        preds = list((out / "predictions" / "Fake").glob("*.png"))
        assert len(preds) == 6

    def test_per_image_deficits_name_the_training_images(self, synthetic_dataset, tmp_path):
        """How label noise is told apart from genuine difficulty."""
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "deficits", "pot.tail.mode": "gpd",
            "pot.tail.buffer_size": 48, "pot.tail.min_buffer": 12,
            "pot.tail.min_exceedances": 4,
        })
        train(cfg)
        rows = (tmp_path / "runs" / "deficits" / "per_image_deficits.csv").read_text().strip().split("\n")
        assert rows[0] == "epoch,dataset_index,stem,deficit"
        assert len(rows) == 1 + 2 * 12       # 2 epochs x 12 training images
        assert all(0.0 <= float(r.split(",")[3]) <= 1.0 for r in rows[1:])

    def test_results_json_records_the_manifest_and_parameter_count(self, synthetic_dataset, tmp_path):
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{"run.name": "prov"})
        summary = train(cfg)
        blob = json.loads((tmp_path / "runs" / "prov" / "results.json").read_text())
        assert blob["params"] == summary["params"] > 0
        assert blob["manifest"].endswith("manifest.json")
        assert blob["wall_clock_seconds"] > 0


class TestProtocolGuards:
    def test_training_refuses_a_dataset_that_drifted_from_the_manifest(self, synthetic_dataset, tmp_path):
        import numpy as np
        from PIL import Image

        root, _ = synthetic_dataset
        p = root / "TrainDataset" / "images" / "000.png"
        arr = np.asarray(Image.open(p)).copy()
        arr[0, 0] = (arr[0, 0] + 77) % 255
        Image.fromarray(arr).save(p)
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{"run.name": "drift"})
        with pytest.raises(RuntimeError, match="frozen manifest"):
            train(cfg)

    def test_selection_on_a_validation_fold_holds_images_out_of_training(
            self, synthetic_dataset, tmp_path):
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "val", "data.val_frac": 0.25, "run.select": "val_dice",
        })
        train(cfg)
        assert (tmp_path / "runs" / "val" / "best.pth").is_file()
        blob = json.loads((tmp_path / "runs" / "val" / "results.json").read_text())
        assert blob["selected"]["source"] == "val_dice"

    def test_same_seed_reproduces_a_run_to_tolerance(self, synthetic_dataset, tmp_path):
        """A tolerance, not equality -- deliberately.

        `run.deterministic=true` removes cuDNN autotuning and every kernel
        that has a deterministic alternative, but bilinear upsample backward
        has none on CUDA and this decoder uses it throughout. So a fixed seed
        reproduces a GPU run closely, not bit-exactly. Asserting equality here
        would pass on CPU and quietly mislead anyone reading it as a guarantee
        for the runs that matter.
        """
        a = train(smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "seedA", "run.seed": 7, "run.deterministic": "true"}))
        b = train(smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "seedB", "run.seed": 7, "run.deterministic": "true"}))
        assert a["per_split"]["TestDataset/Fake"]["dice"] == pytest.approx(
            b["per_split"]["TestDataset/Fake"]["dice"], abs=1e-6)

    def test_different_seeds_give_different_runs(self, synthetic_dataset, tmp_path):
        """Compared on the training trajectory, not the test metric: two epochs
        of a 0.1M-parameter net on 12 images leave the test score saturated, so
        equal test Dice there would prove nothing either way."""
        train(smoke_cfg(synthetic_dataset, tmp_path, **{"run.name": "s0", "run.seed": 0}))
        train(smoke_cfg(synthetic_dataset, tmp_path, **{"run.name": "s1", "run.seed": 1}))

        def final_loss(name):
            lines = (tmp_path / "runs" / name / "train_metrics.jsonl").read_text().strip().split("\n")
            return json.loads(lines[-1])["base_loss"]

        assert final_loss("s0") != final_loss("s1")


def test_the_loss_term_adds_no_parameters(synthetic_dataset, tmp_path):
    """The equal-parameter control, asserted rather than asserted-in-prose."""
    a = train(smoke_cfg(synthetic_dataset, tmp_path, **{
        "run.name": "p_none", "pot.tail.mode": "none"}))
    b = train(smoke_cfg(synthetic_dataset, tmp_path, **{
        "run.name": "p_gpd", "pot.tail.mode": "gpd",
        "pot.tail.buffer_size": 48, "pot.tail.min_buffer": 12, "pot.tail.min_exceedances": 4}))
    assert a["params"] == b["params"]


class TestDeterminismIsHonest:
    """Asking for determinism on CUDA must not silently under-deliver."""

    def test_warns_that_gpu_runs_are_not_bit_identical(self, monkeypatch, caplog):
        import logging

        import torch

        from polyptail.utils.env import seed_everything

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: None)
        with caplog.at_level(logging.WARNING, logger="polyptail.utils.env"):
            seed_everything(0, deterministic=True)
        text = " ".join(r.message for r in caplog.records)
        assert "bilinear" in text and "NOT" in text, (
            "requesting determinism on CUDA must say that bit-identical runs are "
            f"not achievable; got: {text!r}"
        )

    def test_stays_quiet_when_determinism_is_not_requested(self, monkeypatch, caplog):
        import logging

        import torch

        from polyptail.utils.env import seed_everything

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: None)
        with caplog.at_level(logging.WARNING, logger="polyptail.utils.env"):
            seed_everything(0, deterministic=False)
        assert not [r for r in caplog.records if "bilinear" in r.message]


class TestHeldOutValidationDirectory:
    """`data.val_split` selects on a directory rather than on a fold carved
    out at a seed. The gain is provenance -- the split is in the manifest, so
    which images chose the checkpoint survives the run."""

    def test_it_selects_and_records_where_the_criterion_came_from(
            self, synthetic_dataset, tmp_path):
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "valdir", "data.val_split": "ValidationDataset",
            "run.select": "val_dice",
        })
        train(cfg)
        blob = json.loads((tmp_path / "runs" / "valdir" / "results.json").read_text())
        assert blob["selected"]["source"] == "val_dice"
        assert blob["selected"]["val_set"] == "ValidationDataset"
        assert blob["selected"]["val_pairs"] == 5

    def test_the_training_split_keeps_every_image(self, synthetic_dataset, tmp_path):
        """A held-out directory must not also shrink the training split --
        that is what val_frac does, and doing both would be a silent third
        experiment."""
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "valdir2", "data.val_split": "ValidationDataset",
            "run.select": "val_dice", "pot.tail.mode": "gpd",
            "pot.tail.buffer_size": 48, "pot.tail.min_buffer": 12,
            "pot.tail.min_exceedances": 4,
        })
        train(cfg)
        rows = (tmp_path / "runs" / "valdir2" / "per_image_deficits.csv"
                ).read_text().strip().split("\n")[1:]
        stems = {r.split(",")[2] for r in rows}
        assert len(stems) == 12, "all 12 training images are still trained on"
        assert not any(s.startswith("v") for s in stems), "no validation image trains"

    def test_an_overlap_with_training_is_refused_before_the_first_epoch(
            self, synthetic_dataset, tmp_path):
        """Name-level overlap is the cheap check; it is worth running because
        the expensive one -- pixel identity, via tools/hash_collisions.py --
        is not something a training run can do for itself."""
        import shutil

        from polyptail.data.manifest import build_manifest, write_manifest

        root, _ = synthetic_dataset
        for sub in ("images", "masks"):
            shutil.copy(root / "TrainDataset" / sub / "000.png",
                        root / "ValidationDataset" / sub / "000.png")
        manifest = build_manifest(root, ["TrainDataset", "ValidationDataset", "TestDataset/Fake"])
        write_manifest(manifest, root / "manifest.json", root / "manifest.sha256")
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "overlap", "data.val_split": "ValidationDataset",
            "run.select": "val_dice",
        })
        with pytest.raises(ValueError, match="appear in both"):
            train(cfg)

    def test_a_missing_validation_split_names_the_fix(self, synthetic_dataset, tmp_path):
        cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
            "run.name": "novaldir", "data.val_split": "NoSuchDataset",
            "run.select": "val_dice",
        })
        with pytest.raises(KeyError, match="freeze_manifest"):
            train(cfg)


def test_folding_the_validation_split_back_in_trains_on_everything(
        synthetic_dataset, tmp_path):
    """The reproduction gate: 12 training + 5 validation images, all trained
    on, nothing held out, checkpoint taken at the last epoch."""
    cfg = smoke_cfg(synthetic_dataset, tmp_path, **{
        "run.name": "gate", "data.extra_train_splits": '["ValidationDataset"]',
        "data.val_split": "null", "run.select": "last",
        "pot.tail.mode": "gpd", "pot.tail.buffer_size": 48,
        "pot.tail.min_buffer": 12, "pot.tail.min_exceedances": 4,
    })
    train(cfg)
    rows = (tmp_path / "runs" / "gate" / "per_image_deficits.csv"
            ).read_text().strip().split("\n")[1:]
    stems = {r.split(",")[2] for r in rows}
    assert len(stems) == 17, "12 training + 5 validation images"
    assert sum(s.startswith("v") for s in stems) == 5
    blob = json.loads((tmp_path / "runs" / "gate" / "results.json").read_text())
    assert blob["selected"]["source"] == "last"
