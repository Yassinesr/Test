"""Config loading, inheritance and override coercion."""

from pathlib import Path

import pytest

from polyptail.config import Config, apply_overrides, config_to_dict, dump_config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class TestShippedConfigs:
    @pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("*.yaml")))
    def test_every_shipped_config_loads_and_validates(self, name):
        load_config(CONFIGS / name).validate()

    def test_inheritance_overrides_only_the_named_keys(self):
        base = load_config(CONFIGS / "base.yaml")
        a1 = load_config(CONFIGS / "a1_pottc.yaml")
        assert a1.pot.tail.mode == "gpd"
        assert base.pot.tail.mode == "none"
        assert a1.optim.lr == base.optim.lr
        assert a1.data.size == base.data.size

    def test_the_ablation_arms_differ_only_in_the_tail_term(self):
        """A1/A2/A3 must be identical apart from the weighting, or the
        ablation measures something other than the weighting."""
        arms = {n: load_config(CONFIGS / f"{n}.yaml")
                for n in ("a1_pottc", "a2_cvar", "a3_ohem")}
        dicts = {n: config_to_dict(c) for n, c in arms.items()}
        for n, d in dicts.items():
            d.pop("run")
            d["pot"]["tail"].pop("mode")
            d["pot"]["tail"].pop("cvar_selection")
        assert dicts["a1_pottc"] == dicts["a2_cvar"] == dicts["a3_ohem"]

    def test_the_shape_clamp_cannot_bind_in_the_normal_regime(self):
        """A lower clamp at -0.5 inverts the weight profile for a bounded
        score; the shipped A1 config must sit well clear of it."""
        assert load_config(CONFIGS / "a1_pottc.yaml").pot.tail.xi_min <= -1.0


class TestOverrides:
    def test_types_are_coerced_from_the_current_value(self):
        c = apply_overrides(Config(), [
            "optim.epochs=7", "optim.lr=3e-5", "run.deterministic=true",
            "optim.scales=[1.0, 1.25]", "model.pretrained=/tmp/x.pth",
        ])
        assert c.optim.epochs == 7 and isinstance(c.optim.epochs, int)
        assert c.optim.lr == pytest.approx(3e-5)
        assert c.run.deterministic is True
        assert c.optim.scales == [1.0, 1.25]
        assert c.model.pretrained == "/tmp/x.pth"

    def test_none_is_expressible(self):
        c = apply_overrides(Config(), ["model.pretrained=none"])
        assert c.model.pretrained is None

    def test_nested_tail_keys_are_reachable(self):
        c = apply_overrides(Config(), ["pot.tail.mode=ohem", "pot.tail.lam=0.7"])
        assert c.pot.tail.mode == "ohem" and c.pot.tail.lam == pytest.approx(0.7)

    def test_unknown_key_is_rejected(self):
        with pytest.raises(KeyError):
            apply_overrides(Config(), ["optim.nope=1"])

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ValueError):
            apply_overrides(Config(), ["optim.epochs"])

    def test_invalid_combination_is_rejected(self):
        with pytest.raises(ValueError):
            apply_overrides(Config(), ["run.select=val_dice"])
        with pytest.raises(ValueError):
            apply_overrides(Config(), ["optim.amp=fp8"])
        with pytest.raises(ValueError):
            apply_overrides(Config(), ["pot.tail.alpha=0.9"])

    def test_unit_scale_tail_requires_scale_one_present(self):
        with pytest.raises(ValueError):
            apply_overrides(Config(), ["optim.scales=[0.75, 1.25]"])


class TestRoundTrip:
    def test_dump_then_load_is_the_identity(self, tmp_path):
        c = apply_overrides(load_config(CONFIGS / "a1_pottc.yaml"),
                            ["run.seed=3", "pot.tail.lam=0.9"])
        dump_config(c, tmp_path / "c.yaml")
        assert config_to_dict(load_config(tmp_path / "c.yaml")) == config_to_dict(c)


class TestValidationSetGuards:
    """Selection has one job: never see the test set, never see the training
    set. Each guard below is a way of failing that while looking fine."""

    def test_a_directory_and_a_fold_together_are_rejected(self):
        cfg = Config()
        cfg.data.val_split = "ValidationDataset"
        cfg.data.val_frac = 0.1
        with pytest.raises(ValueError, match="Pick one"):
            cfg.validate()

    def test_val_dice_without_any_validation_set_is_rejected(self):
        cfg = Config()
        cfg.run.select = "val_dice"
        with pytest.raises(ValueError, match="needs a validation set"):
            cfg.validate()

    def test_val_dice_with_a_directory_is_accepted(self):
        cfg = Config()
        cfg.run.select = "val_dice"
        cfg.data.val_split = "ValidationDataset"
        cfg.validate()

    def test_a_validation_split_that_is_also_a_test_split_is_rejected(self):
        """Selection on the test set is the protocol violation this whole
        project exists to avoid; renaming it does not make it legal."""
        cfg = Config()
        cfg.data.val_split = cfg.data.test_splits[0]
        with pytest.raises(ValueError, match="wearing a different name"):
            cfg.validate()

    def test_a_validation_split_that_is_the_training_split_is_rejected(self):
        cfg = Config()
        cfg.data.val_split = cfg.data.train_split
        with pytest.raises(ValueError, match="images it trained on"):
            cfg.validate()

    def test_the_shipped_base_selects_on_validation_not_test(self):
        cfg = load_config(CONFIGS / "base.yaml")
        assert cfg.run.select == "val_dice"
        assert cfg.data.val_split == "ValidationDataset"
        assert cfg.data.val_split not in cfg.data.test_splits

    @pytest.mark.parametrize("name", sorted(p.name for p in CONFIGS.glob("a*.yaml")))
    def test_no_ablation_arm_selects_differently_from_the_others(self, name):
        """A comparison between arms selected by different rules measures the
        rules as much as the method."""
        base = load_config(CONFIGS / "base.yaml")
        arm = load_config(CONFIGS / name)
        assert (arm.run.select, arm.data.val_split, arm.data.val_frac) == (
            base.run.select, base.data.val_split, base.data.val_frac)


class TestClearingAnOptionalField:
    """base.yaml sets data.val_split to a string, so `=null` has to clear it
    rather than store the word."""

    def test_null_clears_an_optional_string(self):
        cfg = load_config(CONFIGS / "base.yaml")
        assert cfg.data.val_split == "ValidationDataset"
        apply_overrides(cfg, ["data.val_split=null", "run.select=last"])
        assert cfg.data.val_split is None

    def test_none_clears_it_too(self):
        cfg = load_config(CONFIGS / "base.yaml")
        apply_overrides(cfg, ["data.val_split=None", "run.select=last"])
        assert cfg.data.val_split is None

    def test_a_required_string_is_left_alone(self):
        """run.name is not Optional, so the word stays a word rather than
        silently becoming a None that breaks a path join later."""
        cfg = load_config(CONFIGS / "base.yaml")
        apply_overrides(cfg, ["run.name=null"])
        assert cfg.run.name == "null"


class TestFoldingSplitsBackIntoTraining:
    """The reproduction gate trains on the whole distributed pool, which on a
    re-split dataset means putting the validation half back."""

    def test_the_gate_shape_is_accepted(self):
        cfg = Config()
        cfg.data.extra_train_splits = ["ValidationDataset"]
        cfg.data.val_split = None
        cfg.run.select = "last"
        cfg.validate()

    def test_folding_in_a_test_split_is_rejected(self):
        cfg = Config()
        cfg.data.extra_train_splits = [cfg.data.test_splits[0]]
        with pytest.raises(ValueError, match="training on the test set"):
            cfg.validate()

    def test_folding_in_the_training_split_itself_is_rejected(self):
        cfg = Config()
        cfg.data.extra_train_splits = [cfg.data.train_split]
        with pytest.raises(ValueError, match="loaded twice"):
            cfg.validate()

    def test_training_on_the_split_you_select_on_is_rejected(self):
        """The whole point of the validation split, undone in one line."""
        cfg = Config()
        cfg.data.val_split = "ValidationDataset"
        cfg.data.extra_train_splits = ["ValidationDataset"]
        cfg.run.select = "val_dice"
        with pytest.raises(ValueError, match="selected on images it"):
            cfg.validate()

    def test_the_shipped_configs_fold_in_nothing(self):
        for name in sorted(p.name for p in CONFIGS.glob("*.yaml")):
            assert load_config(CONFIGS / name).data.extra_train_splits == [], name
