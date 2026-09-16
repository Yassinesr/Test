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
