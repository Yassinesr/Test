"""Architectures and, above all, that a bad checkpoint fails loudly."""

import pytest
import torch

from polyptail.models import MODELS, available_models, build_model
from polyptail.models.polyp_pvt import PolypPVT, load_pretrained_backbone
from polyptail.models.pranet import PraNet
from polyptail.models.pvtv2 import pvt_v2_b2
from polyptail.models.res2net import res2net50_v1b_26w_4s


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class TestBackbones:
    def test_pvtv2_b2_stage_shapes_and_size(self):
        m = pvt_v2_b2()
        outs = m(torch.randn(1, 3, 352, 352))
        assert [tuple(o.shape) for o in outs] == [
            (1, 64, 88, 88), (1, 128, 44, 44), (1, 320, 22, 22), (1, 512, 11, 11)]
        assert 24.5e6 < n_params(m) < 25.2e6   # published PVTv2-B2: 25.4M with head

    def test_res2net_matches_the_published_parameter_count(self):
        """Pinned so that a silent architectural drift cannot pass the
        ImageNet checkpoint's key check by accident."""
        assert n_params(res2net50_v1b_26w_4s()) == pytest.approx(25.72e6, rel=0.002)

    def test_pvtv2_imports_without_timm(self):
        import polyptail.models.pvtv2 as mod
        assert "timm" not in {k.split(".")[0] for k in dir(mod)}


class TestPolypPVT:
    @pytest.mark.parametrize("size", [256, 352, 448])
    def test_outputs_match_the_input_grid_at_every_training_scale(self, size):
        m = PolypPVT(pretrained=None)
        p1, p2 = m(torch.randn(1, 3, size, size))
        assert p1.shape == p2.shape == (1, 1, size, size)

    def test_parameter_count_matches_the_paper(self):
        assert n_params(PolypPVT(pretrained=None)) == pytest.approx(25.11e6, rel=0.005)

    def test_gradients_reach_the_encoder(self):
        m = PolypPVT(pretrained=None)
        (sum(o.sum() for o in m(torch.randn(1, 3, 128, 128)))).backward()
        g = m.backbone.patch_embed1.proj.weight.grad
        assert g is not None and torch.isfinite(g).all()


class TestPraNet:
    def test_four_deep_supervision_heads_at_input_resolution(self):
        m = PraNet()
        outs = m(torch.randn(1, 3, 352, 352))
        assert len(outs) == 4
        assert all(o.shape == (1, 1, 352, 352) for o in outs)

    def test_parameter_count_matches_the_paper(self):
        assert n_params(PraNet()) == pytest.approx(32.55e6, rel=0.005)


class TestCheckpointLoading:
    """The reference loader filters mismatched keys silently, so pointing it at
    the wrong file trains from scratch and says nothing.  These pin the fix."""

    def test_wrong_checkpoint_raises_instead_of_loading_nothing(self, tmp_path):
        bad = tmp_path / "bad.pth"
        torch.save({"not.a.real.key": torch.zeros(3)}, bad)
        with pytest.raises(RuntimeError, match="missing"):
            load_pretrained_backbone(pvt_v2_b2(), bad, strict=True)

    def test_non_strict_warns_but_proceeds(self, tmp_path, caplog):
        bad = tmp_path / "bad.pth"
        torch.save({"not.a.real.key": torch.zeros(3)}, bad)
        rep = load_pretrained_backbone(pvt_v2_b2(), bad, strict=False)
        assert rep["missing"]

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pretrained_backbone(pvt_v2_b2(), tmp_path / "nope.pth")

    def test_a_correct_checkpoint_loads_cleanly(self, tmp_path):
        ref = pvt_v2_b2()
        good = tmp_path / "good.pth"
        torch.save(ref.state_dict(), good)
        target = pvt_v2_b2()
        rep = load_pretrained_backbone(target, good, strict=True)
        assert rep["missing"] == [] and rep["unexpected"] == []
        for k, v in ref.state_dict().items():
            assert torch.equal(target.state_dict()[k], v)

    def test_module_prefixes_are_stripped(self, tmp_path):
        ref = pvt_v2_b2()
        p = tmp_path / "ddp.pth"
        torch.save({f"module.{k}": v for k, v in ref.state_dict().items()}, p)
        assert load_pretrained_backbone(pvt_v2_b2(), p, strict=True)["missing"] == []

    def test_none_means_scratch_and_says_so(self, caplog):
        rep = load_pretrained_backbone(pvt_v2_b2(), None)
        assert rep["loaded"] == 0


class TestRegistry:
    def test_every_model_declares_its_evaluation_head(self):
        for name in available_models():
            assert MODELS[name].head in ("sum", "mean", "last")

    def test_tiny_unet_is_marked_unreportable(self):
        assert MODELS["tiny_unet"].reportable is False
        assert all(MODELS[n].reportable for n in ("polyp_pvt", "pranet"))

    def test_unknown_model_is_a_keyerror(self):
        with pytest.raises(KeyError):
            build_model("nope")
