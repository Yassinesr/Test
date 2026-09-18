"""The streaming tail objective: state, gradient routing, mode equivalences."""

import math

import pytest
import torch

from polyptail.losses.tail import DeficitBuffer, TailConfig, TailRiskLoss


def make(mode="gpd", total_steps=1000, **kw):
    defaults = dict(warmup_frac=0.0, buffer_size=256, min_buffer=32, min_exceedances=8)
    defaults.update(kw)
    return TailRiskLoss(TailConfig(mode=mode, **defaults), total_steps=total_steps)


def warm(module, gen, n_batches=40, batch=16, loc=0.3, scale=0.15):
    """Fill the buffer with plausible deficits so the threshold is defined."""
    for _ in range(n_batches):
        d = torch.clamp(torch.randn(batch, generator=gen) * scale + loc, 0.0, 1.0)
        module(d.requires_grad_(False))
    return module


class TestBuffer:
    def test_wraps_and_reports_length(self):
        b = DeficitBuffer(10)
        b.push(torch.arange(7.0))
        assert len(b) == 7
        b.push(torch.arange(7.0, 13.0))
        assert len(b) == 10
        assert set(b.values().tolist()) == set(range(3, 13))

    def test_oversized_push_keeps_the_most_recent(self):
        b = DeficitBuffer(4)
        b.push(torch.arange(10.0))
        assert sorted(b.values().tolist()) == [6.0, 7.0, 8.0, 9.0]

    def test_roundtrips_through_state_dict(self):
        b = DeficitBuffer(8)
        b.push(torch.rand(5))
        c = DeficitBuffer(8)
        c.load_state_dict(b.state_dict())
        assert torch.equal(b.values(), c.values())


class TestSchedule:
    def test_lambda_warms_up_linearly_then_saturates(self):
        m = TailRiskLoss(TailConfig(mode="gpd", lam=0.4, warmup_frac=0.5), total_steps=100)
        assert m.lam_at(0) == pytest.approx(0.0)
        assert m.lam_at(25) == pytest.approx(0.2)
        assert m.lam_at(50) == pytest.approx(0.4)
        assert m.lam_at(999) == pytest.approx(0.4)

    def test_mode_none_is_identically_zero(self):
        gen = torch.Generator().manual_seed(0)
        m = warm(make("none"), gen)
        d = torch.rand(16, generator=gen, requires_grad=True)
        loss, stats = m(d)
        assert float(loss) == 0.0
        assert stats["tail/lam"] == 0.0

    def test_inactive_before_the_buffer_fills(self):
        gen = torch.Generator().manual_seed(0)
        m = make("gpd")
        d = torch.rand(16, generator=gen, requires_grad=True)
        loss, stats = m(d)
        assert float(loss) == 0.0
        assert stats["tail/active"] == 0.0


class TestGradient:
    @pytest.mark.parametrize("mode", ["gpd", "cvar", "ohem"])
    def test_gradient_never_rewards_a_larger_deficit(self, mode):
        gen = torch.Generator().manual_seed(3)
        m = warm(make(mode), gen)
        any_positive = False
        for _ in range(20):
            d = torch.clamp(torch.randn(16, generator=gen) * 0.15 + 0.3, 0.0, 1.0).requires_grad_(True)
            loss, stats = m(d)
            if not stats.get("tail/active"):
                continue
            loss.backward()
            assert float(d.grad.min()) >= -1e-9, f"{mode}: negative gradient {float(d.grad.min())}"
            any_positive |= float(d.grad.max()) > 0.0
        assert any_positive, f"{mode} never produced a gradient at all"

    def test_the_empirical_quantile_ablation_updates_far_less_often(self):
        """A2's sparsity is a property of the estimator, not a bug.

        At alpha = 0.02 the empirical (1-alpha) tail is the top 13% of the
        exceedance pool, and a batch of 16 supplies ~2 live exceedances, so
        most steps contribute nothing. Extrapolating a fitted GPD instead uses
        every exceedance on every step.  That difference in effective sample
        size *is* the extreme-value argument, so A2 must be reported with its
        update count, and `cvar_selection="live"` run as a cross-check that A2
        is not merely losing on update count.
        """
        counts = {}
        for mode in ("gpd", "cvar"):
            gen = torch.Generator().manual_seed(47)
            m = warm(make(mode), gen)
            n_updates = 0
            for _ in range(60):
                d = torch.clamp(torch.randn(16, generator=gen) * 0.15 + 0.3, 0.0, 1.0).requires_grad_(True)
                loss, stats = m(d)
                if stats.get("tail/active") and float(loss.detach()) != 0.0:
                    loss.backward()
                    n_updates += int(float(d.grad.abs().max()) > 0)
            counts[mode] = n_updates
        assert counts["gpd"] > counts["cvar"]

    @pytest.mark.parametrize("mode", ["gpd", "cvar", "ohem"])
    def test_gradient_only_touches_images_above_the_threshold(self, mode):
        gen = torch.Generator().manual_seed(5)
        m = warm(make(mode), gen)
        d = torch.clamp(torch.randn(64, generator=gen) * 0.15 + 0.3, 0.0, 1.0).requires_grad_(True)
        loss, stats = m(d)
        assert stats["tail/active"] == 1.0
        loss.backward()
        below = d.detach() <= stats["tail/u"]   # the threshold this call used
        assert float(d.grad[below].abs().max()) == 0.0

    def test_the_live_batch_is_not_double_counted_in_the_pool(self):
        """The buffer half of the pool must exclude the batch being weighted,
        or the current batch double-votes on the fit that weights it."""
        gen = torch.Generator().manual_seed(53)
        m = warm(make("ohem"), gen, n_batches=20, batch=16)
        before = m.buffer.values().clone()          # snapshot: the buffer wraps
        d = torch.clamp(torch.randn(16, generator=gen) * 0.15 + 0.3, 0.0, 1.0).requires_grad_(True)
        _, stats = m(d)
        u = stats["tail/u"]
        expected_buf = int(((before - u) > 0).sum())
        assert stats["tail/n_pool"] == pytest.approx(stats["tail/n_live"] + expected_buf)
        # and the batch really did land in the buffer, for the *next* call
        assert not torch.equal(before, m.buffer.values())

    def test_loss_value_is_the_pooled_statistic_not_the_batch_one(self):
        gen = torch.Generator().manual_seed(7)
        m = warm(make("ohem"), gen)
        d = torch.clamp(torch.randn(16, generator=gen) * 0.15 + 0.3, 0.0, 1.0).requires_grad_(True)
        loss, stats = m(d)
        assert stats["tail/n_pool"] > stats["tail/n_live"]
        assert float(loss.detach()) == pytest.approx(stats["tail/loss"], rel=1e-6)

    def test_ohem_gradient_is_the_mean_over_live_exceedances(self):
        gen = torch.Generator().manual_seed(11)
        m = warm(make("ohem", lam=1.0), gen)
        d = torch.clamp(torch.randn(64, generator=gen) * 0.15 + 0.35, 0.0, 1.0).requires_grad_(True)
        loss, stats = m(d)
        loss.backward()
        n_live = int(stats["tail/n_live"])
        above = d.detach() > stats["tail/u"]
        assert float(d.grad[above].min()) == pytest.approx(1.0 / n_live, rel=1e-6)

    def test_rescale_makes_the_gradient_independent_of_buffer_size(self):
        totals = []
        for buf in (128, 512):
            gen = torch.Generator().manual_seed(13)
            m = make("ohem", lam=1.0, buffer_size=buf, min_buffer=32)
            warm(m, gen, n_batches=buf // 16 + 10)
            d = torch.clamp(torch.randn(64, generator=gen) * 0.15 + 0.35, 0.0, 1.0).requires_grad_(True)
            loss, _ = m(d)
            loss.backward()
            totals.append(float(d.grad.sum()))
        assert totals[0] == pytest.approx(totals[1], rel=1e-6)

    def test_cap_removes_gradient_above_the_cap(self):
        gen = torch.Generator().manual_seed(17)
        m = warm(make("gpd", exceedance_cap=0.05), gen)
        d = torch.clamp(torch.randn(64, generator=gen) * 0.15 + 0.3, 0.0, 1.0)
        d[0] = 1.0  # a mis-annotated image
        d = d.requires_grad_(True)
        loss, stats = m(d)
        if stats["tail/active"]:
            loss.backward()
            assert float(d.grad[0]) == 0.0


class TestModes:
    def test_normalised_weights_sum_to_one(self):
        gen = torch.Generator().manual_seed(19)
        for mode in ("gpd", "cvar", "ohem"):
            m = warm(make(mode, normalize="sum1", lam=1.0, rescale_live=False), gen)
            d = torch.clamp(torch.randn(256, generator=gen) * 0.15 + 0.35, 0.0, 1.0)
            loss, stats = m(d.requires_grad_(True))
            if not stats["tail/active"]:
                continue
            pool_max = max(float(d.max()) - stats["tail/u"], 0.0)
            assert 0.0 <= float(loss) <= pool_max + 1e-9

    def test_cvar_concentrates_on_fewer_images_than_ohem(self):
        gen = torch.Generator().manual_seed(23)
        counts = {}
        for mode in ("cvar", "ohem"):
            g2 = torch.Generator().manual_seed(23)
            m = warm(make(mode), g2)
            d = torch.clamp(torch.randn(512, generator=g2) * 0.15 + 0.35, 0.0, 1.0).requires_grad_(True)
            loss, stats = m(d)
            loss.backward()
            counts[mode] = int((d.grad > 0).sum())
        assert counts["cvar"] < counts["ohem"]

    def test_gpd_reports_a_finite_fit_and_return_level(self):
        gen = torch.Generator().manual_seed(29)
        m = warm(make("gpd"), gen)
        d = torch.clamp(torch.randn(128, generator=gen) * 0.15 + 0.35, 0.0, 1.0).requires_grad_(True)
        _, stats = m(d)
        assert math.isfinite(stats["tail/xi"])
        assert stats["tail/beta"] > 0
        assert stats["tail/q_hat"] > stats["tail/u"]


class TestSerialisation:
    def test_module_state_restores_buffer_threshold_and_step(self):
        gen = torch.Generator().manual_seed(31)
        m = warm(make("gpd"), gen)
        sd = m.state_dict()
        m2 = make("gpd")
        m2.load_state_dict(sd)
        assert m2.threshold == pytest.approx(m.threshold)
        assert len(m2.buffer) == len(m.buffer)
        assert int(m2._step) == int(m._step)


class TestValidation:
    def test_rejects_alpha_not_below_p(self):
        with pytest.raises(ValueError):
            TailConfig(mode="gpd", p=0.02, alpha=0.15)

    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError):
            TailConfig(mode="nope")

    def test_rejects_min_buffer_above_capacity(self):
        with pytest.raises(ValueError):
            TailConfig(buffer_size=32, min_buffer=64)
