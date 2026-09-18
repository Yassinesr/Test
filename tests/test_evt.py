"""The extreme-value core.  If these fail, nothing downstream means anything."""

import math

import pytest
import torch

from polyptail.evt import (
    gpd_g, gpd_g_prime, gpd_pwm_fit, gpd_return_level, pot_return_level_weights, pwm_moments,
)

P, ALPHA = 0.15, 0.02
LOG_RATIO = math.log(P / ALPHA)


def sample_gpd(n, xi, beta, gen):
    u = torch.rand(n, generator=gen, dtype=torch.float64)
    if abs(xi) < 1e-12:
        return -beta * torch.log1p(-u)
    return beta / xi * ((1 - u) ** (-xi) - 1)


class TestRemovableSingularities:
    def test_g_at_zero_is_log_ratio(self):
        assert float(gpd_g(torch.tensor(0.0, dtype=torch.float64), LOG_RATIO)) == pytest.approx(LOG_RATIO, abs=1e-12)

    def test_g_prime_at_zero_is_half_log_ratio_squared(self):
        got = float(gpd_g_prime(torch.tensor(0.0, dtype=torch.float64), LOG_RATIO))
        assert got == pytest.approx(LOG_RATIO ** 2 / 2, abs=1e-12)

    @pytest.mark.parametrize("xi", [-0.9, -0.4, -1e-4, 0.0, 1e-4, 0.3, 0.7])
    def test_g_prime_matches_finite_difference(self, xi):
        t = torch.tensor(xi, dtype=torch.float64)
        h = 1e-6
        fd = (float(gpd_g(t + h, LOG_RATIO)) - float(gpd_g(t - h, LOG_RATIO))) / (2 * h)
        assert float(gpd_g_prime(t, LOG_RATIO)) == pytest.approx(fd, rel=1e-6)

    @pytest.mark.parametrize("xi", [1e-3, 5e-3, 1e-2, 2e-2])
    def test_branches_agree_where_both_are_valid(self, xi, monkeypatch):
        """Near the cutoff the two code paths must be interchangeable.

        Compared at one xi with the branch forced either way -- comparing two
        different xi would just be measuring dg/dxi.
        """
        import polyptail.evt as evt

        t = torch.tensor(xi, dtype=torch.float64)
        monkeypatch.setattr(evt, "_SERIES_CUTOFF", 0.0)          # force closed form
        closed_g, closed_gp = float(gpd_g(t, LOG_RATIO)), float(gpd_g_prime(t, LOG_RATIO))
        monkeypatch.setattr(evt, "_SERIES_CUTOFF", 1.0)          # force series
        series_g, series_gp = float(gpd_g(t, LOG_RATIO)), float(gpd_g_prime(t, LOG_RATIO))
        assert series_g == pytest.approx(closed_g, rel=1e-11)
        assert series_gp == pytest.approx(closed_gp, rel=1e-8)

    @pytest.mark.parametrize("xi,worst_rel", [(1e-9, 0.5), (1e-7, 1e-4), (1e-5, 1e-7)])
    def test_the_closed_form_really_does_fail_below_the_cutoff(self, xi, worst_rel, monkeypatch):
        """Why the series branch exists: catastrophic cancellation in ``_h``.

        ``((x-1)e^x + 1)`` is a difference of O(1) quantities whose true value
        is O(x^2/2), so the closed form loses ~``eps/x^2`` relative accuracy --
        at ``xi = 1e-9`` it returns exactly zero.  The shipped code uses the
        series there and stays within O(xi) of the analytic limit.
        """
        import polyptail.evt as evt

        t = torch.tensor(xi, dtype=torch.float64)
        exact_gp = LOG_RATIO ** 2 / 2          # the analytic limit as xi -> 0
        # g'(xi) departs from g'(0) at first order, so allow 2*xi on top.
        assert float(gpd_g_prime(t, LOG_RATIO)) == pytest.approx(exact_gp, rel=1e-8 + 2 * xi)
        monkeypatch.setattr(evt, "_SERIES_CUTOFF", 0.0)
        forced_closed = float(gpd_g_prime(t, LOG_RATIO))
        assert abs(forced_closed - exact_gp) > worst_rel * exact_gp

class TestPWM:
    def test_moments_are_linear_in_order_statistics(self):
        e = torch.sort(torch.rand(50, dtype=torch.float64)).values
        a0, a1, t = pwm_moments(e)
        assert float(a0) == pytest.approx(float(e.mean()))
        assert float(a1) == pytest.approx(float((t * e).mean()))
        assert float(t[0]) == pytest.approx(1.0)
        assert float(t[-1]) == pytest.approx(0.0)

    def test_needs_at_least_two_points(self):
        with pytest.raises(ValueError):
            pwm_moments(torch.tensor([1.0]))

    @pytest.mark.parametrize("xi,beta", [(0.0, 1.0), (0.3, 2.0), (-0.3, 1.5), (0.5, 0.7), (-0.7, 1.0)])
    def test_recovers_known_parameters(self, xi, beta):
        gen = torch.Generator().manual_seed(17)
        xis, betas = [], []
        for _ in range(30):
            f = gpd_pwm_fit(sample_gpd(20000, xi, beta, gen), xi_min=-5, xi_max=5)
            xis.append(f.xi_raw)
            betas.append(f.beta)
        assert sum(xis) / len(xis) == pytest.approx(xi, abs=0.02)
        assert sum(betas) / len(betas) == pytest.approx(beta, rel=0.02)

    def test_degenerate_fallback_on_constant_input(self):
        f = gpd_pwm_fit(torch.zeros(20, dtype=torch.float64))
        assert f.degenerate and f.xi == 0.0

    def test_clamp_is_recorded(self):
        gen = torch.Generator().manual_seed(1)
        e = sample_gpd(5000, -1.0, 1.0, gen)
        assert gpd_pwm_fit(e, xi_min=-0.5, xi_max=0.7).clamped
        assert not gpd_pwm_fit(e, xi_min=-3.0, xi_max=0.7).clamped

    def test_return_level_matches_the_analytic_quantile(self):
        gen = torch.Generator().manual_seed(5)
        for xi, beta in [(0.0, 1.0), (0.3, 2.0), (-0.3, 1.5)]:
            u = 0.5
            exact = u + (beta * LOG_RATIO if xi == 0 else beta / xi * ((P / ALPHA) ** xi - 1))
            f = gpd_pwm_fit(sample_gpd(200000, xi, beta, gen), xi_min=-5, xi_max=5)
            assert gpd_return_level(u, f, P, ALPHA) == pytest.approx(exact, rel=0.02)

    def test_return_level_rejects_alpha_above_p(self):
        f = gpd_pwm_fit(torch.rand(20, dtype=torch.float64) + 0.1)
        with pytest.raises(ValueError):
            gpd_return_level(0.5, f, p=0.02, alpha=0.15)


class TestWeights:
    """The weights must be the true derivative, and must never reward failure."""

    def _weights(self, e, xi_mode, floor=None, xi_min=-3.0):
        f = gpd_pwm_fit(e, xi_min=xi_min, xi_max=0.7)
        return f, pot_return_level_weights(
            e, f, P, ALPHA, xi_mode=xi_mode, weight_floor=floor, xi_min=xi_min, xi_max=0.7
        )

    def test_live_weights_equal_the_finite_difference_of_the_return_level(self):
        gen = torch.Generator().manual_seed(11)
        e = torch.sort(sample_gpd(120, 0.2, 1.0, gen)).values
        f, w = self._weights(e, "live")
        for i in (0, 30, 60, 119):
            h = 1e-7
            up, dn = e.clone(), e.clone()
            up[i] += h
            dn[i] -= h
            q_up = gpd_return_level(0.0, gpd_pwm_fit(up, xi_min=-3, xi_max=0.7), P, ALPHA)
            q_dn = gpd_return_level(0.0, gpd_pwm_fit(dn, xi_min=-3, xi_max=0.7), P, ALPHA)
            assert float(w[i]) == pytest.approx((q_up - q_dn) / (2 * h), rel=1e-4)

    @pytest.mark.parametrize("xi_true", [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    def test_detached_shape_rewards_making_the_worst_image_worse(self, xi_true):
        """Documents the pathology that motivates xi_mode='live' being default."""
        gen = torch.Generator().manual_seed(23)
        e = torch.sort(sample_gpd(3000, xi_true, 1.0, gen)).values
        _, w = self._weights(e, "detached")
        assert float(w[-1]) < 0.0

    @pytest.mark.parametrize("xi_true", [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    def test_live_shape_does_not(self, xi_true):
        gen = torch.Generator().manual_seed(23)
        e = torch.sort(sample_gpd(3000, xi_true, 1.0, gen)).values
        _, w = self._weights(e, "live")
        assert float(w[-1]) > 0.0

    def test_weight_floor_forbids_negative_weights_in_every_mode(self):
        gen = torch.Generator().manual_seed(29)
        for xi_true in (-0.8, -0.4, 0.0, 0.4):
            for mode in ("live", "detached"):
                e = torch.sort(sample_gpd(500, xi_true, 1.0, gen)).values
                _, w = self._weights(e, mode, floor=0.0)
                assert float(w.min()) >= 0.0

    def test_negative_shape_weights_increase_with_exceedance_rank(self):
        """Bounded tail -> chase the worst frames."""
        gen = torch.Generator().manual_seed(31)
        e = torch.sort(sample_gpd(2000, -0.5, 1.0, gen)).values
        _, w = self._weights(e, "live")
        assert bool((w[1:] >= w[:-1] - 1e-12).all())
        assert float(w[-1]) > float(w[0])

    def test_positive_shape_weights_decrease_with_exceedance_rank(self):
        """Heavy tail -> discount the extremes (built-in label-noise robustness)."""
        gen = torch.Generator().manual_seed(37)
        e = torch.sort(sample_gpd(2000, 0.4, 1.0, gen)).values
        _, w = self._weights(e, "live")
        assert bool((w[1:] <= w[:-1] + 1e-12).all())
        assert float(w[-1]) < float(w[0])

    def test_a_binding_lower_clamp_inverts_the_profile(self):
        """The regression this project actually hit: a clamp at -0.5 zeroes the
        shape gradient for a bounded score and flips which images are chased."""
        gen = torch.Generator().manual_seed(41)
        e = torch.sort(sample_gpd(2000, -0.7, 1.0, gen)).values
        _, w_bad = self._weights(e, "live", floor=0.0, xi_min=-0.5)
        _, w_ok = self._weights(e, "live", floor=0.0, xi_min=-3.0)
        assert float(w_bad[-1]) == 0.0           # worst images get no gradient
        assert float(w_bad[0]) > float(w_bad[-1])
        assert float(w_ok[-1]) > float(w_ok[0])  # correct ordering when unclamped

    def test_degenerate_fit_gives_uniform_positive_weights(self):
        e = torch.zeros(20, dtype=torch.float64)
        f = gpd_pwm_fit(e)
        w = pot_return_level_weights(e, f, P, ALPHA)
        assert f.degenerate
        assert bool((w > 0).all())
        assert float(w.max() - w.min()) == pytest.approx(0.0, abs=1e-15)
