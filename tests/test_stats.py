"""Paired intervals and the mechanical falsification verdict."""

import numpy as np
import pytest

from polyptail.stats.paired import (
    compare_runs, falsification_verdict, hierarchical_bootstrap, holm_adjust,
    paired_image_bootstrap, paired_seed_t,
)

ETIS = "TestDataset/ETIS-LaribPolypDB"
COLON = "TestDataset/CVC-ColonDB"
CVC300 = "TestDataset/CVC-300"
KVASIR = "TestDataset/Kvasir"


class TestIntervals:
    def test_seed_t_matches_the_textbook_formula(self):
        a, b = [0.70, 0.72, 0.71], [0.73, 0.76, 0.74]
        ci = paired_seed_t(a, b)
        d = np.array(b) - np.array(a)
        half = 4.302652729911275 * d.std(ddof=1) / np.sqrt(3)
        assert ci.mean == pytest.approx(d.mean())
        assert ci.lo == pytest.approx(d.mean() - half)
        assert ci.hi == pytest.approx(d.mean() + half)

    def test_identical_runs_give_a_degenerate_interval_at_zero(self):
        ci = paired_seed_t([0.8, 0.8, 0.8], [0.8, 0.8, 0.8])
        assert ci.mean == 0.0 and not ci.excludes_zero

    def test_three_seeds_are_wide_enough_to_swallow_a_one_point_gain(self):
        """The reason a 1-point improvement over 3 noisy seeds is not a result.

        The per-seed deltas here are +0.020 / -0.005 / +0.015: a mean gain of
        1.0 mDice with realistic run-to-run spread.  With 2 degrees of freedom
        the interval still straddles zero.
        """
        a = [0.780, 0.790, 0.771]
        b = [0.800, 0.785, 0.786]
        ci = paired_seed_t(a, b)
        assert ci.mean == pytest.approx(0.010, abs=1e-9)
        assert not ci.significantly_positive
        assert ci.lo < 0.0 < ci.hi

    def test_zero_spread_is_reported_as_degenerate_not_significant(self):
        """A zero-width interval formally excludes zero; three seeds agreeing
        to the last digit is a data problem, not evidence."""
        ci = paired_seed_t([0.80, 0.81, 0.79], [0.8001, 0.8101, 0.7901])
        assert ci.degenerate
        assert not ci.significantly_positive and not ci.excludes_zero

    def test_unpaired_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            paired_seed_t([0.1, 0.2], [0.1])

    def test_bootstrap_recovers_a_clear_shift(self):
        rng = np.random.default_rng(0)
        stems = [f"{i:03d}" for i in range(400)]
        a_runs = [{s: float(rng.normal(0.7, 0.1)) for s in stems} for _ in range(3)]
        b_runs = [{s: a[s] + 0.05 for s in stems} for a in a_runs]
        ci = paired_image_bootstrap(a_runs, b_runs, n_boot=2000)
        assert ci.mean == pytest.approx(0.05, abs=1e-9)
        assert ci.significantly_positive

    def test_hierarchical_is_wider_than_image_only_when_seeds_disagree(self):
        rng = np.random.default_rng(1)
        stems = [f"{i:03d}" for i in range(300)]
        a_runs, b_runs = [], []
        for k, offset in enumerate((-0.03, 0.0, 0.05)):   # strong seed effect
            a = {s: float(rng.normal(0.7, 0.05)) for s in stems}
            a_runs.append(a)
            b_runs.append({s: a[s] + offset for s in stems})
        img = paired_image_bootstrap(a_runs, b_runs, n_boot=3000, seed=2)
        hier = hierarchical_bootstrap(a_runs, b_runs, n_boot=3000, seed=2)
        assert (hier.hi - hier.lo) > (img.hi - img.lo)

    def test_bootstrap_requires_common_stems(self):
        with pytest.raises(ValueError):
            paired_image_bootstrap([{"a": 1.0}], [{"b": 1.0}], n_boot=10)


class TestHolm:
    def test_step_down_is_monotone_and_bounded(self):
        adj = holm_adjust({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
        assert adj["a"] <= adj["b"] <= adj["c"] <= adj["d"] <= 1.0
        assert adj["a"] == pytest.approx(0.004)

    def test_nans_pass_through(self):
        adj = holm_adjust({"a": float("nan"), "b": 0.01})
        assert np.isnan(adj["a"]) and adj["b"] == pytest.approx(0.01)


def arm(values: dict[str, list[float]], tail: dict[str, list[float]] | None = None) -> dict:
    return {ds: {"per_seed": v, "per_image": [], "aux": {"dice_le_05": (tail or {}).get(ds, [0.3] * len(v))}}
            for ds, v in values.items()}


def cmp_of(a_vals, b_vals):
    return compare_runs(arm(a_vals), arm(b_vals))


def shift(values: dict[str, list[float]], delta: float, jitter=(0.002, -0.001, 0.0015)):
    """Apply a mean shift with realistic per-seed spread.

    A *constant* shift gives the paired differences zero variance, which makes
    the t interval degenerate; see ``Interval.degenerate``.  Every fixture
    here therefore jitters, as real runs do.
    """
    return {k: [x + delta + jitter[i % len(jitter)] for i, x in enumerate(v)]
            for k, v in values.items()}


class TestVerdict:
    """Each rejection condition must fire exactly when the card says it does."""

    def test_c1_fires_when_etis_gains_neither_in_mean_nor_in_tail_mass(self):
        cm = cmp_of({ETIS: [0.79, 0.78, 0.80]}, {ETIS: [0.785, 0.781, 0.792]})
        tail = {ETIS: {"a": [0.30, 0.31, 0.29], "b": [0.30, 0.31, 0.29]}}
        v = falsification_verdict(cm, tail)
        assert v.rejected
        assert v.criteria["C1 no ETIS gain in mean or in tail mass"]["triggered"]

    def test_c1_does_not_fire_when_the_tail_mass_drops_enough(self):
        cm = cmp_of({ETIS: [0.79, 0.78, 0.80]}, {ETIS: [0.785, 0.781, 0.792]})
        tail = {ETIS: {"a": [0.30, 0.31, 0.29], "b": [0.24, 0.25, 0.24]}}
        v = falsification_verdict(cm, tail)
        assert not v.criteria["C1 no ETIS gain in mean or in tail mass"]["triggered"]

    def test_c1_does_not_fire_on_a_clear_mean_gain(self):
        cm = cmp_of({ETIS: [0.700, 0.702, 0.701]}, {ETIS: [0.760, 0.762, 0.761]})
        tail = {ETIS: {"a": [0.30] * 3, "b": [0.30] * 3}}
        assert not falsification_verdict(cm, tail).criteria[
            "C1 no ETIS gain in mean or in tail mass"]["triggered"]

    def test_c2_fires_when_a1_and_a2_are_indistinguishable_everywhere(self):
        base = {COLON: [0.80, 0.81, 0.79], CVC300: [0.90, 0.89, 0.91], ETIS: [0.78, 0.79, 0.77]}
        cm = cmp_of(base, base)
        a2 = cmp_of(base, shift(base, 0.0001, jitter=(0.0002, -0.0001, 0.00015)))
        v = falsification_verdict(cm, {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}}, a1_vs_a2=a2)
        assert v.criteria["C2 A1-A2 CI contains zero on every external set"]["triggered"]

    def test_c2_does_not_fire_when_a1_beats_a2_somewhere(self):
        base = {COLON: [0.80, 0.81, 0.79], CVC300: [0.90, 0.89, 0.91], ETIS: [0.78, 0.79, 0.77]}
        better = dict(base)
        better[ETIS] = [0.832, 0.838, 0.825]   # +0.052 / +0.048 / +0.055
        a2 = cmp_of(base, better)
        v = falsification_verdict(cmp_of(base, better),
                                  {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}}, a1_vs_a2=a2)
        assert not v.criteria["C2 A1-A2 CI contains zero on every external set"]["triggered"]

    def test_c3_fires_when_a_dataset_falls_while_the_pool_rises(self):
        a = {KVASIR: [0.917, 0.918, 0.916], ETIS: [0.787, 0.788, 0.786]}
        b = {KVASIR: [0.898, 0.903, 0.897], ETIS: [0.842, 0.839, 0.841]}
        v = falsification_verdict(cmp_of(a, b), {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}})
        c3 = v.criteria["C3 a dataset degrades while the pooled average rises"]
        assert c3["triggered"]
        assert c3["fallers"][0]["dataset"] == KVASIR

    def test_c3_ignores_a_small_or_noisy_decline(self):
        a = {KVASIR: [0.917, 0.918, 0.916], ETIS: [0.787, 0.788, 0.786]}
        b = {KVASIR: [0.914, 0.917, 0.912], ETIS: [0.842, 0.839, 0.841]}
        v = falsification_verdict(cmp_of(a, b), {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}})
        assert not v.criteria["C3 a dataset degrades while the pooled average rises"]["triggered"]

    def test_a_clean_win_is_not_rejected(self):
        a = {KVASIR: [0.917, 0.918, 0.916], COLON: [0.808, 0.807, 0.809],
             CVC300: [0.900, 0.901, 0.899], ETIS: [0.787, 0.786, 0.788]}
        b = shift(a, 0.03)
        # different jitter, or the A1-A2 differences would have zero spread
        a2 = cmp_of(shift(a, 0.01, jitter=(-0.001, 0.003, 0.0005)), b)
        tail = {k: {"a": [0.30] * 3, "b": [0.20] * 3} for k in a}
        v = falsification_verdict(cmp_of(a, b), tail, a1_vs_a2=a2)
        assert not v.rejected

    def test_missing_ablation_is_flagged_not_silently_passed(self):
        v = falsification_verdict(cmp_of({ETIS: [0.7] * 3}, {ETIS: [0.8] * 3}),
                                  {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}})
        assert any("C2 could not be evaluated" in n for n in v.notes)

    def test_summary_names_the_verdict(self):
        v = falsification_verdict(cmp_of({ETIS: [0.7] * 3}, {ETIS: [0.8] * 3}),
                                  {ETIS: {"a": [0.3] * 3, "b": [0.2] * 3}})
        assert "VERDICT" in v.summary()
