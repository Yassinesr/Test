"""Paired statistics across seeds and images, and the pre-registered verdict.

§6.2 items 6-8 of the brief make three things blocking: the baseline's own
per-dataset seed variance, paired 95% confidence intervals over >= 3 seeds per
dataset, and a per-dataset non-degradation check.  §1.3 W5 shows the
cross-paper noise floor alone is 0.3-0.8 Dice points, so an unqualified point
estimate is not evidence of anything.

Three intervals are produced, because they answer different questions and
disagreeing about which to quote is how 1-point "improvements" get published:

``seed_t``
    Paired t interval over seeds.  Accounts for run-to-run variance; with 3
    seeds it has 2 degrees of freedom and ``t* = 4.303``, so it is *wide*.
    This is the interval that licenses "method A beats method B", because
    seeds are the unit the claim is about.
``image_bootstrap``
    Bootstrap over images of the seed-averaged per-image difference.
    Accounts for which images are in the test set, conditional on this seed
    set.  Narrow, and it is the one most papers implicitly report.
``hierarchical``
    Resamples seeds *and* images.  Crude at 3 seeds but it is the only one of
    the three that is honest about both sources of variance; it is the
    headline number in ``tools/analyze.py``.

Nothing here corrects for multiplicity across the five datasets.  With five
datasets and two-sided 95% intervals the family-wise false-positive rate is
about 23%, so ``compare_runs`` also reports Holm-adjusted p-values; quote them
whenever the claim ranges over more than one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "Interval", "DatasetComparison", "paired_seed_t", "paired_image_bootstrap",
    "hierarchical_bootstrap", "holm_adjust", "compare_runs", "falsification_verdict",
]


@dataclass(frozen=True)
class Interval:
    mean: float
    lo: float
    hi: float
    method: str
    n: int = 0
    degenerate: bool = False
    """The paired differences had exactly zero spread, so the t interval
    collapses to a point.  A zero-width interval formally excludes zero, which
    would let an arbitrarily small difference read as significant off three
    seeds; a spread of exactly zero across continuous measurements almost
    always means the inputs are not what they appear to be (copied numbers, a
    re-used run directory, a metric that saturated).  Significance is
    therefore reported as False for a degenerate interval, and the flag is
    surfaced so the cause can be found rather than papered over."""

    @property
    def excludes_zero(self) -> bool:
        return (not self.degenerate) and ((self.lo > 0.0) or (self.hi < 0.0))

    @property
    def significantly_positive(self) -> bool:
        return (not self.degenerate) and self.lo > 0.0

    @property
    def significantly_negative(self) -> bool:
        return (not self.degenerate) and self.hi < 0.0

    def __str__(self) -> str:
        tag = " DEGENERATE" if self.degenerate else ""
        return f"{self.mean:+.4f} [{self.lo:+.4f}, {self.hi:+.4f}] ({self.method}, n={self.n}){tag}"


def paired_seed_t(a: Sequence[float], b: Sequence[float], conf: float = 0.95) -> Interval:
    """Paired t interval for ``mean(b - a)`` over matched seeds."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"unpaired seed arrays: {a_arr.shape} vs {b_arr.shape}")
    d = b_arr - a_arr
    n = d.size
    m = float(d.mean())
    if n < 2:
        return Interval(m, float("-inf"), float("inf"), "seed_t", n, degenerate=True)
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return Interval(m, m, m, "seed_t", n, degenerate=True)
    tstar = float(stats.t.ppf(0.5 + conf / 2.0, df=n - 1))
    half = tstar * sd / np.sqrt(n)
    return Interval(m, m - half, m + half, "seed_t", n)


def paired_seed_pvalue(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    d = b_arr - a_arr
    if d.size < 2 or float(d.std(ddof=1)) == 0.0:
        return float("nan")
    return float(stats.ttest_rel(b_arr, a_arr).pvalue)


def _matched(a_runs: Sequence[dict[str, float]], b_runs: Sequence[dict[str, float]]):
    """Per-image seed-averaged differences, on the stems both methods scored."""
    if len(a_runs) != len(b_runs):
        raise ValueError("compare needs the same number of seeds for both methods")
    stems = set(a_runs[0])
    for r in list(a_runs) + list(b_runs):
        stems &= set(r)
    stems_sorted = sorted(stems)
    if not stems_sorted:
        raise ValueError("no stems in common between the two methods")
    diffs = np.array([
        np.mean([b[s] - a[s] for a, b in zip(a_runs, b_runs)]) for s in stems_sorted
    ], dtype=float)
    per_seed = np.array([
        [b[s] - a[s] for s in stems_sorted] for a, b in zip(a_runs, b_runs)
    ], dtype=float)
    return stems_sorted, diffs, per_seed


def paired_image_bootstrap(
    a_runs: Sequence[dict[str, float]],
    b_runs: Sequence[dict[str, float]],
    n_boot: int = 10000,
    conf: float = 0.95,
    seed: int = 0,
) -> Interval:
    _, diffs, _ = _matched(a_runs, b_runs)
    rng = np.random.default_rng(seed)
    n = diffs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return Interval(float(diffs.mean()), float(lo), float(hi), "image_bootstrap", n)


def hierarchical_bootstrap(
    a_runs: Sequence[dict[str, float]],
    b_runs: Sequence[dict[str, float]],
    n_boot: int = 10000,
    conf: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Resample seeds with replacement, then images within the resampled seeds."""
    _, diffs, per_seed = _matched(a_runs, b_runs)
    rng = np.random.default_rng(seed)
    n_seeds, n_img = per_seed.shape
    boots = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        s_idx = rng.integers(0, n_seeds, size=n_seeds)
        i_idx = rng.integers(0, n_img, size=n_img)
        boots[k] = per_seed[np.ix_(s_idx, i_idx)].mean()
    lo, hi = np.percentile(boots, [(1 - conf) / 2 * 100, (1 + conf) / 2 * 100])
    return Interval(float(diffs.mean()), float(lo), float(hi), "hierarchical", n_img)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment.  NaNs pass through untouched."""
    named = [(k, v) for k, v in pvalues.items() if np.isfinite(v)]
    named.sort(key=lambda kv: kv[1])
    m = len(named)
    out: dict[str, float] = {k: v for k, v in pvalues.items() if not np.isfinite(v)}
    running = 0.0
    for i, (k, p) in enumerate(named):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


@dataclass
class DatasetComparison:
    dataset: str
    metric: str
    a_per_seed: list[float]
    b_per_seed: list[float]
    seed_t: Interval
    image_bootstrap: Optional[Interval]
    hierarchical: Optional[Interval]
    p_seed_t: float
    p_holm: float = float("nan")

    @property
    def a_mean(self) -> float:
        return float(np.mean(self.a_per_seed))

    @property
    def b_mean(self) -> float:
        return float(np.mean(self.b_per_seed))

    @property
    def a_sd(self) -> float:
        return float(np.std(self.a_per_seed, ddof=1)) if len(self.a_per_seed) > 1 else 0.0

    @property
    def b_sd(self) -> float:
        return float(np.std(self.b_per_seed, ddof=1)) if len(self.b_per_seed) > 1 else 0.0

    def to_json(self) -> dict:
        return {
            "dataset": self.dataset, "metric": self.metric,
            "a_per_seed": self.a_per_seed, "b_per_seed": self.b_per_seed,
            "a_mean": self.a_mean, "a_sd": self.a_sd,
            "b_mean": self.b_mean, "b_sd": self.b_sd,
            "delta": self.b_mean - self.a_mean,
            "seed_t": {"mean": self.seed_t.mean, "lo": self.seed_t.lo, "hi": self.seed_t.hi},
            "image_bootstrap": None if self.image_bootstrap is None else
                {"mean": self.image_bootstrap.mean, "lo": self.image_bootstrap.lo,
                 "hi": self.image_bootstrap.hi},
            "hierarchical": None if self.hierarchical is None else
                {"mean": self.hierarchical.mean, "lo": self.hierarchical.lo,
                 "hi": self.hierarchical.hi},
            "p_seed_t": self.p_seed_t, "p_holm": self.p_holm,
        }


def compare_runs(
    a: dict[str, dict],
    b: dict[str, dict],
    metric: str = "dice",
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, DatasetComparison]:
    """Compare two methods.

    ``a`` and ``b`` are ``{dataset: {"per_seed": [float, ...],
    "per_image": [ {stem: value}, ... ] }}``; ``per_image`` is optional and
    enables the bootstrap intervals.  ``tools/analyze.py`` assembles this from
    the per-run ``results.json`` / ``per_image_metrics.json`` files.
    """
    out: dict[str, DatasetComparison] = {}
    pvals: dict[str, float] = {}
    for ds in sorted(set(a) & set(b)):
        a_seeds, b_seeds = a[ds]["per_seed"], b[ds]["per_seed"]
        img_ci = hier_ci = None
        if a[ds].get("per_image") and b[ds].get("per_image"):
            img_ci = paired_image_bootstrap(a[ds]["per_image"], b[ds]["per_image"], n_boot, seed=seed)
            hier_ci = hierarchical_bootstrap(a[ds]["per_image"], b[ds]["per_image"], n_boot, seed=seed)
        p = paired_seed_pvalue(a_seeds, b_seeds)
        pvals[ds] = p
        out[ds] = DatasetComparison(
            dataset=ds, metric=metric, a_per_seed=list(map(float, a_seeds)),
            b_per_seed=list(map(float, b_seeds)),
            seed_t=paired_seed_t(a_seeds, b_seeds), image_bootstrap=img_ci,
            hierarchical=hier_ci, p_seed_t=p,
        )
    adjusted = holm_adjust(pvals)
    for ds, cmpn in out.items():
        cmpn.p_holm = adjusted.get(ds, float("nan"))
    return out


# --------------------------------------------------------------------------
# The pre-registered falsification criterion (candidate card §9)
# --------------------------------------------------------------------------

ETIS = "TestDataset/ETIS-LaribPolypDB"
EXTERNAL = (
    "TestDataset/CVC-ColonDB",
    "TestDataset/CVC-300",
    "TestDataset/ETIS-LaribPolypDB",
)


@dataclass
class Verdict:
    rejected: bool
    criteria: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["VERDICT: " + ("REJECT POT-TC" if self.rejected else "NOT REJECTED")]
        for name, blob in self.criteria.items():
            mark = "TRIGGERED" if blob.get("triggered") else "not triggered"
            lines.append(f"  [{mark:>13}] {name}: {blob.get('detail', '')}")
        lines += ["  note: " + n for n in self.notes]
        return "\n".join(lines)


def falsification_verdict(
    dice_cmp: dict[str, DatasetComparison],
    tail_frac: dict[str, dict[str, list[float]]],
    a1_vs_a2: Optional[dict[str, DatasetComparison]] = None,
    ci: str = "seed_t",
    tail_frac_drop_required: float = 0.03,
    degradation_threshold: float = 0.010,
) -> Verdict:
    """Evaluate the three rejection conditions of the candidate card, verbatim.

    ``dice_cmp`` compares A1 against A0 on mDice; ``tail_frac`` holds the
    per-seed fraction of images with Dice <= 0.5 as
    ``{dataset: {"a": [...], "b": [...]}}``; ``a1_vs_a2`` compares A1 against
    the empirical-quantile ablation.  ``ci`` selects which of the three
    intervals the conditions are read against -- ``seed_t`` is the default
    because seeds are the unit the claim is about.

    Thresholds are fixed here, not tuned: 3 absolute points of tail-fraction
    reduction, 1.0 mDice of per-dataset degradation.  Changing them after
    seeing results turns a pre-registered test into a post-hoc one.
    """
    v = Verdict(rejected=False)

    def interval(c: DatasetComparison) -> Interval:
        got = {"seed_t": c.seed_t, "image_bootstrap": c.image_bootstrap,
               "hierarchical": c.hierarchical}[ci]
        if got is None:
            raise ValueError(f"interval {ci!r} unavailable for {c.dataset}; per-image scores missing")
        return got

    # --- C1: no ETIS mDice gain AND no tail-mass reduction -----------------
    c1 = {"triggered": False, "detail": "ETIS comparison unavailable"}
    if ETIS in dice_cmp:
        it = interval(dice_cmp[ETIS])
        no_dice_gain = (it.mean <= 0.0) and (it.lo <= 0.0 <= it.hi)
        tf = tail_frac.get(ETIS)
        drop = float("nan")
        if tf:
            drop = float(np.mean(tf["a"]) - np.mean(tf["b"]))  # positive == fewer failures
        no_tail_gain = not (np.isfinite(drop) and drop >= tail_frac_drop_required)
        c1 = {
            "triggered": bool(no_dice_gain and no_tail_gain),
            "etis_dice_delta": it.mean, "etis_dice_ci": [it.lo, it.hi],
            "etis_dice_le_05_drop": drop, "required_drop": tail_frac_drop_required,
            "not_significantly_positive": not it.significantly_positive,
            "detail": (f"ETIS dDice {it.mean:+.4f} CI [{it.lo:+.4f},{it.hi:+.4f}]; "
                       f"Dice<=0.5 fraction drop {drop:+.4f} (need >= {tail_frac_drop_required:.3f})"),
        }
    v.criteria["C1 no ETIS gain in mean or in tail mass"] = c1

    # --- C2: GPD extrapolation adds nothing over the empirical quantile ----
    c2 = {"triggered": False, "detail": "A1-vs-A2 comparison not supplied (ablation A2 not run)"}
    if a1_vs_a2:
        present = [d for d in EXTERNAL if d in a1_vs_a2]
        if not present:
            c2 = {"triggered": False,
                  "detail": ("A2 was supplied but none of the external datasets "
                             f"{[d.split('/')[-1] for d in EXTERNAL]} are present; "
                             "C2 is undecidable on this split set")}
        else:
            per = {d: interval(a1_vs_a2[d]) for d in present}
            all_null = all(not i.excludes_zero for i in per.values())
            c2 = {
                "triggered": bool(all_null),
                "per_dataset": {d: {"mean": i.mean, "lo": i.lo, "hi": i.hi} for d, i in per.items()},
                "detail": "; ".join(f"{d.split('/')[-1]} {i.mean:+.4f}[{i.lo:+.4f},{i.hi:+.4f}]"
                                    for d, i in per.items()),
            }
    v.criteria["C2 A1-A2 CI contains zero on every external set"] = c2

    # --- C3: a dataset falls materially while the pooled average rises -----
    pooled_delta = float(np.mean([c.b_mean - c.a_mean for c in dice_cmp.values()])) if dice_cmp else 0.0
    fallers = []
    for ds, c in dice_cmp.items():
        it = interval(c)
        if it.mean < -degradation_threshold and it.significantly_negative:
            fallers.append({"dataset": ds, "delta": it.mean, "ci": [it.lo, it.hi]})
    c3 = {
        "triggered": bool(fallers and pooled_delta > 0.0),
        "pooled_delta": pooled_delta, "fallers": fallers,
        "detail": (f"pooled dDice {pooled_delta:+.4f}; "
                   + ("datasets falling >1.0 mDice with CI excluding zero: "
                      + ", ".join(f['dataset'].split('/')[-1] for f in fallers) if fallers
                      else "no dataset falls significantly")),
    }
    v.criteria["C3 a dataset degrades while the pooled average rises"] = c3

    degenerate = sorted(
        d for d, c in list(dice_cmp.items()) + list((a1_vs_a2 or {}).items())
        if interval(c).degenerate
    )
    if degenerate:
        v.notes.append(
            "zero-spread paired differences on " + ", ".join(sorted(set(degenerate)))
            + " -- the interval is degenerate and is treated as non-significant; "
              "check those runs are genuinely distinct before reading the verdict."
        )
    v.rejected = any(bool(b.get("triggered")) for b in v.criteria.values())
    if not a1_vs_a2 or not [d for d in EXTERNAL if d in (a1_vs_a2 or {})]:
        v.notes.append("C2 could not be evaluated: run ablation A2 on the external "
                       "datasets before reading this verdict as a pass.")
    if ETIS not in dice_cmp:
        v.notes.append("C1 could not be evaluated: ETIS-LaribPolypDB is absent from the results.")
    v.notes.append(f"intervals read at ci={ci!r}; 95% two-sided, unadjusted for the 5-dataset family.")
    return v
