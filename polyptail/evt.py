"""Closed-form peaks-over-threshold (POT) / generalized-Pareto machinery.

This module is deliberately **pure**: it contains no training state, no
buffers and no side effects, so every identity below is unit-testable in
isolation (see ``tests/test_evt.py``).  ``polyptail/losses/tail.py`` wraps it
with the streaming state (threshold tracker, deficit replay buffer) that the
training objective needs.

Background
----------
Pickands (1975) / Balkema-de Haan: for a broad class of distributions the
conditional law of the exceedances ``X - u | X > u`` converges, as ``u``
approaches the right endpoint, to a generalized Pareto distribution (GPD)

    F(e) = 1 - (1 + xi * e / beta) ** (-1 / xi),     xi != 0,
    F(e) = 1 - exp(-e / beta),                       xi == 0.

If ``u`` is the ``(1 - p)`` quantile of ``X`` (so ``P[X > u] = p``) then the
``(1 - alpha)`` quantile of ``X`` itself, for ``alpha < p``, is the POT
*return level*

    q_{1-alpha} = u + (beta / xi) * ((p / alpha) ** xi - 1),        xi != 0,
    q_{1-alpha} = u + beta * log(p / alpha),                        xi == 0.

Writing ``r = p / alpha`` and ``L = log r``, the bracket factorizes as

    q_{1-alpha} = u + beta * g(xi),      g(xi) = (r ** xi - 1) / xi,

with the removable singularity ``g(0) = L``.  ``g`` and its derivative are
implemented below with series expansions so they are exact and finite at
``xi = 0`` (a naive ``(r**xi - 1)/xi`` is ``0/0`` there and catastrophically
cancels nearby).

Probability-weighted-moment (PWM) estimator
-------------------------------------------
Hosking & Wallis (1987).  With ``a_s = E[X (1 - F(X)) ** s]`` one has, for a
GPD with ``xi < 1``,

    a_s = beta / ((s + 1) * (s + 1 - xi)),

so that ``a_0 = beta / (1 - xi)`` and ``a_1 = beta / (2 * (2 - xi))`` and the
estimator inverts in closed form:

    D    = a_0 - 2 * a_1        ( = beta / ((1 - xi) * (2 - xi)) > 0 for xi < 1 )
    xi   = 2 - a_0 / D
    beta = 2 * a_0 * a_1 / D

The sample versions are *linear* in the ascending order statistics
``e_(1) <= ... <= e_(n)``:

    a_0 = (1 / n) * sum_i e_(i)
    a_1 = (1 / n) * sum_i t_i * e_(i),      t_i = (n - i) / (n - 1)

which is exactly what makes the plug-in return level differentiable in the
exceedances with *closed-form* weights -- no autograd through a sort, no
iterative MLE, no implicit-function theorem.

Gradient weights
----------------
Differentiating the plug-in ``q = u + beta(a_0, a_1) * g(xi(a_0, a_1))``:

    d beta / d e_(i) = (2 / (n * D**2)) * (a_0**2 * t_i - 2 * a_1**2)
    d xi   / d e_(i) = (2 / (n * D**2)) * (a_1 - a_0 * t_i)
    d q    / d e_(i) = g(xi) * d beta / d e_(i)
                     + beta * g'(xi) * d xi / d e_(i)

``pot_return_level_weights`` returns that vector.  Two facts about it drive
the design of the loss:

1.  With ``xi`` held fixed (``xi_mode="detached"``) the weight on the
    *largest* exceedance is ``-4 * a_1**2 / (n * D**2) < 0``.  Minimising the
    scale estimate alone therefore rewards making the worst image *worse*,
    because inflating ``e_(n)`` inflates ``D`` faster than the numerator.
    That is estimator gaming, not tail reduction.
2.  Letting ``xi`` carry gradient too (``xi_mode="live"``, the default)
    cancels that pathology over almost all of the admissible shape range,
    because a fatter observed tail raises ``xi`` and hence the return level.

``weight_floor`` (default ``0.0``) clips whatever negative weight survives,
so the objective can never pay the network to increase a per-image deficit.
``tests/test_evt.py`` asserts both behaviours numerically.

3.  With ``xi`` live and *unclamped*, the weights are monotone increasing in
    exceedance rank whenever ``xi < 0`` (chase the worst frames, which is the
    regime a bounded score sits in) and monotone decreasing whenever
    ``xi > 0`` (discount the extremes, because a heavy fitted tail means the
    biggest lever on the return level is the body of the exceedances, not the
    single worst image -- a built-in robustness to the label noise the card
    lists as failure mode #1).  That adaptivity is the entire difference
    between POT-TC and ablation A3, which weights uniformly always.  It is
    also why the shape clamp must not bind: see ``gpd_pwm_fit``.

References
----------
Pickands, J. (1975). Statistical inference using extreme order statistics.
    Ann. Statist. 3(1):119-131. doi:10.1214/aos/1176343003
Hosking, J. R. M. & Wallis, J. R. (1987). Parameter and quantile estimation
    for the generalized Pareto distribution. Technometrics 29(3):339-349.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "GPDFit",
    "gpd_g",
    "gpd_g_prime",
    "pwm_moments",
    "gpd_pwm_fit",
    "gpd_return_level",
    "pot_return_level_weights",
]

# Below this |x| the series expansions are used instead of the closed forms.
#
# The two branches fail at opposite ends.  The closed form of ``_h`` computes
# ``((x-1)e^x + 1)`` -- a difference of O(1) quantities whose true value is
# O(x^2/2) -- so its relative error grows like ``eps / x^2`` and is ~4e-10 at
# ``x = 1e-3`` and useless by ``x = 1e-6``.  The truncated series errs the
# other way, like ``x^5/720``.  Setting the two equal puts the crossover near
# ``x = 0.016``; ``1e-2`` sits just inside it, where the series is good to
# ~1.4e-13 and the closed form to ~4e-12.  ``tests/test_evt.py`` pins both
# directions, including that the closed form really does fail below it.
_SERIES_CUTOFF = 1e-2


def _expm1_over_x(x: Tensor) -> Tensor:
    """``(exp(x) - 1) / x`` with the removable singularity at 0 handled.

    Series: ``sum_{k>=1} x^(k-1)/k!`` truncated after ``x**5``, which is
    exact to ~1e-15 relative at the ``1e-3`` cutoff and still ~4e-12 at
    ``x = 0.02`` -- comfortably inside float64 and far inside float32.
    """
    small = x.abs() < _SERIES_CUTOFF
    safe = torch.where(small, torch.ones_like(x), x)
    closed = torch.expm1(safe) / safe
    series = 1.0 + x * (0.5 + x * (1.0 / 6.0 + x * (1.0 / 24.0
                 + x * (1.0 / 120.0 + x * (1.0 / 720.0)))))
    return torch.where(small, series, closed)


def _h(x: Tensor) -> Tensor:
    """``((x - 1) * exp(x) + 1) / x**2`` with the singularity at 0 handled.

    This is ``d/dx [(exp(x) - 1) / x]``.  The coefficient of ``x**n`` in
    ``(x - 1) exp(x) + 1`` is ``(n - 1)/n!``, giving
    ``1/2 + x/3 + x**2/8 + x**3/30 + x**4/144``.
    """
    small = x.abs() < _SERIES_CUTOFF
    safe = torch.where(small, torch.ones_like(x), x)
    closed = ((safe - 1.0) * torch.exp(safe) + 1.0) / (safe * safe)
    series = 0.5 + x * (1.0 / 3.0 + x * (0.125 + x * (1.0 / 30.0 + x * (1.0 / 144.0))))
    return torch.where(small, series, closed)


def gpd_g(xi: Tensor, log_ratio: float) -> Tensor:
    """``g(xi) = (r ** xi - 1) / xi`` where ``log_ratio = log(r)``.

    ``g`` is the factor that turns the GPD scale into a return level:
    ``q = u + beta * g(xi)``.  It is positive and increasing in ``xi`` for
    ``r > 1``, with ``g(0) = log(r)``.
    """
    x = xi * log_ratio
    return log_ratio * _expm1_over_x(x)


def gpd_g_prime(xi: Tensor, log_ratio: float) -> Tensor:
    """``dg/dxi``.  Equals ``log(r) ** 2 / 2`` at ``xi = 0``."""
    x = xi * log_ratio
    return (log_ratio * log_ratio) * _h(x)


def pwm_moments(exceedances_sorted: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """First two probability-weighted moments of ascending-sorted exceedances.

    Returns ``(a0, a1, t)`` where ``t_i = (n - i) / (n - 1)`` for ``i = 1..n``
    are the PWM plotting weights (``t_1 = 1`` on the smallest exceedance,
    ``t_n = 0`` on the largest).  ``t`` is returned because the gradient
    weights need it.

    Requires ``n >= 2``.
    """
    n = exceedances_sorted.numel()
    if n < 2:
        raise ValueError(f"PWM needs at least 2 exceedances, got {n}")
    idx = torch.arange(1, n + 1, device=exceedances_sorted.device, dtype=exceedances_sorted.dtype)
    t = (n - idx) / (n - 1)
    a0 = exceedances_sorted.mean()
    a1 = (t * exceedances_sorted).mean()
    return a0, a1, t


@dataclass(frozen=True)
class GPDFit:
    """Result of a PWM fit.  All fields are detached scalars (as floats)."""

    xi: float
    beta: float
    xi_raw: float
    """``xi`` before clamping -- log this, it is the instability canary."""
    a0: float
    a1: float
    n: int
    degenerate: bool
    """True when the PWM denominator collapsed and the exponential
    (``xi = 0``, ``beta = a0``) fallback was used."""
    clamped: bool = False
    """True when ``xi_raw`` fell outside ``[xi_min, xi_max]``.  A run in which
    this is often true is a run whose tail-weight profile is not the one the
    method describes -- widen the bounds or report the diagnostic."""


def gpd_pwm_fit(
    exceedances: Tensor,
    xi_min: float = -1.5,
    xi_max: float = 0.7,
    eps: float = 1e-8,
) -> GPDFit:
    """Fit ``(xi, beta)`` by probability-weighted moments.

    ``xi`` is clamped to ``[xi_min, xi_max]``.  The two bounds are not
    symmetric in either motivation or consequence.

    *Upper* (``0.7``, from the candidate card): the GPD has no finite mean for
    ``xi >= 1`` and no finite variance for ``xi >= 1/2``, and small-sample PWM
    shape estimates are notoriously high-variance.  A real constraint.

    *Lower*: the card proposes ``-0.5``, but that number inherits an
    upper-tail moment argument that does not apply downward -- ``xi < 0``
    simply means the tail has a finite right endpoint, which is *exactly* the
    regime a bounded score lives in.  The POT-TC deficit is ``1 - softDice``,
    hard-bounded at 1, so fitted shapes of ``-0.4`` to ``-0.8`` are the normal
    case, not an anomaly.  Clamping at ``-0.5`` therefore binds almost every
    step, and because a bound that binds has zero local sensitivity the shape
    gradient switches off exactly when it binds -- which **inverts** the
    weight profile: measured on a GPD sample with true ``xi = -0.7``, the
    clamped fit puts 14.6x weight on the *smallest* exceedances and zero on
    the largest fifth, while the unclamped fit is correctly monotone in
    exceedance rank.  The default here is ``-1.5``, comfortably below the
    operating range, and ``GPDFit.clamped`` records whenever either bound
    binds so a run can be audited instead of trusted.

    Falls back to the exponential fit (``xi = 0``, ``beta = a0``) when the
    denominator ``D = a0 - 2*a1`` is non-positive, which cannot happen in
    expectation for ``xi < 1`` but does happen in small samples.
    """
    e = exceedances.detach().flatten()
    e, _ = torch.sort(e)
    a0, a1, _ = pwm_moments(e)
    a0f, a1f = float(a0), float(a1)
    n = e.numel()

    denom = a0f - 2.0 * a1f
    if not math.isfinite(denom) or a0f <= eps or denom <= eps * max(a0f, 1.0):
        return GPDFit(
            xi=0.0, beta=max(a0f, 0.0), xi_raw=float("nan"),
            a0=a0f, a1=a1f, n=n, degenerate=True, clamped=False,
        )

    xi_raw = 2.0 - a0f / denom
    beta = 2.0 * a0f * a1f / denom
    xi = min(max(xi_raw, xi_min), xi_max)
    if not math.isfinite(beta) or beta <= 0.0:
        return GPDFit(
            xi=0.0, beta=max(a0f, 0.0), xi_raw=xi_raw,
            a0=a0f, a1=a1f, n=n, degenerate=True, clamped=False,
        )
    return GPDFit(xi=xi, beta=beta, xi_raw=xi_raw, a0=a0f, a1=a1f, n=n,
                  degenerate=False, clamped=(xi != xi_raw))


def gpd_return_level(u: float, fit: GPDFit, p: float, alpha: float) -> float:
    """POT return level ``q_{1-alpha}`` given the exceedance rate ``p``.

    ``u`` is the ``(1 - p)`` quantile of the parent distribution, so this is a
    quantile of the *deficit* distribution, not of the exceedances.
    Requires ``0 < alpha < p <= 1``.
    """
    if not (0.0 < alpha < p <= 1.0):
        raise ValueError(f"need 0 < alpha < p <= 1, got alpha={alpha}, p={p}")
    log_ratio = math.log(p / alpha)
    xi = torch.tensor(fit.xi, dtype=torch.float64)
    g = float(gpd_g(xi, log_ratio))
    return u + fit.beta * g


def pot_return_level_weights(
    exceedances_sorted: Tensor,
    fit: GPDFit,
    p: float,
    alpha: float,
    xi_mode: str = "live",
    weight_floor: Optional[float] = 0.0,
    xi_min: float = -1.5,
    xi_max: float = 0.7,
) -> Tensor:
    """``d q_{1-alpha} / d e_(i)`` for each ascending-sorted exceedance.

    Parameters
    ----------
    exceedances_sorted
        Ascending-sorted exceedances, detached.  Only its length and the
        derived ``t_i`` enter; the values enter through ``fit``.
    fit
        The PWM fit on the *same* sample.
    xi_mode
        ``"live"`` -- both the scale and shape paths contribute (default, and
        the only mode whose weights are non-negative across the admissible
        shape range; see module docstring).
        ``"detached"`` -- only the scale path.  Kept because it is the literal
        reading of "treat ``(xi, beta)`` as detached plug-ins", and because
        contrasting the two isolates how much of the effect is tail *shape*.
    weight_floor
        Clip weights from below at this value (``None`` disables).  ``0.0``
        guarantees the objective never rewards increasing a deficit.

    Returns a vector of the same length as ``exceedances_sorted``, detached.
    """
    if xi_mode not in ("live", "detached"):
        raise ValueError(f"xi_mode must be 'live' or 'detached', got {xi_mode!r}")
    if not (0.0 < alpha < p <= 1.0):
        raise ValueError(f"need 0 < alpha < p <= 1, got alpha={alpha}, p={p}")

    e = exceedances_sorted.detach()
    n = e.numel()
    device, dtype = e.device, e.dtype
    log_ratio = math.log(p / alpha)

    idx = torch.arange(1, n + 1, device=device, dtype=dtype)
    t = (n - idx) / (n - 1)

    xi_t = torch.tensor(fit.xi, device=device, dtype=dtype)
    g = gpd_g(xi_t, log_ratio)

    if fit.degenerate:
        # Exponential fallback: beta = a0 = mean(e), q = u + a0 * log(r).
        # d q / d e_(i) = log(r) / n -- uniform and strictly positive.
        w = torch.full((n,), float(log_ratio) / n, device=device, dtype=dtype)
    else:
        a0 = torch.tensor(fit.a0, device=device, dtype=dtype)
        a1 = torch.tensor(fit.a1, device=device, dtype=dtype)
        D = a0 - 2.0 * a1
        pref = 2.0 / (n * D * D)
        dbeta = pref * (a0 * a0 * t - 2.0 * a1 * a1)
        w = g * dbeta
        if xi_mode == "live":
            # A clamped shape has zero local sensitivity, exactly as
            # ``torch.clamp`` would propagate.  See ``gpd_pwm_fit`` for why a
            # bound that binds often is a bug in the bound, not a safeguard.
            if not fit.clamped and math.isfinite(fit.xi_raw):
                gp = gpd_g_prime(xi_t, log_ratio)
                beta = torch.tensor(fit.beta, device=device, dtype=dtype)
                dxi = pref * (a1 - a0 * t)
                w = w + beta * gp * dxi

    if weight_floor is not None:
        w = w.clamp_min(weight_floor)
    return w
