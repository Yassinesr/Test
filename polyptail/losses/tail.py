"""POT-TC: the tail-calibrated training objective (Candidate 1).

Design
------
All three tail modes share one pipeline, and differ **only** in the weight
vector they place on a common exceedance set:

    1. every image contributes a scalar *deficit* ``d_i = 1 - softDice_i``;
    2. a sliding replay buffer of recent detached deficits defines a
       threshold ``u`` at the ``(1 - p)`` empirical quantile, EMA-smoothed;
    3. the *pooled* exceedance set is the live batch exceedances (which carry
       gradient) concatenated with the buffer exceedances (which do not) --
       pooling is what makes the tail statistics estimable at batch size 16,
       where a single step yields only ``p * B ~ 2`` exceedances;
    4. a weight vector ``w`` over the pooled, ascending-sorted exceedances;
    5. ``L_tail = sum_i w_i * e_i``.

    mode="gpd"   w = d q_{1-alpha} / d e_(i) from a closed-form PWM fit of a
                 generalized Pareto to the pooled exceedances.  This is A1.
    mode="cvar"  w = uniform on the top ``alpha / p`` fraction of the pooled
                 exceedances, i.e. the *empirical* (1-alpha) tail average.
                 This is ablation A2: same threshold, same pool, same
                 schedule -- only the extrapolation is removed.  If A1 and A2
                 agree, the EVT content is doing nothing.
    mode="ohem"  w = uniform on *all* pooled exceedances, i.e. the mean over
                 the worst ``p`` fraction.  This is ablation A3, which
                 separates "focus on hard images" from "shape the tail".
    mode="none"  disabled; the module returns 0 and only logs diagnostics.

Because the three modes are identical apart from ``w``, the A1/A2/A3 contrast
is a clean test of the *shape* of the tail weighting rather than a comparison
of three differently-plumbed objectives.

Gradient routing
----------------
The reported loss *value* is the pooled weighted mean (low variance, uses all
the buffer information).  The *gradient* flows only through the live batch
exceedances, via a straight-through construction:

    L = <w, e_pool_detached> + scale * sum_{j live} w_{rank(j)} * (e_j - e_j.detach())

so ``L.item()`` is the pooled statistic while ``dL/dtheta`` is the
closed-form weighted sum over the current batch.  ``scale = n_pool / n_live``
(``rescale_live=True``, default) keeps the gradient magnitude independent of
the buffer size, so ``lambda`` does not have to be retuned when ``K`` changes.

Nothing here touches inference.  The model, its forward pass, its parameter
count and its test-time behaviour are all unchanged; this is a loss term.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..evt import GPDFit, gpd_pwm_fit, gpd_return_level, pot_return_level_weights

__all__ = ["TailConfig", "TailRiskLoss", "DeficitBuffer"]


@dataclass
class TailConfig:
    """Hyper-parameters of the tail term.  Defaults are the A1 card values."""

    mode: str = "gpd"
    """gpd (A1) | cvar (A2) | ohem (A3) | none (A0)."""

    p: float = 0.15
    """Exceedance rate.  The worst ``p`` fraction of images define the tail."""

    alpha: float = 0.02
    """Target upper-tail level.  Must satisfy ``alpha < p``."""

    lam: float = 0.3
    """Weight of the tail term once warm-up finishes."""

    warmup_frac: float = 0.2
    """Fraction of total steps over which ``lam`` ramps linearly from 0.

    Not optional: before the model is roughly fit the deficit distribution is
    dominated by initialisation noise and the tail estimate is meaningless."""

    buffer_size: int = 512
    """``K * B`` detached deficits.  512 == 32 minibatches at batch 16.

    This is the main control on shape-estimate variance, and small-sample PWM
    is unforgiving.  At 512 the pool carries ~77 exceedances and fitted shapes
    move smoothly (a measured run: -0.36 to -0.68 across epochs).  At 128 the
    pool carries ~19, the raw estimate swings from +0.06 to -1.9 epoch to
    epoch, and the lower clamp starts binding -- at which point the weighting
    is no longer the advertised one.  If ``tail_xi_clamped`` is not near zero,
    raise this before touching anything else."""

    min_buffer: int = 128
    """Below this the tail term is inactive (returns 0)."""

    min_exceedances: int = 8
    """Below this many pooled exceedances the PWM fit is not attempted."""

    ema_momentum: float = 0.05
    """EMA rate for the threshold.  ``u <- (1-m) u + m quantile(buffer)``."""

    xi_mode: str = "live"
    """live | detached -- see ``polyptail.evt.pot_return_level_weights``."""

    xi_min: float = -1.5
    xi_max: float = 0.7
    """Shape clamp.  The upper bound is a real constraint (no finite GPD mean
    for ``xi >= 1``).  The lower bound is only a numerical guard and must sit
    well below the operating range: a soft-Dice deficit is bounded above by 1,
    so fitted shapes of -0.4 to -0.8 are normal, and a lower clamp that binds
    switches off the shape gradient and inverts the weight profile.  The
    ``tail/xi_clamped`` diagnostic counts how often either bound binds; if it
    is not near zero, the reported run is not running the advertised method."""

    weight_floor: Optional[float] = 0.0
    """Clip gradient weights from below.  ``0.0`` forbids the objective from
    ever rewarding an increase in a per-image deficit."""

    normalize: str = "sum1"
    """sum1 | none.

    ``sum1`` divides ``w`` by its sum, so every mode yields a *weighted mean
    exceedance* and ``lambda`` means the same thing across A1/A2/A3 and does
    not drift with the fitted ``xi``.  ``none`` gives the literal
    "minimise the estimated return level" objective; the return level is
    logged as ``q_hat`` in both cases."""

    rescale_live: bool = True
    """Scale live weights by ``n_pool / n_live`` so gradient magnitude does
    not depend on ``buffer_size``."""

    exceedance_cap: Optional[float] = None
    """A6.  Cap each exceedance at this value before weighting, bounding the
    influence of any single (possibly mis-annotated) image.  ``None`` = off."""

    cvar_selection: str = "pool"
    """pool | live -- which set the empirical top-``alpha/p`` is taken from
    in ``mode="cvar"``.  ``pool`` keeps A2 maximally comparable to A1."""

    def __post_init__(self) -> None:
        if self.mode not in ("gpd", "cvar", "ohem", "none"):
            raise ValueError(f"unknown tail mode {self.mode!r}")
        if self.mode != "none" and not (0.0 < self.alpha < self.p <= 1.0):
            raise ValueError(f"need 0 < alpha < p <= 1, got alpha={self.alpha}, p={self.p}")
        if self.normalize not in ("sum1", "none"):
            raise ValueError(f"unknown normalize {self.normalize!r}")
        if self.cvar_selection not in ("pool", "live"):
            raise ValueError(f"unknown cvar_selection {self.cvar_selection!r}")
        if self.min_buffer > self.buffer_size:
            raise ValueError("min_buffer must not exceed buffer_size")


class DeficitBuffer:
    """Fixed-capacity circular buffer of detached per-image deficits."""

    def __init__(self, capacity: int, device: torch.device | str = "cpu") -> None:
        self.capacity = int(capacity)
        self._buf = torch.zeros(self.capacity, dtype=torch.float32, device=device)
        self._n = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._n

    @torch.no_grad()
    def push(self, values: Tensor) -> None:
        v = values.detach().flatten().to(self._buf.dtype).to(self._buf.device)
        k = v.numel()
        if k == 0:
            return
        if k >= self.capacity:
            self._buf.copy_(v[-self.capacity:])
            self._n = self.capacity
            self._ptr = 0
            return
        end = self._ptr + k
        if end <= self.capacity:
            self._buf[self._ptr:end] = v
        else:
            first = self.capacity - self._ptr
            self._buf[self._ptr:] = v[:first]
            self._buf[: k - first] = v[first:]
        self._ptr = end % self.capacity
        self._n = min(self._n + k, self.capacity)

    @torch.no_grad()
    def values(self) -> Tensor:
        return self._buf[: self._n]

    def state_dict(self) -> dict:
        return {"buf": self._buf.clone(), "n": self._n, "ptr": self._ptr, "capacity": self.capacity}

    def load_state_dict(self, state: dict) -> None:
        self.capacity = int(state["capacity"])
        self._buf = state["buf"].clone().to(self._buf.device)
        self._n = int(state["n"])
        self._ptr = int(state["ptr"])


class TailRiskLoss(nn.Module):
    """Stateful tail term.  Call once per optimiser step with per-image deficits.

    The state (buffer, threshold, step counter) is serialised with
    ``state_dict``/``load_state_dict`` so that resuming a run resumes the tail
    estimator too, rather than silently restarting its warm-up.
    """

    def __init__(self, cfg: TailConfig, total_steps: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.total_steps = max(int(total_steps), 1)
        self.buffer = DeficitBuffer(cfg.buffer_size)
        self.register_buffer("_u", torch.tensor(float("nan")))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    # ---------------------------------------------------------------- utils

    @property
    def threshold(self) -> float:
        return float(self._u)

    def lam_at(self, step: Optional[int] = None) -> float:
        """Linear warm-up of ``lambda`` over the first ``warmup_frac`` of training."""
        if self.cfg.mode == "none":
            return 0.0
        s = int(self._step) if step is None else int(step)
        w = self.cfg.warmup_frac * self.total_steps
        if w <= 0:
            return self.cfg.lam
        return self.cfg.lam * min(1.0, s / w)

    @torch.no_grad()
    def _update_threshold(self) -> None:
        vals = self.buffer.values()
        if vals.numel() < self.cfg.min_buffer:
            return
        q = torch.quantile(vals.float(), 1.0 - self.cfg.p)
        if torch.isnan(self._u):
            self._u.fill_(float(q))
        else:
            m = self.cfg.ema_momentum
            self._u.mul_(1.0 - m).add_(m * float(q))

    # ------------------------------------------------------------- weights

    def _weights(self, e_pool_sorted: Tensor, fit: Optional[GPDFit]) -> Tensor:
        cfg = self.cfg
        n = e_pool_sorted.numel()
        if cfg.mode == "ohem":
            return torch.full((n,), 1.0 / n, device=e_pool_sorted.device, dtype=e_pool_sorted.dtype)
        if cfg.mode == "cvar":
            frac = cfg.alpha / cfg.p
            m = max(1, int(round(frac * n)))
            w = torch.zeros(n, device=e_pool_sorted.device, dtype=e_pool_sorted.dtype)
            w[n - m:] = 1.0 / m  # sorted ascending -> the last m are the largest
            return w
        assert cfg.mode == "gpd" and fit is not None
        return pot_return_level_weights(
            e_pool_sorted, fit, p=cfg.p, alpha=cfg.alpha,
            xi_mode=cfg.xi_mode, weight_floor=cfg.weight_floor,
            xi_min=cfg.xi_min, xi_max=cfg.xi_max,
        )

    # ------------------------------------------------------------- forward

    def forward(self, deficits: Tensor, advance: bool = True) -> tuple[Tensor, dict]:
        """Return ``(loss, stats)``.

        ``deficits`` is a 1-D tensor of per-image deficits for the current
        batch, **with** gradient.  ``loss`` is already multiplied by the
        warm-up-scheduled ``lambda``; add it straight to the base loss.

        Ordering matters and is deliberate: the threshold and the buffer half
        of the exceedance pool are read *before* this batch is appended.  That
        keeps ``u`` a function of data the model has already been scored on
        rather than of the current parameters, and it keeps the live batch
        from appearing twice in the pooled statistics -- which would let the
        current batch double-vote on the fit that weights it.
        """
        cfg = self.cfg
        d = deficits.flatten().float()
        device = d.device
        zero = torch.zeros((), device=device, dtype=d.dtype)

        stats: dict[str, float] = {
            "tail/lam": self.lam_at(),
            "tail/u": float(self._u) if not torch.isnan(self._u) else float("nan"),
            "tail/n_live": 0.0,
            "tail/n_pool": 0.0,
            "tail/active": 0.0,
            "tail/deficit_mean": float(d.detach().mean()) if d.numel() else float("nan"),
            "tail/deficit_max": float(d.detach().max()) if d.numel() else float("nan"),
        }
        try:
            return self._tail_loss(d, device, zero, stats), stats
        finally:
            self.buffer.push(d)
            self._update_threshold()
            if advance:
                self._step += 1

    def _tail_loss(self, d: Tensor, device, zero: Tensor, stats: dict) -> Tensor:
        cfg = self.cfg
        lam = self.lam_at()
        if cfg.mode == "none" or lam == 0.0 or math.isnan(float(self._u)):
            return zero

        u = float(self._u)
        e_live_all = d - u
        live_mask = e_live_all > 0
        n_live = int(live_mask.sum())
        e_live = e_live_all[live_mask]

        buf = self.buffer.values().to(device)
        e_buf = buf - u
        e_buf = e_buf[e_buf > 0]

        if cfg.exceedance_cap is not None:
            e_live = e_live.clamp(max=cfg.exceedance_cap)
            e_buf = e_buf.clamp(max=cfg.exceedance_cap)

        n_pool = n_live + int(e_buf.numel())
        stats["tail/n_live"] = float(n_live)
        stats["tail/n_pool"] = float(n_pool)
        if n_pool < cfg.min_exceedances or n_live == 0:
            return zero

        pool = torch.cat([e_live.detach(), e_buf])
        order = torch.argsort(pool)
        pool_sorted = pool[order]
        rank = torch.empty_like(order)
        rank[order] = torch.arange(n_pool, device=device)
        live_rank = rank[:n_live]

        fit: Optional[GPDFit] = None
        if cfg.mode == "gpd":
            fit = gpd_pwm_fit(pool_sorted, xi_min=cfg.xi_min, xi_max=cfg.xi_max)
            stats.update({
                "tail/xi": fit.xi,
                "tail/xi_raw": fit.xi_raw,
                "tail/beta": fit.beta,
                "tail/degenerate": float(fit.degenerate),
                "tail/xi_clamped": float(fit.clamped),
                "tail/q_hat": gpd_return_level(u, fit, cfg.p, cfg.alpha),
            })

        w = self._weights(pool_sorted, fit)
        w_sum = float(w.sum())
        stats["tail/w_sum"] = w_sum
        if cfg.normalize == "sum1":
            if w_sum <= 0.0:
                return zero
            w = w / w_sum

        w_live = w[live_rank]
        stats["tail/n_live_active"] = float((w_live > 0).sum())
        if cfg.rescale_live and n_live > 0:
            w_live = w_live * (n_pool / n_live)

        value = (w * pool_sorted).sum()                      # detached: reported statistic
        grad_path = (w_live * (e_live - e_live.detach())).sum()  # zero value, carries d/dtheta
        loss = self.lam_at() * (value + grad_path)
        stats["tail/loss_unweighted"] = float(value)
        stats["tail/loss"] = float(loss.detach())
        stats["tail/active"] = 1.0
        return loss

    # ------------------------------------------------------------ (de)serial

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        sd = super().state_dict(*args, **kwargs)
        sd["_deficit_buffer"] = self.buffer.state_dict()
        return sd

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        state_dict = dict(state_dict)
        buf = state_dict.pop("_deficit_buffer", None)
        if buf is not None:
            self.buffer.load_state_dict(buf)
        return super().load_state_dict(state_dict, strict=strict)
