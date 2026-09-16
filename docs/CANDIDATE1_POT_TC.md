# Candidate 1 — POT-TC: peaks-over-threshold tail calibration

**Source field:** extreme-value statistics (hydrology, insurance, operational
risk).
**Attachment point:** the loss function. Nothing else.
**Novelty status (from the brief):** ACCEPTED, bounded — no implementation-level
precedent found within the brief's stated search scope as of 2026-09-15. That
is a bounded negative finding, not proof of global novelty.

---

## 1. The problem it targets

External-set degradation in polyp segmentation is **not** a uniform downward
shift. It is a heavy left tail of per-image scores:

* EndoCV2021, six centres: *"around 28.33% of data 1, 21.25% of data 2, 21.66%
  of data 3 and 31.5% of data 4 has DSC equal or lower than 0.50"*
  ([Sci. Rep. 14, s41598-024-52063-x](https://www.nature.com/articles/s41598-024-52063-x)).
* On the fixed split, ETIS Dice runs .637–.790 with **HD95 of 126.8–230.1 px**,
  while CVC-300 reaches .869–.904 at HD95 20.8–32.5 px
  ([arXiv:2607.08203](https://arxiv.org/html/2607.08203), preprint). HD95 above
  126 px on a 966×1225 image means *spatially displaced false-positive blobs*,
  not imprecise boundaries.

A mean-reducing objective weights a 0.95 → 0.96 improvement identically to a
0.10 → 0.20 improvement, and there are far more of the former. POT-TC changes
what the objective is a mean *of*.

---

## 2. The mechanism

Per training image, one scalar: the **deficit** `d_i = 1 − softDice_i`,
computed on the same prediction inference uses (`P1 + P2` for Polyp-PVT).

Four steps, all closed-form, no extra parameters, no architectural change, and
**no change whatsoever to the inference path** — which is what makes it legal
under a zero-shot external-test protocol:

1. **Threshold.** A sliding replay buffer of the last `buffer_size` detached
   deficits defines `u`, the empirical `(1 − p)` quantile, EMA-smoothed. `u` is
   a buffer, not a parameter: no gradient flows through it.
2. **Pool.** The exceedance set is the live batch exceedances (which carry
   gradient) concatenated with the buffer exceedances (which do not). Pooling
   is not a convenience: at batch 16 with `p = 0.15` a single step yields about
   **two** exceedances, which is not enough to estimate anything.
3. **Fit.** A generalized Pareto is fitted to the pooled exceedances by
   probability-weighted moments, in closed form. Pickands–Balkema–de Haan says
   this is the right family for exceedances above a high threshold; PWM gives
   `(ξ, β)` from two linear statistics of the order statistics, with no
   iteration and no autograd through a sort.
4. **Weight.** The loss is `Σ w_i e_i`, where `w_i = ∂q_{1−α}/∂e_(i)` — the
   sensitivity of the model-implied `(1−α)` quantile to each exceedance,
   available in closed form because the PWM moments are linear in the order
   statistics.

`L = L_base + λ(t) · L_tail`, with `λ` ramped linearly from zero over the first
20% of training. The warm-up is not optional: before the model is roughly fit,
the deficit distribution is initialisation noise and the tail estimate is
meaningless.

### Full derivation

With `a_s = E[X(1−F(X))^s]`, a GPD with `ξ < 1` satisfies
`a_s = β / ((s+1)(s+1−ξ))`, which inverts in closed form:

```
D    = a_0 − 2 a_1        ( = β / ((1−ξ)(2−ξ)) > 0 for ξ < 1 )
ξ    = 2 − a_0 / D
β    = 2 a_0 a_1 / D
```

The sample versions are linear in the ascending order statistics
`e_(1) ≤ … ≤ e_(n)`:

```
a_0 = (1/n) Σ e_(i)
a_1 = (1/n) Σ t_i e_(i),     t_i = (n − i)/(n − 1)
```

The POT return level, with `r = p/α` and `g(ξ) = (r^ξ − 1)/ξ` (removable
singularity at `g(0) = log r`), is `q_{1−α} = u + β g(ξ)`, and

```
∂β/∂e_(i) = (2/(n D²)) (a_0² t_i − 2 a_1²)
∂ξ/∂e_(i) = (2/(n D²)) (a_1 − a_0 t_i)
∂q/∂e_(i) = g(ξ) ∂β/∂e_(i) + β g'(ξ) ∂ξ/∂e_(i)
```

`polyptail/evt.py` implements exactly this; `tests/test_evt.py` checks the
weights against finite differences of the return level to 1e-4 relative, and
checks PWM recovery of `(ξ, β)` from synthetic GPD samples.

---

## 3. Two things the implementation discovered

Both were found by running the code, not by reading the card, and both change
the method materially.

### 3.1 Holding the shape fixed makes the objective reward failure

The card says to treat `(ξ, β)` as "detached plug-ins". Taken literally — only
the scale path carries gradient — the weight on the **largest** exceedance is
`−4a_1²/(n D²) < 0` for every admissible shape. Minimising the plug-in scale
therefore *pays the network to make the worst image worse*, because inflating
`e_(n)` inflates `D` faster than it inflates the numerator. Measured, over
3000-sample GPD draws:

| true ξ | `w[-1] × n`, shape detached | `w[-1] × n`, shape live |
|---|---|---|
| −0.4 | **−2.72** | +2.99 |
| 0.0 | **−2.00** | +2.03 |
| +0.4 | **−1.10** | +0.91 |
| +0.6 | **−0.51** | +0.33 |

Letting the shape carry gradient cancels the pathology: a fatter observed tail
raises `ξ`, which raises the return level, and the two effects oppose. Default
is therefore `xi_mode: live`, with `weight_floor: 0.0` clipping whatever small
negative weight survives at strongly negative shapes. `xi_mode: detached`
remains available, and contrasting the two isolates how much of the effect is
tail *shape*.

### 3.2 The card's shape clamp is wrong for a bounded score, and inverts the weighting

The card proposes clamping `ξ ∈ [−0.5, 0.7]`. The upper bound is a real
constraint (the GPD has no finite mean for `ξ ≥ 1`). The lower bound inherits
an upper-tail moment argument that does not apply downward: `ξ < 0` simply
means the tail has a finite right endpoint, and the POT-TC deficit is
`1 − softDice`, **hard-bounded at 1**. Fitted shapes of −0.4 to −0.8 are the
normal operating regime.

A clamp that binds has zero local sensitivity, so the shape gradient switches
off exactly when it binds. Measured on a GPD sample with true `ξ = −0.7`,
weight profile by exceedance rank:

| | rank 0% | 50% | 90% | 99% | 100% |
|---|---|---|---|---|---|
| clamp at −0.5 (binds) | **14.62** | 5.53 | 0.00 | 0.00 | **0.00** |
| clamp at −1.5 (clear) | 0.00 | 0.91 | 2.91 | 3.36 | **3.41** |

The binding clamp puts 14.6× weight on the *smallest* exceedances and zero on
the worst fifth — the exact opposite of a tail objective. **Default here is
`xi_min: −1.5`**, and every run logs `tail_xi_clamped`; the trainer warns
loudly if the clamp binds on more than 20% of steps.

A first run of this repository on synthetic data produced fitted shapes of
−0.36 to −0.68 with the clamp never binding, which is the regime the defaults
are set for.

---

## 4. Why this is not hard-example mining

With the shape live and unclamped, the weight profile is **adaptive in a way
uniform weighting cannot be**:

* **ξ̂ < 0 (bounded tail, the usual regime here):** weights increase
  monotonically with exceedance rank. Chase the worst frames.
* **ξ̂ > 0 (heavy tail):** weights *decrease* with rank. When the fitted tail
  is heavy, the biggest lever on the `(1−α)` quantile is the body of the
  exceedances, not the single worst image — a built-in robustness to the label
  noise the card lists as failure mode #1.
* **ξ̂ ≈ 0:** nearly uniform, i.e. approximately hard-example mining.

Measured profiles (weights normalised to mean 1, by rank):

| fitted ξ̂ | rank 0% | 25% | 50% | 75% | 95% | 100% |
|---|---|---|---|---|---|---|
| −0.26 | 0.11 | 0.55 | 1.00 | 1.44 | 1.80 | 1.89 |
| −0.06 | 0.79 | 0.90 | 1.00 | 1.10 | 1.19 | 1.21 |
| +0.15 | 1.33 | 1.17 | 1.00 | 0.84 | 0.70 | 0.67 |
| +0.70 | 2.03 | 1.52 | 1.01 | 0.49 | 0.08 | 0.00 |

Ablation A3 sits at exactly 1.00 in every cell, always.

This yields a **testable sub-hypothesis** that costs nothing to check: the
A1 − A3 gap should track how far ξ̂ sits from zero during training. Every run
logs `tail_xi` per epoch, so it can be read off directly. If ξ̂ hovers at zero
for the whole run and A1 ≈ A3, the honest conclusion is that on this data the
EVT content reduces to hard-example mining.

---

## 5. Ablations, and what each one can kill

All arms are identical apart from the weight vector over a common exceedance
pool — same threshold, same buffer, same schedule, same `λ`. That is what
makes the contrast a test of the weighting rather than of three differently
plumbed objectives.

| arm | config | weighting | what it isolates |
|---|---|---|---|
| **A0** | `a0_baseline.yaml` | none | the reproduced baseline |
| **A1** | `a1_pottc.yaml` | GPD return-level sensitivity | the candidate |
| **A2** | `a2_cvar.yaml` | uniform on the top `α/p` of the pool (the *empirical* `(1−α)` tail average) | **extrapolation.** If A1 ≈ A2, the EVT content does nothing and POT-TC is a re-parameterisation of CVaR. |
| **A3** | `a3_ohem.yaml` | uniform on the whole pool | **shape.** Separates "focus on hard images" from "shape the tail". |
| **A4** | `configs/sweeps/` | λ ∈ {0.1, 0.3, 1.0}, p ∈ {0.05, 0.15, 0.30} | sensitivity, on a **training-split** fold |
| **A5** | `a5_equal_budget.yaml` | none, extra epochs | equal compute. POT-TC adds no parameters, so the control is wall-clock. `tools/run_ablation.py --auto-budget` computes the epoch count. |
| **A6** | `a6_capped.yaml` | A1 with capped exceedances | label-noise robustness |
| **A7** | `a7_pranet_{a0,a1}.yaml` | A0/A1 on PraNet | backbone independence |

One property of A2 worth reporting alongside its score: at `α = 0.02` the
empirical tail is the top 13% of the exceedance pool, and a batch of 16
supplies about two live exceedances, so **A2 receives a gradient on roughly a
third of steps while A1 receives one on nearly all of them**. That difference
in effective sample size *is* the extreme-value argument, made mechanically.
Report A2's update count, and run `pot.tail.cvar_selection=live` as a
cross-check that A2 is not merely losing on update count.

---

## 6. Falsification criterion — pre-registered, evaluated mechanically

`tools/analyze.py` computes this. **Reject POT-TC** if any of:

* **C1** — the paired ETIS mDice improvement over A0 is ≤ 0 with a CI crossing
  zero, **and** the fraction of ETIS images with Dice ≤ 0.5 is not reduced by
  at least 3 absolute points; **or**
* **C2** — the A1 − A2 CI contains zero on **every** external dataset; **or**
* **C3** — any single dataset declines by more than 1.0 mDice with a CI
  excluding zero while the pooled average rises.

Thresholds are fixed in code, not tuned. Changing them after seeing results
converts a pre-registered test into a post-hoc one, and the result stops
meaning anything.

---

## 7. Failure modes, honestly

| failure mode | mitigation in this implementation |
|---|---|
| **Label-noise amplification** — the worst-loss images may simply be mis-annotated | A6 caps per-image exceedances; every run writes `per_image_deficits.csv` naming which training images populate the tail each epoch. If the same handful of stems dominate, inspect them before believing anything else. Note also that heavy fitted tails automatically down-weight the extremes (§4). |
| **Estimator instability** — `ξ` is high-variance in small samples | The pooled buffer supplies ~77 exceedances per step instead of ~2. `buffer_size` is the control: measured, 512 gives fitted shapes moving smoothly across −0.36 to −0.68, while 128 (~19 exceedances) swings from +0.06 to −1.9 epoch to epoch and starts hitting the clamp. `tail_xi_raw`, `tail_xi_clamped` and `tail_degenerate` are logged every epoch; the PWM fit falls back to the exponential (`ξ=0, β=a_0`) when the denominator collapses. |
| **`ξ ≥ 1` degeneracy** — no finite GPD mean | clamped at 0.7, with the binding rate logged. |
| **Optimisation conflict** — the tail term suppresses gradient on easy images | `λ` warm-up over the first 20%; `normalize: sum1` keeps `λ` from drifting with ξ̂. |
| **Negative transfer** — if the training-set tail (hard Kvasir/ClinicDB frames) is a different failure mode from the external-set tail (ETIS specular/low-contrast frames), optimising the former is at best irrelevant | **This is the single most likely way the idea fails, and nothing in the implementation can fix it.** It is what C1 tests. |
| **Calibration drift** — a softened boundary may move the optimal threshold away from the protocol's fixed 0.5 | both scoring modes are reported; `dice` (fixed 0.5) versus `dice_sweep` (threshold-averaged) diverging between A0 and A1 is the signature. |
| **Compute** | one scalar buffer, one closed-form fit per step. Expect a few percent; `results.json` records wall-clock and A5 matches it. |

---

## 8. What this implementation does *not* claim

Nothing about accuracy. The brief is explicit that every benefit is `[HYP]`,
and no measurement on polyp data exists — including here. What is established
is narrower and worth stating precisely:

* the estimator is correct (PWM recovers known GPD parameters; the weights are
  the true derivative of the return level, to 1e-4);
* the objective cannot reward increasing a per-image deficit;
* it adds zero parameters and leaves inference untouched, so the
  equal-parameter control is exact;
* the weighting is genuinely non-uniform and shape-adaptive, so A1 vs A3 is
  not a vacuous comparison;
* two specific ways the candidate card's own parameterisation would have
  silently broken the method have been found and fixed.
