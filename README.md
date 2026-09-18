# POT-TC — extreme-value tail calibration for polyp segmentation

Implementation of **Candidate 1** of the research-verification brief: a
peaks-over-threshold / generalized-Pareto training objective that shapes the
**left tail** of the per-image Dice distribution, together with the protocol
infrastructure the brief marks as *blocking* before any comparison means
anything.

Two ideas, in one sentence each:

1. External-set failure in polyp segmentation is a heavy left tail — 21-31% of
   frames at Dice ≤ 0.5 — not a uniform downward shift, so fit a generalized
   Pareto to the worst per-image losses and optimise the model-implied extreme
   quantile instead of the mean.
2. The benchmark this would be measured on is not well-defined by its own
   sources, so freeze it, audit it for duplicates, fix one evaluation
   implementation, and make the falsification test something a script decides
   rather than a discussion.

**No accuracy claim is made anywhere in this repository.** The brief marks
every expected benefit as a hypothesis with no supporting measurement, and
that is still true here. What is established is narrower: the estimator is
correct, the objective cannot reward failure, it adds zero parameters, its
weighting is genuinely shape-adaptive, and two ways the candidate card's own
parameterisation would have silently broken the method have been found and
fixed.

---

## Quick start

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117   # see docs/HARDWARE.md
pip install -r requirements-dev.txt
pytest -q                                            # 201 tests, ~25 s, CPU only

# End-to-end on synthetic data, no GPU and no downloads, ~30 s:
python tools/make_smoke_data.py --out ./_smoke_data
python tools/train.py --config configs/smoke.yaml
```

With the real datasets in `./dataset/` (PraNet layout) and backbones in
`./pretrained_pth/`:

```bash
# 1. Freeze the data and publish the duplicate audit.  Do this first.
python tools/freeze_manifest.py  --root ./dataset --out manifests/pranet_protocol.json
python tools/hash_collisions.py  --root ./dataset --manifest manifests/pranet_protocol.json \
                                 --out manifests/collisions.json

# 2. Check the configuration fits your card before committing a week to it.
python tools/check_memory.py --config configs/a1_pottc.yaml

# 3. Baseline, candidate, decisive ablation.  Three seeds each.
python tools/run_ablation.py \
    --configs configs/a0_baseline.yaml configs/a1_pottc.yaml configs/a2_cvar.yaml \
    --seeds 0 1 2 --out-dir runs/main --auto-budget a0_baseline a1_pottc

# 4. Paired statistics and the pre-registered verdict.
python tools/analyze.py --runs runs/main --baseline a0_baseline --method a1_pottc \
    --a2 a2_cvar --seeds 0 1 2 --out reports/main.md
```

---

## The method, briefly

Each training image contributes one scalar, the **deficit** `d = 1 − softDice`,
computed on the same prediction inference uses. A sliding buffer of recent
deficits defines a threshold `u` at the `(1 − p)` quantile. The exceedances
above `u` — the live batch's, pooled with the buffer's for stability — are
fitted with a generalized Pareto by probability-weighted moments, in closed
form. The loss is the sensitivity of the model-implied `(1 − α)` quantile to
each exceedance:

```
L = L_base + λ(t) · Σ_i (∂q_{1−α}/∂e_(i)) · e_i
```

Everything is closed-form: no iteration, no autograd through a sort, no extra
parameters, and **the inference path is untouched** — which is what makes it
legal under a zero-shot external-test protocol and makes the equal-parameter
control exact rather than approximate.

The full derivation, the measured weight profiles, and the two design
corrections are in **[`docs/CANDIDATE1_POT_TC.md`](docs/CANDIDATE1_POT_TC.md)**.

### It is not hard-example mining

With the shape parameter carrying gradient, the weighting adapts to the fitted
tail. Relative weight by exceedance rank, measured:

| fitted ξ̂ | rank 0% | 50% | 95% | 100% | behaviour |
|---|---|---|---|---|---|
| −0.26 | 0.11 | 1.00 | 1.80 | 1.89 | bounded tail → chase the worst frames |
| −0.06 | 0.79 | 1.00 | 1.19 | 1.21 | ≈ uniform |
| +0.70 | 2.03 | 1.01 | 0.08 | 0.00 | heavy tail → discount the extremes |

Ablation A3 is 1.00 everywhere, always. Whether that adaptivity buys anything
on polyp data is exactly what A1 vs A3 measures, and every run logs `tail_xi`
per epoch so the answer can be read off directly.

---

## Two things running the code found

Both change the method materially, and both came from executing it rather than
from reading the specification.

**The literal reading of the card rewards failure.** Holding the GPD shape
fixed and letting only the scale carry gradient puts a *negative* weight on
the largest exceedance for every admissible shape — the objective pays the
network to make its worst image worse. Letting the shape carry gradient
cancels it; a weight floor at zero catches the remainder.

**The card's shape clamp inverts the weighting.** `ξ ∈ [−0.5, 0.7]` inherits
an upper-tail moment argument that does not apply downward. A soft-Dice
deficit is hard-bounded at 1, so fitted shapes of −0.4 to −0.8 are normal, the
lower clamp binds almost every step, and a binding clamp zeroes the shape
gradient — putting 14.6× weight on the *smallest* exceedances and zero on the
worst fifth. The default here is `−1.5`, and the trainer warns if it binds.

---

## The protocol infrastructure

The brief establishes that the canonical polyp benchmark disagrees with itself:
PraNet's paper describes an 80/10/10 split while its repository ships
900+550 train; CVC-ColonDB's originating paper documents 300 images while the
released directory holds 380; CVC-300 may be a subset of CVC-ColonDB; and one
frozen model on one frozen test set is reported as 0.709 / 0.712 / 0.715 /
0.716 mDice by four different papers — a **0.3-0.8 point noise floor** that
swallows most published margins.

So this repository ships the missing pieces:

| tool | discharges |
|---|---|
| `tools/freeze_manifest.py` | SHA-256 of every file **and** of every decoded image, so a re-encode is distinguishable from a data change |
| `tools/hash_collisions.py` | the CVC-300 ∩ CVC-ColonDB question, plus a train/test leakage audit the brief does not even ask for |
| `polyptail/eval/` | one evaluation implementation, two clearly-labelled scoring modes: fixed-0.5 (what deployment gets) and the PraNet MATLAB threshold-average (what published tables contain) |
| `polyptail/stats/` | paired intervals three ways, Holm adjustment across the five datasets, and the §9 falsification criterion evaluated mechanically |

`docs/PROTOCOL.md` is the contract, including the four places where the
released Polyp-PVT code and its own paper disagree — among them a
`reduce='none'` argument that silently makes the "weighted BCE" an unweighted
one in the code that produced the published numbers.

---

## Layout

```
polyptail/
  evt.py               closed-form GPD / PWM / return-level machinery (pure, no state)
  config.py            typed config, YAML inheritance, dotted CLI overrides
  losses/              structure loss, per-image deficit, the stateful tail term
  data/                content-addressed manifests, datasets, perceptual-hash audit
  models/              Polyp-PVT (+PVTv2), PraNet (+Res2Net), a tiny U-Net for CI
  eval/                the one evaluation implementation
  stats/               paired intervals and the falsification verdict
  engine/              the training loop
tools/                 freeze_manifest, verify_manifest, hash_collisions, check_memory,
                       train, evaluate, run_ablation, analyze, make_smoke_data
configs/               base + A0/A1/A2/A3/A5/A6/A7 + sweeps + a CPU smoke config
tests/                 201 tests, CPU only
docs/                  PROTOCOL, CANDIDATE1_POT_TC, EXPERIMENTS, HARDWARE, REPRODUCIBILITY
```

No torchvision, no timm, no albumentations — see `docs/HARDWARE.md` for why
that matters on CUDA 11.4.

---

## Documentation

* **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — start here if you are about to
  run this: a linear step-by-step from empty directory to verdict, with the
  decision points and what to do when each step misbehaves.
* **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** — the locked protocol, the
  dataset provenance warnings, and the §6.2 blocking-items scoreboard.
* **[`docs/CANDIDATE1_POT_TC.md`](docs/CANDIDATE1_POT_TC.md)** — derivation,
  measured weight profiles, the two design corrections, failure modes.
* **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** — the ablation ladder and
  the exact commands, in order.
* **[`docs/HARDWARE.md`](docs/HARDWARE.md)** — CUDA 11.4 / 12 GB specifics,
  the memory ladder, and the time budget.
* **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** — what
  "reproduced" has to mean, and how to check it.

## Acknowledgements

The Polyp-PVT decoder and the PVTv2 backbone are adapted from the official
release at <https://github.com/DengPingFan/Polyp-PVT>; PraNet and Res2Net-v1b
follow <https://github.com/DengPingFan/PraNet> and the Res2Net authors'
release. Changes to those files are listed in their module docstrings.
