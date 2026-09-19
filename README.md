# POT-TC — extreme-value tail calibration for polyp segmentation

Implementation of **Candidate 1** of the research-verification brief: a
peaks-over-threshold / generalized-Pareto training objective that shapes the
**left tail** of the per-image Dice distribution, together with the protocol
infrastructure the brief marks as *blocking* before any comparison means
anything.

Two ideas, one sentence each:

1. External-set failure in polyp segmentation is a heavy left tail — 21–31% of
   frames at Dice ≤ 0.5 — not a uniform downward shift. So fit a generalized
   Pareto to the worst per-image losses and optimise the model-implied extreme
   quantile instead of the mean.
2. The benchmark this would be measured on is not well defined by its own
   sources. So freeze it, audit it for duplicates, fix one evaluation
   implementation, and make the falsification test something a script decides.

**No accuracy claim is made anywhere in this repository.** The brief marks
every expected benefit as a hypothesis with no supporting measurement, and
that is still true. What *is* established is narrower: the estimator is
correct, the objective cannot reward failure, it adds zero parameters, its
weighting is genuinely shape-adaptive, and two ways the candidate card's own
parameterisation would have silently broken the method were found and fixed.

---

# Getting started on your server

Steps 1–3 are cheap and need no GPU. Step 4 onwards is days of GPU time.
Step 6 is a gate: if it fails, nothing after it is interpretable.

`docs/RUNBOOK.md` is the same sequence with every diagnostic explained.

## 1. Create the conda environment

```bash
git clone https://github.com/Yassinesr/Test.git polyptail && cd polyptail
git checkout claude/cool-rubin-kd7m6s     # until this branch is merged to main

conda env create -f environment.yml
conda activate polyptail
```

Three environment files, all producing the same package set:

| file | use it when |
|---|---|
| `environment.yml` | **default** — GPU, torch 2.0.1+cu117 for a CUDA 11.4 driver |
| `environment-cpu.yml` | no GPU: unit tests, the smoke run, `tools/analyze.py` |
| `environment-cn.yml` | pypi.org / repo.anaconda.com blocked or slow — routes through the Tsinghua (TUNA) mirrors |

Verify before going further:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: 2.0.1+cu117 True NVIDIA GeForce RTX 3080 Ti
```

Three things in those files are deliberate, and each prevents a confusing
first-run failure:

* **Torch comes from pip, not conda.** PyTorch deprecated its own Anaconda
  channel, and the pip wheels bundle their own CUDA runtime — so the cu117
  build runs against a CUDA 11.4 driver with no `cudatoolkit` to reconcile.
* **`nodefaults` is declared.** Otherwise conda contacts `repo.anaconda.com`
  for channels this project does not use, which fails on restricted networks.
* **NumPy is capped below 2.0.** torch 2.0.1 predates NumPy 2 C-API support
  and fails at import under it. The cap is identical in all four install paths
  and `tests/test_packaging.py` asserts they agree.

Behind a proxy, or `conda env create` failing? `ProxyError … Connection
refused` means a proxy is configured and dead — a mirror cannot fix that,
because you would need the proxy to reach the mirror. Triage, mirrors, and a
route that needs **no network at all** (reuse the environment you already run
Polyp-PVT with — this project needs only torch, numpy, scipy, pillow, pyyaml)
are in [`docs/HARDWARE.md`](docs/HARDWARE.md#restricted-networks-proxies-and-mirrors).

## 2. Verify the install — no data, no GPU, under a minute

```bash
pytest -q                                          # 289 passed, ~25 s
python tools/make_smoke_data.py --out ./_smoke_data
python tools/train.py --config configs/smoke.yaml  # full pipeline on synthetic data
```

The smoke run's Dice is poor by design — a 0.1M-parameter net, 2 epochs, 48
images. It proves plumbing, not accuracy. Fix any failure here before
downloading anything.

## 3. Point at your data

**If the datasets are already on the machine — e.g. a Polyp-PVT checkout next
door — do not download them again.**

```bash
python tools/prepare_data.py --link ../Polyp-PVT/dataset   # symlink; nothing copied
python tools/prepare_data.py --check
```

`--link` validates the target *before* creating the symlink, so a wrong path
fails there — where the message is about your dataset — rather than at freeze
time where it is about a hash. It is idempotent, refuses to clobber a real
directory, and refuses to silently repoint an existing link.

Two alternatives: `--check --root <path>` to leave the data where it is, or
`data.root=<path>` on every command. Manifests store paths **relative to the
root**, so a manifest frozen through the symlink verifies unchanged against the
real path. Two checkouts can share one copy without either holding a weaker
protocol for it.

Only if you have no copy: `python tools/prepare_data.py --download --pretrained`
(best-effort; Google Drive rate-limits, and the tool prints the manual links).

The pretrained backbone must exist at `pretrained_pth/pvt_v2_b2.pth` — link
that from a Polyp-PVT checkout too if you have one.

`--check` compares your tree against what the reference implementation
*hard-codes* and names the actual problem — an archive unzipped one level too
deep, a split renamed `CVC-T`, an unpaired mask, an extension the reference
silently skips:

```
split                               pairs  expected  status
TrainDataset                         1450      1450  ok
TestDataset/Kvasir                    100       100  ok
TestDataset/CVC-ClinicDB               62        62  ok
TestDataset/CVC-ColonDB               380       380  ok
TestDataset/CVC-300                    60        60  ok
TestDataset/ETIS-LaribPolypDB         196       196  ok
```

## 4. Freeze the data and publish the duplicate audit

```bash
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
python tools/hash_collisions.py --root ./dataset \
    --manifest manifests/pranet_protocol.json --out manifests/collisions.json
git add manifests/ && git commit -m "Freeze dataset manifest and publish collision matrix"
```

Commit both — they are two of the blocking items the literature does not
supply. A non-empty cross-split collision matrix means the "five independent
test sets" are not independent; `docs/PROTOCOL.md` §2.2 says what to report.

## 5. Check it fits, and time one epoch

```bash
python tools/check_memory.py --config configs/a1_pottc.yaml
python tools/train.py --config configs/a1_pottc.yaml \
    optim.epochs=1 eval.hd95=false run.name=timing_probe
```

`check_memory` runs real steps at each scale; the 1.25 scale (448×448) decides
whether a 12 GB card OOMs. If it does not fit, take the first rung that does —
`optim.batch_size=8 optim.grad_accum=2`, then `4`/`4` — and **append it to
every later command**, or the arms stop being comparable. Never lower
`batch_size` without raising `grad_accum`.

Multiply the probe's `epoch_seconds` by 100 for one full run. Expect roughly
8–14 h per run on a 3080 Ti, but use your measured number.

## 6. Run the baseline — this is the gate

```bash
python tools/run_ablation.py --configs configs/a0_baseline.yaml \
    --seeds 0 1 2 --out-dir runs/main
```

Sequential on purpose: two runs do not fit on 12 GB, and interleaving destroys
the wall-clock measurement A5 needs. The runner skips any `(config, seed)` that
already has a `results.json`, so an interrupted sweep resumes by re-running the
same command. Use `tmux`.

Then check the band — **every split within ±0.5 mDice**, compared against
`dice_sweep`, not `dice`, because the published numbers are threshold-averaged:

| dataset | Polyp-PVT paper | independent re-evaluation |
|---|---|---|
| Kvasir | 0.917 | — |
| CVC-ClinicDB | 0.937 | — |
| CVC-ColonDB | 0.808 | 0.811 |
| CVC-300 | 0.900 | 0.904 |
| ETIS-LaribPolypDB | 0.787 | 0.790 |

The check script and the five things that usually explain a miss are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) §3. Outside the band you
are measuring the harness, not the method — the cross-paper noise floor is
already 0.3–0.8 mDice, so a harness a point off produces differences larger
than anything POT-TC could add.

## 7. Run the candidate and the ablation that can kill it

```bash
python tools/run_ablation.py \
    --configs configs/a1_pottc.yaml configs/a2_cvar.yaml \
    --seeds 0 1 2 --out-dir runs/main --auto-budget a0_baseline a1_pottc

python tools/analyze.py --runs runs/main --baseline a0_baseline --method a1_pottc \
    --a2 a2_cvar --seeds 0 1 2 --out reports/main.md
```

A2 replaces the GPD return level with the *empirical* tail average over the
same exceedance pool — same threshold, buffer, schedule and λ. If A1 − A2
straddles zero on every external dataset, the extrapolation contributes
nothing. That is rejection condition **C2**, and it is cheap, so run it early.

`analyze.py` prints per-dataset tables with three kinds of paired 95% interval,
the left-tail diagnostics, and a **pre-registered verdict**. If it says REJECT,
the candidate is rejected — the thresholds live in `polyptail/stats/paired.py`
and editing them after seeing results makes the test post-hoc.

While it runs, watch `clamped` in the epoch summaries — it must stay near
0.00. Above 0.20 the trainer warns: the shape gradient is off and the weighting
is not the advertised one. Raise `pot.tail.buffer_size` first.

## 8. The rest of the ladder

| arm | config | answers |
|---|---|---|
| **A3** | `a3_ohem.yaml` | is the EVT *shape* doing anything, or is this hard-example mining? |
| **A5** | `a5_equal_budget.yaml` | is the gain just extra compute? (`--auto-budget` printed the epoch count) |
| **A6** | `a6_capped.yaml` | is the tail genuine difficulty or label noise? |
| **A4** | `configs/sweeps/` | how sensitive to λ and p? (on a *training-split* fold) |
| **A7** | `a7_pranet_{a0,a1}.yaml` | is the effect backbone-specific? |

---

# The method

Each training image contributes one scalar, the **deficit** `d = 1 − softDice`,
computed on the same prediction inference uses. A sliding buffer of recent
deficits defines a threshold `u` at the `(1 − p)` quantile. The exceedances
above `u` — the live batch's, pooled with the buffer's, because batch 16 yields
only ~2 — are fitted with a generalized Pareto by probability-weighted moments,
in closed form. The loss weights each exceedance by the sensitivity of the
model-implied `(1 − α)` quantile to it:

```
L = L_base + λ(t) · Σ_i (∂q_{1−α}/∂e_(i)) · e_i
```

No iteration, no autograd through a sort, no extra parameters, and **the
inference path is untouched** — which makes it legal under a zero-shot
external-test protocol and makes the equal-parameter control exact rather than
approximate.

Full derivation, measured weight profiles and failure modes:
[`docs/CANDIDATE1_POT_TC.md`](docs/CANDIDATE1_POT_TC.md).

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

### Two things running the code found

**The literal reading of the card rewards failure.** Holding the GPD shape
fixed and letting only the scale carry gradient puts a *negative* weight on the
largest exceedance for every admissible shape — the objective pays the network
to make its worst image worse. Letting the shape carry gradient cancels it; a
weight floor at zero catches the remainder.

**The card's shape clamp inverts the weighting.** `ξ ∈ [−0.5, 0.7]` inherits an
upper-tail moment argument that does not apply downward. A soft-Dice deficit is
hard-bounded at 1, so fitted shapes of −0.4 to −0.8 are normal, the lower clamp
binds almost every step, and a binding clamp zeroes the shape gradient — putting
14.6× weight on the *smallest* exceedances and zero on the worst fifth. The
default here is `−1.5`, chosen where the profile saturates, and the trainer
warns if it binds.

---

# The protocol infrastructure

The brief establishes that the canonical polyp benchmark disagrees with itself:
PraNet's paper describes an 80/10/10 split while its repository ships 900+550
train; CVC-ColonDB's originating paper documents 300 images while the released
directory holds 380; CVC-300 may be a subset of CVC-ColonDB; and one frozen
model on one frozen test set is reported as 0.709 / 0.712 / 0.715 / 0.716 mDice
by four different papers — a **0.3–0.8 point noise floor** that swallows most
published margins.

| tool | discharges |
|---|---|
| `tools/prepare_data.py` | validates a tree against what the reference hard-codes, before anything else runs |
| `tools/freeze_manifest.py` | SHA-256 of every file **and** every decoded image, so a re-encode is distinguishable from a data change |
| `tools/hash_collisions.py` | the CVC-300 ∩ CVC-ColonDB question, plus a train/test leakage audit the brief does not even ask for |
| `polyptail/eval/` | one evaluation implementation, two labelled modes: fixed-0.5 at native resolution, and the PraNet MATLAB threshold-average that published tables contain |
| `polyptail/stats/` | paired intervals three ways, Holm adjustment across the five datasets, and the §9 falsification criterion evaluated mechanically |

[`docs/PROTOCOL.md`](docs/PROTOCOL.md) is the contract, including five places
where the released Polyp-PVT code and its own paper disagree — among them a
`reduce='none'` argument that silently makes the "weighted BCE" unweighted, and
a `TestDataset/test/` directory the training script requires but the archive
does not ship.

---

# Layout

```
polyptail/
  evt.py               closed-form GPD / PWM / return-level machinery (pure, no state)
  config.py            typed config, YAML inheritance, dotted CLI overrides
  losses/              structure loss, per-image deficit, the stateful tail term
  data/                manifests, layout validation, datasets, perceptual-hash audit
  models/              Polyp-PVT (+PVTv2), PraNet (+Res2Net), a tiny U-Net for CI
  eval/                the one evaluation implementation
  stats/               paired intervals and the falsification verdict
  engine/              the training loop
tools/                 prepare_data, freeze_manifest, verify_manifest, hash_collisions,
                       check_memory, train, evaluate, run_ablation, analyze,
                       make_smoke_data
configs/               base + A0/A1/A2/A3/A5/A6/A7 + sweeps + a CPU smoke config
tests/                 289 tests, CPU only
docs/                  RUNBOOK, PROTOCOL, CANDIDATE1_POT_TC, EXPERIMENTS, HARDWARE,
                       REPRODUCIBILITY
environment.yml        conda (GPU): conda-forge + torch 2.0.1+cu117 via pip
environment-cpu.yml    conda (CPU): tests, smoke run, analysis
environment-cn.yml     the same, via the Tsinghua mirrors
```

No torchvision, no timm — see `docs/HARDWARE.md` for why that matters on
CUDA 11.4.

# Documentation

* **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** — the sequence above, with every
  diagnostic explained and what to do when each step misbehaves.
* **[`docs/PROTOCOL.md`](docs/PROTOCOL.md)** — the locked protocol, dataset
  provenance warnings, and the blocking-items scoreboard.
* **[`docs/CANDIDATE1_POT_TC.md`](docs/CANDIDATE1_POT_TC.md)** — derivation,
  measured weight profiles, the two design corrections, failure modes.
* **[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)** — the ablation ladder and
  the rationale for each arm.
* **[`docs/HARDWARE.md`](docs/HARDWARE.md)** — CUDA 11.4 / 12 GB specifics, the
  memory ladder, proxies and mirrors, the time budget.
* **[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)** — what "reproduced"
  has to mean, and how to check it.

# What is verified, and what is not

Verified by running it, on CPU, against synthetic data shaped like the PraNet
distribution: the 289 tests; the smoke run; `prepare_data --link/--check`;
`freeze_manifest`; `verify_manifest` against both a symlink and the real path;
`hash_collisions`; a 3-arm × 3-seed `run_ablation`; `evaluate --compare`; and
`analyze` including the verdict. Under both NumPy majors.

Not verified here, because this machine has neither: **`conda env create`** (no
conda installed), anything needing a **GPU**, and anything needing the **real
datasets**. The acceptance-band numbers in step 6 are quoted from the
literature — reproducing them is your step 6, and it is the whole point of it.

# Acknowledgements

The Polyp-PVT decoder and the PVTv2 backbone are adapted from
<https://github.com/DengPingFan/Polyp-PVT>; PraNet and Res2Net-v1b follow
<https://github.com/DengPingFan/PraNet> and the Res2Net authors' release.
Changes to those files are listed in their module docstrings.
