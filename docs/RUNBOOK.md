# Runbook — from nothing to a verdict

A linear sequence. The other docs are organised by topic; this one is organised
by what you type next. Written for a single 12 GB RTX 3080 Ti on CUDA 11.4.

Each step says what to expect and what to do if it does not happen. **Steps 1-7
are cheap (under an hour total). Step 8 onwards is days of GPU time.** Do not
skip ahead: step 7 is a gate, and if it fails, everything after it is
uninterpretable.

---

## Step 1 — Get the code and an environment

```bash
git clone <your fork or this repo> polyptail && cd polyptail
git checkout claude/cool-rubin-kd7m6s

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

Install torch **first**, for your CUDA, then everything else:

```bash
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements-dev.txt
```

CUDA 11.4 means a ~470 driver. Minor-version compatibility applies inside
11.x, so any `cu11x` wheel runs on a driver >= 450.80.02, and sm_86 (your
3080 Ti) has been natively compiled in since CUDA 11.1. If anything looks odd,
`torch==1.13.1` with the same index URL is the conservative fallback.

**Check:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `2.0.1+cu117 True NVIDIA GeForce RTX 3080 Ti`. If `False`, stop here —
nothing below will work. Usual causes: the driver is older than 450.80.02, or
you installed the CPU wheel by omitting `--index-url`.

---

## Step 2 — Prove the pipeline works before downloading anything

```bash
pytest -q
```

Expect `202 passed` in about 25 seconds. These are CPU-only and need no data.

```bash
python tools/make_smoke_data.py --out ./_smoke_data
python tools/train.py --config configs/smoke.yaml
```

Expect a two-epoch run on synthetic data finishing in under a minute, ending
with a `TestDataset/Fake ... mDice ...` line. The Dice will be poor — a
0.1M-parameter net on 48 images for 2 epochs. That is fine; this step proves
plumbing, not accuracy.

If either fails, fix it now. Debugging the harness is much cheaper before
there are 2248 real images and a GPU in the picture.

---

## Step 3 — Download the data and the pretrained backbone

Both come from the official Polyp-PVT release
(<https://github.com/DengPingFan/Polyp-PVT>):

* **Datasets** — the "Data preparation" link,
  Google Drive file `1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC`
  (Baidu mirror code `sydz`). Unpack into `./dataset/`.
* **Pretrained PVTv2-B2** — the "Pretrained model" link,
  Google Drive folder `1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV`
  (Baidu code `w4vk`). Put `pvt_v2_b2.pth` in `./pretrained_pth/`.

Only if you intend to run A7 (the PraNet control), also:

```bash
mkdir -p pretrained_pth
curl -L -o pretrained_pth/res2net50_v1b_26w_4s-3cf99910.pth \
  https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
```

The layout must end up exactly like this:

```
dataset/
  TrainDataset/{images,masks}/
  TestDataset/Kvasir/{images,masks}/
  TestDataset/CVC-ClinicDB/{images,masks}/
  TestDataset/CVC-ColonDB/{images,masks}/
  TestDataset/CVC-300/{images,masks}/
  TestDataset/ETIS-LaribPolypDB/{images,masks}/
pretrained_pth/
  pvt_v2_b2.pth
```

**Check:**

```bash
for d in dataset/TrainDataset dataset/TestDataset/*; do
  echo "$(ls "$d/images" | wc -l)  $d"
done
```

Expect `1450, 100, 62, 380, 60, 196` (order depends on your shell's glob).
Directory names must match exactly — `CVC-300`, not `CVC-T` or `EndoScene`.

---

## Step 4 — Freeze the data (this is the protocol)

```bash
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
```

This hashes every file twice — the raw bytes, and the decoded pixels — for all
2248 image/mask pairs. Takes a minute or two. It prints a table of counts
against what the PraNet repository distributes:

```
split                                pairs  expected  status
TrainDataset                          1450      1450  ok
TestDataset/CVC-300                     60        60  ok
...
```

**Any `MISMATCH` line is a real finding, not a nuisance.** The brief documents
an unresolved 300-vs-380 discrepancy for CVC-ColonDB, so a mismatch may be
legitimate — but you now have to be able to explain it, and you must say so
whenever you report a number from that split.

The command also fails loudly if an image and its mask disagree in size, or if
a stem exists in `images/` but not `masks/`. Both mean the archive did not
unpack cleanly; re-download rather than working around it.

**Commit the output.** It is what makes every later result auditable:

```bash
git add manifests/pranet_protocol.json manifests/pranet_protocol.sha256
git commit -m "Freeze dataset manifest"
```

---

## Step 5 — Audit for duplicates

```bash
python tools/hash_collisions.py --root ./dataset \
    --manifest manifests/pranet_protocol.json --out manifests/collisions.json
```

A few minutes (it decodes every image three times). You get a matrix of
"images in ROW with a near-duplicate in COLUMN", a table of median
nearest-neighbour perceptual distances, and a list of exact decoded-pixel
duplicates.

**How to read it:**

* Median nearest-neighbour pHash distance **near 32** between two splits means
  they are independent. **Near 0** means they overlap.
* `TestDataset/CVC-300 -> TestDataset/CVC-ColonDB` non-zero → the two
  "independent external" sets are not independent. Report CVC-300 and
  CVC-ColonDB-minus-CVC-300 separately, and do not quote a pooled external
  average. This is the open question from §1.2.1 of the brief.
* `TrainDataset -> TestDataset/<X>` non-zero → that test set is contaminated.
  Say so in every table that includes it.
* An empty matrix is a publishable negative result. It is currently absent
  from the literature.

Flag counts are a screening signal only — colonoscopy frames share a dark
vignette and collide easily. Open `manifests/collisions.pairs.csv` and look at
a few flagged pairs before concluding anything. The exact pixel-hash
duplicates admit no interpretation at all; those are real.

```bash
git add manifests/collisions.json manifests/collisions.pairs.csv
git commit -m "Publish duplicate-collision matrix"
```

---

## Step 6 — Check the configuration fits your card

```bash
python tools/check_memory.py --config configs/a1_pottc.yaml
```

It runs real training steps at each scale in the schedule and reports peak
allocation. The 1.25 scale (448x448) is the one that decides whether you OOM,
and it is the last one tried, so read to the end.

Expect something like:

```
  scale 0.75  -> 264x264, micro-batch 16: peak X.XX GiB
  scale 1.0   -> 352x352, micro-batch 16: peak X.XX GiB
  scale 1.25  -> 448x448, micro-batch 16: peak X.XX GiB
Peak X.XX GiB, headroom Y.YY GiB.
```

If it says **This configuration fits** with more than ~1 GiB of headroom, go
on. Otherwise it prints a ladder; take the first rung that fits:

| rung | change | effective batch |
|---|---|---|
| 1 | `optim.amp=fp16` (already the default) | 16 |
| 2 | `optim.batch_size=8 optim.grad_accum=2` | 16 |
| 3 | `optim.batch_size=4 optim.grad_accum=4` | 16 |

Whatever you pick, **append it to every command from here on**, including the
baseline, or the arms stop being comparable. Never lower `batch_size` without
raising `grad_accum`: that changes the effective batch and breaks the one
comparison this project licenses.

---

## Step 7 — Time one epoch, then decide the budget

```bash
python tools/train.py --config configs/a1_pottc.yaml \
    optim.epochs=1 eval.hd95=false data.verify=hash run.name=timing_probe
```

This does one real epoch and one evaluation pass. Read the per-epoch time:

```bash
python -c "import json; print(json.loads(open('runs/timing_probe/train_metrics.jsonl').readline())['epoch_seconds'], 'sec/epoch')"
```

Multiply by 100 for one full run, then by 9 for three seeds of A0+A1+A2.
On a 3080 Ti with fp16 expect somewhere in the region of 8-14 hours per run —
but **use your measured number, not mine.**

While this runs, check two things in the log:

1. **The backbone loaded.** Look for a line like
   `backbone checkpoint pvt_v2_b2.pth: 350/350 keys loaded, 0 missing, ...`.
   If loading failed you will get a `RuntimeError` naming the mismatched keys
   rather than a silent scratch-training run — that is deliberate. Fix the
   checkpoint path; do not set `strict_pretrained=false` to make it go away.
2. **The tail term engaged.** A line like
   `epoch 1 summary: ... active 0.93 of calls, u=0.31 xi=-0.42 (raw -0.42) ... clamped=0.00`.
   With `warmup_frac: 0.2` the term is still ramping in epoch 1, so a small
   `tail` loss is expected. What matters is `clamped=0.00`.

---

## Step 8 — Run the baseline (this is the gate)

```bash
python tools/run_ablation.py --configs configs/a0_baseline.yaml \
    --seeds 0 1 2 --out-dir runs/main
```

Sequential on purpose: two runs do not fit on 12 GB, and interleaving would
destroy the wall-clock measurement A5 depends on. The runner skips any
`(config, seed)` that already has a `results.json`, so if it dies overnight,
re-run the identical command and it resumes.

Run it under `tmux` or `screen`. Three runs is roughly a day and a half.

---

## Step 9 — Check A0 lands in the band. Stop if it does not.

```bash
python - <<'PY'
import json, pathlib
targets = {"Kvasir": 0.917, "CVC-ClinicDB": 0.937, "CVC-ColonDB": 0.808,
           "CVC-300": 0.900, "ETIS-LaribPolypDB": 0.787}
for seed in (0, 1, 2):
    blob = json.loads(pathlib.Path(f"runs/main/a0_baseline/seed{seed}/results.json").read_text())
    print(f"seed {seed}")
    for split, agg in blob["per_split"].items():
        name = split.split("/")[-1]
        got, want = agg["dice_sweep"], targets[name]
        flag = "ok" if abs(got - want) <= 0.005 else "OUT OF BAND"
        print(f"  {name:<22} {got:.4f}  target {want:.3f}  {flag}")
PY
```

**Every split must be within ±0.5 mDice.** This is not a formality: §1.3 W5 of
the brief shows the cross-paper noise floor is already 0.3-0.8 mDice, so a
harness that is a point off is producing differences larger than anything
POT-TC could plausibly add. Outside the band you are measuring the harness.

Note it compares `dice_sweep`, not `dice`. The published numbers are averaged
over 256 thresholds; `dice` is the fixed-0.5 operating point. They are
different quantities.

If a split is out of band, in the order these usually turn out to be the cause:

1. The pretrained backbone did not load — check `train.log`.
2. You compared against `dice` instead of `dice_sweep`.
3. The manifest counts did not match in step 4.
4. `optim.structure_variant` — `legacy` (default) reproduces the released
   artefact; `weighted` implements what the paper describes and gives
   different numbers.
5. AMP numerics — re-run one seed with `optim.amp=off` as a check.

You now also have the thing the literature does not supply: **the baseline's
own per-dataset seed standard deviation.** Without it, no later delta means
anything.

---

## Step 10 — Run the candidate and the ablation that can kill it

```bash
python tools/run_ablation.py \
    --configs configs/a1_pottc.yaml configs/a2_cvar.yaml \
    --seeds 0 1 2 --out-dir runs/main \
    --auto-budget a0_baseline a1_pottc
```

About three days. `--auto-budget` prints, at the end, the epoch count that
equalises A1's wall-clock for the A5 control later.

Run A2 early rather than late. It replaces the GPD return level with the
*empirical* tail average over the same exceedance pool — same threshold, same
buffer, same schedule, same lambda. If A1 - A2 has a confidence interval
containing zero on every external dataset, the extrapolation contributes
nothing and the candidate is a re-parameterisation of CVaR. That is rejection
condition C2, and it is much better to learn it now than after A3-A7.

### While it runs, watch these

```bash
tail -f runs/main/a1_pottc/seed0/train.log | grep summary
```

| field | what a bad value means |
|---|---|
| `clamped` | must stay near 0.00. Above 0.20 the trainer warns: the shape gradient is off and the weighting is not the advertised one. Raise `pot.tail.buffer_size` first. |
| `xi` | the fitted shape. Expect roughly -0.3 to -0.8 for a soft-Dice deficit. How far it sits from 0 bounds how much A1 can differ from A3 at all. |
| `active` | fraction of steps the term fired. Near 0 means `p` is too small for your batch, not that the term is gentle. |
| `degenerate` | PWM denominator collapsed, exponential fallback used. Occasional is fine; common means the pool is too small. |

---

## Step 11 — Get the verdict

```bash
python tools/analyze.py --runs runs/main --baseline a0_baseline --method a1_pottc \
    --a2 a2_cvar --seeds 0 1 2 --out reports/main.md
```

You get per-dataset tables with per-seed values and three kinds of paired 95%
interval, the left-tail diagnostics POT-TC actually targets (fraction of images
at Dice <= 0.5, the 5th percentile of per-image Dice, HD95), and a
pre-registered verdict:

```
VERDICT: NOT REJECTED   (or REJECT POT-TC)
  [not triggered] C1 no ETIS gain in mean or in tail mass: ...
  [    TRIGGERED] C2 A1-A2 CI contains zero on every external set: ...
  [not triggered] C3 a dataset degrades while the pooled average rises: ...
```

Read the `seed_t` interval for claims about *methods* — seeds are the unit the
claim is about, and with three seeds it is wide by construction. That width is
the honest answer, not a defect to engineer around.

**If it says REJECT, the candidate is rejected.** The thresholds are fixed in
`polyptail/stats/paired.py`. Changing them after seeing the numbers converts a
pre-registered test into a post-hoc one and the result stops meaning anything.

**`NOT REJECTED` is not a positive result either.** It means the candidate
survived three specific ways of being wrong. `docs/PROTOCOL.md` §6 tracks the
eleven blocking items from §6.2 of the brief that gate any superiority claim.

---

## Step 12 — Only now, the rest of the ladder

In descending order of what they buy you:

| arm | command | answers |
|---|---|---|
| **A3** | `--configs configs/a3_ohem.yaml` | is the EVT *shape* doing anything, or is this hard-example mining? |
| **A5** | `--configs configs/a5_equal_budget.yaml optim.epochs=<from --auto-budget>` | is the gain just extra compute? |
| **A6** | `--configs configs/a6_capped.yaml` | is the tail genuine difficulty or label noise? |
| **A4** | see `configs/sweeps/README.md` | how sensitive is it to lambda and p? |
| **A7** | `--configs configs/a7_pranet_a0.yaml configs/a7_pranet_a1.yaml` | is the effect backbone-specific? |

`docs/EXPERIMENTS.md` has the full rationale for each, including the
`per_image_deficits.csv` analysis that tells genuine hard frames from
mis-annotated ones.

---

## If something goes wrong

| symptom | cause | fix |
|---|---|---|
| `FileNotFoundError: manifests/pranet_protocol.json` | step 4 not done | run `tools/freeze_manifest.py` |
| `RuntimeError: dataset does not match the frozen manifest` | files changed since the freeze | re-freeze deliberately, or restore the data. Do **not** set `data.verify=off` to silence it |
| `RuntimeError: ... keys loaded, N missing` | wrong or corrupt backbone checkpoint | re-download; this error exists so you do not train from scratch by accident |
| `torch.cuda.OutOfMemoryError` mid-run | headroom was marginal | drop a rung in step 6, restart the affected seed |
| a seed died overnight | anything | re-run the identical `run_ablation.py` command; completed seeds are skipped |
| numbers differ slightly between identical runs | cuDNN autotuning | expected. `run.deterministic=true` for bit-comparable runs, at 10-20% throughput |

Re-scoring a checkpoint must reproduce its run exactly:

```bash
python tools/evaluate.py --run runs/main/a1_pottc/seed0 --compare
```

If that ever fails, the evaluation path is not deterministic and nothing
downstream is trustworthy.
