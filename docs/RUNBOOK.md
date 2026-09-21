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

conda env create -f environment.yml
conda activate polyptail
```

That is the whole install: `environment.yml` brings up Python 3.10 and the
scientific stack from conda-forge, then pulls `torch==2.0.1+cu117` from the
official PyTorch wheel index via pip.

Two things in that file are deliberate and worth knowing, because both cause
confusing first-run failures if you change them:

* **PyTorch comes from pip, not from conda.** PyTorch deprecated its own
  Anaconda channel, so `conda install pytorch` pulls from a frozen archive.
  The pip wheels also bundle their own CUDA runtime, so the cu117 build runs
  against your CUDA 11.4 driver without conda having to reconcile a
  `cudatoolkit` against it. (A pure-conda alternative is in
  `docs/HARDWARE.md` if you prefer it.)
* **NumPy is capped below 2.0.** torch 2.0.1 was compiled against the NumPy
  1.x C API and fails at import under NumPy 2. That ceiling lifts if you move
  to torch >= 2.4, which needs a newer driver than CUDA 11.4 gives you.

CUDA 11.4 means a ~470 driver. Minor-version compatibility applies inside
11.x, so any `cu11x` wheel runs on a driver >= 450.80.02, and sm_86 (your
3080 Ti) has been natively compiled in since CUDA 11.1. If anything looks odd,
swap `torch==2.0.1+cu117` for `torch==1.13.1+cu117` in `environment.yml` and
re-create the environment. Same index; the most conservative combination that
still supports your card.

Prefer pip and a plain virtualenv? `requirements-dev.txt` plus
`pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117`
gives the same environment.

**pip timed out on `download.pytorch.org` at the end?** The conda half already
succeeded; finish the environment instead of re-creating it:

```bash
conda activate polyptail
pip install torch==2.0.1        # or add -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # expect 11.7
```

**Behind a proxy, or upstream blocked?** Run `python tools/doctor.py` first —
it is standard-library only, so it works in conda's `base` before any
environment exists, and it prints the shortest route from whatever state you
are in. `conda env create` failing with
`ProxyError ... Connection refused` means a proxy is configured and dead — a
mirror will not fix that, because you would need the proxy to reach the mirror
too. Triage it first. If upstream is merely slow or blocked and the Tsinghua
mirrors are reachable directly, use `environment-cn.yml` instead. And if a
working environment already exists on the machine (the one you run Polyp-PVT
with), reuse it — this project needs only torch, numpy, scipy, pillow and
pyyaml. All three routes, with the commands, are in
[`docs/HARDWARE.md`](HARDWARE.md#restricted-networks-proxies-and-mirrors).

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

Expect `349 passed` in about 25 seconds. These are CPU-only and need no data.

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

## Step 3 — Point at the data, and check the layout

### If the data is already on this machine

It usually is — a Polyp-PVT checkout next door already has
`./dataset/TrainDataset` and `./dataset/TestDataset`. **Do not download it
again.** Pick whichever of these suits you:

```bash
# A. Link it, so the default config works untouched. Nothing is copied.
python tools/prepare_data.py --link ../Polyp-PVT/dataset

# B. Leave it where it is and check it in place.
python tools/prepare_data.py --check --root ../Polyp-PVT/dataset

# C. No ./dataset at all: override the root per run.
python tools/train.py --config configs/a0_baseline.yaml data.root=/srv/data/polyp-dataset
```

`--link` validates the target **before** creating the symlink, so a wrong path
fails here — where the message is about your dataset — rather than at freeze
time, where it is about a hash. It is idempotent, it refuses to clobber a real
directory, and it refuses to silently repoint an existing link.

With option C, remember `data.root` on *every* command, including
`freeze_manifest.py --root` and `hash_collisions.py --root`. It is recorded in
each run's `config.yaml`, so provenance survives either way.

Both repositories then read the same bytes. That is fine, and it is worth
knowing why: **manifests store paths relative to the root.** A manifest frozen
through the symlink verifies unchanged against the real path, and vice versa,
so sharing one dataset between two checkouts — or moving it later — does not
invalidate a frozen protocol.

The pretrained backbone still has to exist:

```bash
ls -l pretrained_pth/pvt_v2_b2.pth
```

If a Polyp-PVT checkout already has it, link that too:

```bash
mkdir -p pretrained_pth && ln -s ../../Polyp-PVT/pretrained_pth/pvt_v2_b2.pth pretrained_pth/
```

### Only if you have no copy

```bash
python tools/prepare_data.py --download --pretrained
```

Best-effort: Google Drive rate-limits large public files and changes its
confirmation flow periodically. If `gdown` fails, the tool prints the links —
fetch them by hand and re-run the check. Nothing downstream cares how the
bytes arrived.

* **Datasets** — Polyp-PVT README §4.2, Google Drive file
  `1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC` (Baidu mirror code `sydz`).
* **PVTv2-B2** — Polyp-PVT README §4.3, Google Drive folder
  `1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV` (Baidu code `w4vk`).
* **Res2Net-50-v1b** — A7 control only; fetched over plain HTTPS, so it rarely
  needs a manual step.

### Check the layout

```bash
python tools/prepare_data.py --check          # or --root <wherever it lives>
```

This is the part that matters, and it needs no network. It compares what you
have against what the reference implementation *hard-codes* — the five
test-split spellings, the `images/` and `masks/` subdirectories, the file
extensions each reference dataloader accepts — and names the actual problem
rather than failing several steps later:

```
split                               pairs  expected  status
TrainDataset                         1450      1450  ok
TestDataset/Kvasir                    100       100  ok
TestDataset/CVC-ClinicDB               62        62  ok
TestDataset/CVC-ColonDB               380       380  ok
TestDataset/CVC-300                    60        60  ok
TestDataset/ETIS-LaribPolypDB         196       196  ok
```

The target layout:

```
dataset/                       # a directory, or a symlink to one
  TrainDataset/{images,masks}/
  ValidationDataset/{images,masks}/   # optional; see "If you hold out your own
                                      # validation split" below
  TestDataset/Kvasir/{images,masks}/
  TestDataset/CVC-ClinicDB/{images,masks}/
  TestDataset/CVC-ColonDB/{images,masks}/
  TestDataset/CVC-300/{images,masks}/
  TestDataset/ETIS-LaribPolypDB/{images,masks}/
pretrained_pth/
  pvt_v2_b2.pth
```

What it will tell you, in the order these actually happen:

| it says | what happened |
|---|---|
| *unzipped one level too deep* | the archive expanded to `dataset/dataset/TrainDataset/...`; move the inner contents up |
| *`CVC-T/` looks like it: rename to `CVC-300`* | the 60-image split travels under three names in the literature; the reference code hard-codes one |
| *N image(s) have no mask with a matching name* | incomplete unpack; re-download rather than deleting the orphans |
| *pairs differ in size* | the archive did not unpack cleanly |
| *extension the reference dataloader filters out* | your copy works here but the original repo would silently skip those files, so the two are not comparable |
| *N stem(s) in images/ belong to more than one file* | the same image under two extensions. This project pairs by stem and would take whichever sorts last; the reference counts files and its `assert len(images) == len(gts)` fires. Delete the copy you do not want |
| *N split(s) differ from the counts the PraNet archive distributes* | a warning, not an error — §1.2 documents a real 300-vs-380 discrepancy for CVC-ColonDB. Unexplained, not wrong. The `why <split> differs` block underneath says which cause it is; see below |
| *`TestDataset/test/` is absent* | **expected, and not a problem.** See below |

### If a count differs

First: if you held out your own validation split, this is not the section you
want — a `ValidationDataset/` that accounts for the shortfall is reconciled,
not reported, and the status reads `re-split` instead. See below.

Otherwise, the checker does not stop at `differs (-162)`. It has already
listed the directory, so it spends the rest of that listing on telling you
which cause you are looking at:

```
why TrainDataset differs:
  - files on disk: images/ 1288, masks/ 1288; distinct stems: 1288 and 1288
  - both sides agree and every file is paired, so the 162 absent item(s) are
    missing from images/ and masks/ alike. A transfer that stopped
    mid-directory leaves orphans on one side; there are none.
  - names by shape: 900 alphanumeric (= the Kvasir-SEG share exactly), 388 numeric
  - the archive is Kvasir-SEG 900, CVC-ClinicDB 550, and the two corpora are
    named differently, so the group that is short is the one to re-copy.
  - numeric names run 1-388 unbroken -- nothing is missing from inside that
    range, the sequence simply stops at 388.
```

Read it as a decision tree — the four cases have different fixes:

| the diagnosis says | what happened | what to do |
|---|---|---|
| *N image(s) have no mask* (an error, above) | a copy or unzip stopped part-way through one directory | redo that transfer |
| *one contiguous block, A-B* | same, but restarted past the gap | redo that transfer |
| *scattered, e.g. [...]* | something selected files out on purpose | find out what, and why, before you train |
| *runs 1-N unbroken* / a shape group short | that corpus was never fully copied | re-copy that corpus |

`TrainDataset` is the one to care about, because it is two corpora
concatenated — 900 Kvasir-SEG and 550 CVC-ClinicDB — that name their files
differently. The shape histogram therefore tells you which half is short
without your opening a single directory.

**A short training set is not a small problem.** Training on 1288 of 1450
images is a different experiment from the one every number in §1.2 refers to,
and the gap is not a constant offset you can subtract later: the missing
images are a whole corpus's tail, not a random sample, so the model sees a
different mixture of the two domains. Fix the data before Step 4. If you
genuinely cannot — the source is gone, say — then freeze the manifest anyway
and treat every comparison against a published number as unusable; the
candidate-vs-baseline comparison inside this repository remains valid,
because both arms see the same frozen manifest, and that internal comparison
is what C1/C2/C3 are written against.

### If you hold out your own validation split

The distributed archive ships no validation set, which is why the released
`Train.py` ends up checkpointing on the test sets. If you split the 1450-image
training pool yourself, put the held-out part in `dataset/ValidationDataset/`
(`images/` and `masks/`, or `gts/` — both names are read) and everything below
picks it up:

* `prepare_data.py --check` counts it, and reconciles the partition rather than
  reporting the training half as short:

  ```
  TrainDataset                         1288      1450  re-split (-162)
  ValidationDataset                     162         -  held out by you
  ...
  note: TrainDataset (1288) + ValidationDataset (162) = 1450, the size of the
  pool PraNet distributes as TrainDataset.
  ```

  If the two halves do **not** add up, that is a warning with the arithmetic
  in it: images went missing in the split, or were duplicated into both halves.

* `freeze_manifest.py` includes it automatically and says so. This matters
  more than it sounds: the split that chooses which weights you report is the
  one split that must be verifiable, and a manifest entry gives it a SHA-256
  per file.

* `configs/base.yaml` selects on it — `data.val_split: ValidationDataset`,
  `run.select: val_dice` — and every ablation arm inherits that, so no two
  arms are selected by different rules. The test splits are read once, after
  training. With the stock archive and no validation directory, set
  `data.val_split: null` and `run.select: last`.

* The trainer refuses to start if a stem appears in both `TrainDataset` and
  `ValidationDataset`. That is the cheap check; run `tools/hash_collisions.py`
  for the real one, which compares pixels and catches the same frame saved
  twice under two names.

`data.val_frac` remains for datasets with no such directory: it carves the
fold out of the training split at `data.val_seed`. Setting both is an error.
Prefer the directory — a fold is reproducible only while the seed, the
fraction *and* the ordering of the training split all hold still, whereas a
directory is in the manifest.

### One quirk worth knowing

The released `Train.py` line 116 calls `test(model, test_path, 'test')`, i.e.
it requires `./dataset/TestDataset/test/` — a directory the distributed
archive does not contain. The original training script cannot finish its first
epoch on the released data until you create that directory yourself, and
whatever you put in it silently becomes its checkpoint-selection criterion.
Nothing in this project needs it, which is why the checker reports its absence
as a note rather than an error.

If your Polyp-PVT checkout *does* have a `TestDataset/test/` — because you
created one to make the original code run — leave it. The checker recognises
it and says so, and nothing here reads it: it appears in no split list in
`configs/`, so it cannot leak into a result.

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
| numbers differ slightly between identical runs | cuDNN autotuning, plus bilinear upsample backward, which has no deterministic CUDA kernel | expected, and not fully fixable on GPU. `run.deterministic=true` removes the autotuning part at 10-20% throughput; a fixed seed then reproduces a run closely, not bit-exactly. See `docs/REPRODUCIBILITY.md` §6 |
| `UserWarning: upsample_bilinear2d_backward_out_cuda does not have a deterministic implementation` | you set `run.deterministic=true` | expected and harmless — the op keeps its non-deterministic kernel so training can proceed |

Re-scoring a checkpoint must reproduce its run exactly:

```bash
python tools/evaluate.py --run runs/main/a1_pottc/seed0 --compare
```

If that ever fails, the evaluation path is not deterministic and nothing
downstream is trustworthy.
