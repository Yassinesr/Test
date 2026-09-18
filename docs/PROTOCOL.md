# The locked protocol

This file is the contract. Every number this repository produces is produced
under it, and any deviation is either configurable and logged, or refused.

It follows §1.5 of the research-verification brief, and exists because that
brief establishes something uncomfortable: **the canonical polyp protocol is
not well defined by its own sources.**

---

## 1. Why a written protocol is needed at all

Five findings from the brief, each of which independently breaks cross-paper
comparison:

| | Finding | Consequence here |
|---|---|---|
| §1.1 | PraNet's paper says 80/10/10; PraNet's repository ships 900+550 train and 100/62/380/60/196 test. The two do not describe the same experiment. | The **repository manifest** is the protocol. The paper text is not reproducible from it. |
| §1.2 | CVC-ColonDB's originating paper documents **300** images; the released test directory holds **380**. File-level identity was never demonstrated. | `tools/freeze_manifest.py` prints the mismatch instead of silently accepting it. |
| §1.2.1 | CVC-300 (60 images) is documented as a partition of a 300-image CVC-ColonDB component. The two "independent external" test sets may overlap. | `tools/hash_collisions.py` settles it, and the pooled external average is not quotable until it has. |
| §1.3 W5 | One frozen model on one frozen test set is reported as 0.709 / 0.712 / 0.715 / 0.716 mDice by four different papers. | **The cross-paper noise floor is 0.3-0.8 mDice.** Any improvement read across papers below ~1 point is uninterpretable. |
| §1.3 W6 | Widely-copied comparison rows contain mIoU > mDice, which is arithmetically impossible for the same mask. | Third-party rows are untrusted. The only defensible comparison target is a baseline you reproduced yourself. |

The operational conclusion, which this repository is built around: **compare
against your own reproduced baseline on your own frozen manifest, on the same
seeds, and never against a published number.**

---

## 2. Data

### 2.1 Freeze it

```bash
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
git add manifests/pranet_protocol.json manifests/pranet_protocol.sha256
```

The manifest records, per image/mask pair: the relative paths, the pairing
stem, the SHA-256 of the **file bytes**, the SHA-256 of the **decoded pixels**,
the size and mode, and the positive fraction of the binarised mask.

The two hashes answer different questions. A file hash changes when anyone
re-saves the archive; a pixel hash changes only when the picture changes. A
mismatch in the first with agreement in the second means "someone recompressed
it"; a mismatch in both means "the data is different". Reporting them
separately is the difference between a five-minute check and a day of doubt.

Paths are stored **relative to the root**, never absolutely. So a manifest
frozen against `./dataset` verifies unchanged against `/srv/data/polyp`, or
against a symlink into a Polyp-PVT checkout, or on someone else's machine. Two
repositories can share one copy of the data without either of them holding a
weaker protocol for it, and moving the data later does not invalidate a frozen
run.

Expected counts, from the PraNet-distributed archives:

| split | pairs |
|---|---|
| `TrainDataset` | 1450 (900 Kvasir-SEG + 550 CVC-ClinicDB) |
| `TestDataset/Kvasir` | 100 |
| `TestDataset/CVC-ClinicDB` | 62 |
| `TestDataset/CVC-ColonDB` | 380 |
| `TestDataset/CVC-300` | 60 |
| `TestDataset/ETIS-LaribPolypDB` | 196 |

A mismatch is not automatically an error — §1.2 documents a real unresolved
300-vs-380 discrepancy — but it must be seen, explained, and reported.

Note: the 900/550 split is a **PraNet artefact**. Neither Kvasir-SEG nor
CVC-ClinicDB publishes an official split, and neither publishes patient-level
groupings, so nothing in the protocol prevents consecutive frames of one
sequence from landing on both sides.

### 2.2 Audit it for duplicates

```bash
python tools/hash_collisions.py --root ./dataset \
    --manifest manifests/pranet_protocol.json --out manifests/collisions.json
```

Three things come out, in increasing order of how much they license:

1. **Flag counts** — a screening signal, threshold-dependent. Colonoscopy
   frames share a dark circular vignette, so genuinely distinct frames collide
   more often than in a natural-image corpus. Never act on a count alone.
2. **Nearest-neighbour distance distributions** — threshold-free. An
   overlapping pair of splits shows a spike of near-zero nearest-neighbour
   distances that no threshold choice can hide; a disjoint pair is unimodal
   near 32, the expectation for independent 64-bit hashes.
3. **Exact decoded-pixel duplicates** — admit no interpretation at all.

Act on the result:

* **CVC-300 ∩ CVC-ColonDB non-empty** → report CVC-300 and
  CVC-ColonDB-minus-CVC-300 separately, and do not quote a pooled external
  average.
* **TrainDataset ∩ any test split non-empty** → that test set's numbers are
  contaminated. Say so in the table, in the caption, every time.
* **Empty** → that is a publishable negative result. It is currently absent
  from the literature, and it is the precondition for calling the five test
  sets independent.

### 2.3 Verify it before every reportable run

`data.verify: hash` re-checks every SHA-256 before training starts and refuses
to run on drift. It costs about a minute on 2248 files. `exists` only checks
presence; `off` checks nothing. Reportable runs use `hash`.

---

## 3. Training

The released Polyp-PVT recipe, on the frozen manifest: 352×352, batch 16,
AdamW at lr 1e-4 / weight decay 1e-4, 100 epochs, multi-scale {0.75, 1, 1.25}
as three separate optimiser steps per batch, no augmentation, gradient
**value** clipping at ±0.5.

Four places where the released code and its paper disagree, all of which
affect whether a "reproduction" lands within ±0.5 mDice:

1. **Epoch count.** `for epoch in range(1, opt.epoch)` with `--epoch 100` runs
   **99** epochs. `optim.epochs` here means what it says; set 99 to match the
   artefact bit-for-bit. At constant LR the difference is well under 0.1 mDice.
2. **Learning-rate schedule.** `Train.py` advertises `--decay_epoch 50` but
   calls `adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)`, so within 100 epochs
   the decay never fires: the published recipe is a **constant** 1e-4. Its
   `adjust_lr` also multiplies the *current* lr rather than rescaling the
   initial one, which compounds whenever it does fire. Default here:
   `decay_epoch: 200`, and the non-compounding formula.
3. **The weighted BCE is not weighted.** The released
   `binary_cross_entropy_with_logits(pred, mask, reduce='none')` passes the
   truthy string `'none'` to a deprecated *boolean* argument, so PyTorch
   resolves it to `reduction='mean'` and returns a **scalar**; the subsequent
   `(weit * wbce).sum(dim=(2,3)) / weit.sum(dim=(2,3))` then reduces to that
   scalar exactly. The published numbers come from **plain mean BCE**.
   `optim.structure_variant: legacy` (the default) reproduces that;
   `weighted` implements what the paper describes. Say which you used.
4. **Model selection.** The released `Train.py` evaluates all five test sets
   every epoch and checkpoints on the best test mDice. That is selection on
   the test set. It is not implementable here: `run.select` accepts `last`, or
   `val_dice` against a fold held out of the **training** split.
5. **The selection set does not exist.** That same checkpointing call is
   `test(model, test_path, 'test')` — i.e. `./dataset/TestDataset/test/`, a
   directory the distributed archive does not contain. The released training
   script cannot finish its first epoch on the released data: every user has
   to invent that directory, and whatever they put in it silently becomes the
   selection criterion. So "the Polyp-PVT training recipe" is not merely
   test-set-selecting, it is *under-specified about which test set*, and two
   good-faith reproductions can differ on it without either noticing.
   `tools/prepare_data.py --check` reports the directory's absence as a note,
   because here its absence is correct.

Augmentation: the released default is **none** (`opt.augmentation` defaults to
the boolean `False` and is compared against the string `'True'`, so the
augmentation branch is unreachable from the CLI). `data.augment: none` matches.

---

## 4. Evaluation

One implementation, `polyptail/eval/`, two scoring semantics, both reported:

### `fixed` — the primary, locked number

sigmoid → bilinear resize **up to the ground-truth grid** → threshold 0.5.
No min-max normalisation. Per-image mean across the split ("m" as in mDice).

The resize direction is not a detail. Downsampling the ground truth to 352
inflates scores on the high-resolution sets — ETIS at 1225×966 and CVC-ColonDB
at 500×574 — which are exactly the two sets this project is about.

No smoothing constant is added to the metric numerator. A smoothing epsilon
turns a total failure into a small positive score and quietly lifts the very
left tail that POT-TC is trying to move.

### `sweep_minmax` — the compatibility number

The PraNet MATLAB toolbox semantics that produced essentially every published
table in the brief's §2: min-max normalise the probability map, quantise to
8 bits (it is written to a PNG), score at all 256 thresholds `t = k/255`, and
average Dice/IoU **over thresholds**. Computed exactly, via two 256-bin
histograms and a reverse cumulative sum.

**Compare reproductions against `dice_sweep`, not `dice`.** Polyp-PVT's 0.917
Kvasir is a threshold-averaged number. Comparing it to a fixed-0.5 number is
comparing two different quantities.

Min-max normalisation is not a neutral rescaling: it forces every map to span
[0,1], so a confidently-empty prediction is stretched into a confident one.
That is one reason compatibility-mode numbers look better than fixed-threshold
ones, and one reason the fixed mode is the one that describes deployment.

### Metrics reported per dataset

`mDice`, `mIoU`, `Recall`, `Precision`, `Specificity`, `HD95`, plus the
threshold-averaged `dice_sweep` / `iou_sweep`, plus the left-tail diagnostics
that a mean hides: the **fraction of images with Dice ≤ 0.5**, the 5th and
10th percentiles of per-image Dice, the per-image standard deviation, and the
count of empty predictions.

HD95 follows the `medpy.metric.binary.hd95` convention: pool the directed
surface distances both ways, take the 95th percentile of the pooled set (not
the max of the two directed 95th percentiles — they differ, and papers rarely
say which). When a prediction is empty the distance is undefined; rather than
dropping the image — which would make HD95 optimistic on exactly the failures
that matter — it is reported at the image diagonal and counted in
`n_hd95_undefined`.

### No test-time augmentation

Not in the primary table, not anywhere. If you add it, it is a separate,
labelled row.

---

## 5. Statistics

* **Three seeds minimum.** Report per-dataset mean **and standard deviation**
  for the baseline too: without the baseline's own seed variance, a 1-point
  gain is uninterpretable.
* **Paired intervals**, computed three ways because they answer different
  questions — `seed_t` (paired t over seeds; wide, 2 df at three seeds, and
  the one that licenses a claim about *methods*), `image_bootstrap` (over
  images, conditional on the seed set), and `hierarchical` (resamples both).
* **Holm-adjusted p-values** across the five datasets. Five datasets with
  two-sided 95% intervals gives a family-wise false-positive rate near 23%.
* **Per-dataset non-degradation.** No dataset may fall materially while the
  pooled average rises.
* **Retained prediction maps** for every run, as 8-bit un-normalised
  probability maps, so a third party can recompute either scoring mode.

`tools/analyze.py` produces all of this and evaluates the pre-registered
falsification criterion mechanically.

---

## 6. What is still missing before any superiority claim

From §6.2 of the brief. Items 1–4 and 9–10 are discharged by this repository's
tooling; the rest require GPU time you have to spend.

| # | Item | Status |
|---|---|---|
| 1 | Published frozen manifest with SHA-256 | tooling ready — `tools/freeze_manifest.py` |
| 2 | Published perceptual-hash collision matrix | tooling ready — `tools/hash_collisions.py` |
| 3 | Resolve the 300-vs-380 CVC-ColonDB discrepancy | partially — the count check flags it; file-level comparison against the CVC originating release needs that release |
| 4 | One published evaluation implementation, validated by reproducing Polyp-PVT to ±0.5 mDice | tooling ready — `polyptail/eval/`; **the validation run is yours to do** |
| 5 | Polyp-PVT reproduced, 3 seeds, with sd | **run A0** |
| 6 | The baseline's own per-dataset seed variance | falls out of item 5 |
| 7 | The decisive ablation (A2: GPD vs empirical quantile) | **run A2** |
| 8 | Paired 95% CIs over ≥3 seeds, per dataset | `tools/analyze.py` |
| 9 | Retained prediction masks and evaluation code | done — every run writes `predictions/` |
| 10 | Recall and HD95 alongside Dice/IoU | done |
| 11 | A reproduced, protocol-matched value for each "best reported" cell | **not attempted.** Two of the five cells in the brief's §2.2 come from a table containing arithmetically impossible entries. Until those are re-derived from released weights there is no trustworthy ceiling to beat. |

Until items 1–11 exist, the only defensible statement about POT-TC is the one
the brief gives it: *a mechanism with documented source-field evidence, a
bounded negative prior-art finding, a clear falsification test, and no
measured effect on polyp segmentation.*
