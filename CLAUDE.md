# POT-TC — working notes for Claude

A **verification study**, not a model-improvement project. The deliverable is a
defensible answer to "does tail calibration help polyp segmentation", including
"no". A negative result that survives the pre-registered criteria is a success;
a positive result obtained by bending the protocol is worthless.

Read `docs/PROTOCOL.md` before touching anything that produces a number.

## Never, regardless of how convenient

- **Never set `data.verify=exists`/`off` to get past a manifest failure.** That
  guard catches real changes to the data. A failure is a finding — classify it
  (see below), don't route around it.
- **Never re-freeze the manifest over changed data** until you know *what*
  changed. `verify_manifest` classifies it: a renamed directory, a re-save with
  the same binary content, or a different positive fraction. The third means
  the labels moved and must not be frozen over.
- **Never select a checkpoint on the test splits.** `run.select` accepts `last`
  or `val_dice` against `data.val_split` / `data.val_frac`. The released
  `Train.py` checkpoints on test mDice; that is the protocol violation this
  project exists to avoid, and it is deliberately not implementable here.
- **Never quote a pooled external average.** CVC-300 is a subset of
  CVC-ColonDB — 58 of 60 are byte-identical, median nearest-neighbour pHash
  distance 0.0. Report CVC-300 and CVC-ColonDB-minus-CVC-300 separately.
- **Never edit the thresholds in `polyptail/stats/paired.py` after seeing
  results.** They are pre-registered. Editing them makes the test post-hoc.
- **Never let two ablation arms differ in selection rule.** `configs/base.yaml`
  sets it once; every arm inherits. A test asserts they agree.
- **Never report `lam` as configured without checking `active`.** The weight a
  run applies is `lam * active`.

## What is already known about this dataset — don't re-derive it

The training pool was re-split by the user: **TrainDataset 1288 +
ValidationDataset 162 = 1450**, the distributed pool. The five test splits are
the distributed ones, untouched.

From `manifests/collisions.json` (committed):

- 76 exact decoded-pixel duplicate groups: 58 span CVC-300 ↔ CVC-ColonDB, 15
  inside CVC-ColonDB, 3 inside ETIS. **None involve training or validation.**
- CVC-ClinicDB test → TrainDataset: 54/62 flagged, median 7.0. Inherent to
  PraNet's frame-wise 550/62 split of video sequences. Caption it every time.
- ValidationDataset → TrainDataset: median 10.0, no exact duplicates. Read
  medians *down a column*, not against 32: ETIS (a separate corpus) sits at
  14.0, so 10.0 is sequence sharing, not a broken partition.

**Open issue:** all 1288 `TrainDataset` masks were converted `L -> 1` and 1287
changed binary content (positive fraction drifted both ways). An undithered
`L -> 1` of a hard-edged mask is pixel-*identical*, so a pixel mismatch there
means soft edges or a dithered convert — information was lost. Confirm which
directory `prepare_data.py --check` reads for TrainDataset before trusting any
training run.

## Commands

```bash
pytest -q                                      # 381 passed, 2 skipped, CPU only
python tools/prepare_data.py --check           # layout; says which mask dir each split reads
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
python tools/hash_collisions.py --root ./dataset \
    --manifest manifests/pranet_protocol.json --out manifests/collisions.json
python tools/doctor.py                         # stdlib only; proxies, CUDA, reachability
python tools/check_memory.py --config configs/a1_pottc.yaml
```

`manifests/` **is committed** — it is the protocol. `.gitignore` says so.
Re-freeze and commit together.

## Reading a run

| field | meaning |
|---|---|
| `clamped` | near 0.00. Above 0.20 the shape gradient is off; raise `pot.tail.buffer_size`. |
| `active` | fraction of steps the tail term fired. Low early is a transient (`u` lags a falling loss). Past warm-up, under 0.5 is warned about. |
| `degenerate` | PWM denominator collapsed. Occasional fine; common means the pool is too small. |
| `xi` | roughly −0.3 to −0.8 for a soft-Dice deficit. |

Timing: ~65 s/epoch at batch 16 over 1288 images on a 3080 Ti (fp16), so ~2 h
per 100-epoch run, ~18 h for three seeds of A0+A1+A2. Determinism: a fixed seed
reproduces a run **to tolerance, not bit-exactly** — bilinear upsample backward
has no deterministic CUDA kernel. Quote a tolerance, never equality.

## The gate

Before any ablation, one A0 run on the whole pool reproduces the published
numbers, because a 1288-image run selected on validation differs from them in
two ways at once:

```bash
python tools/run_ablation.py --configs configs/a0_baseline.yaml --seeds 0 \
    --out-dir runs/gate data.extra_train_splits='["ValidationDataset"]' \
    data.val_split=null run.select=last
```

Check `dice_sweep` (not `dice`) against 0.917 / 0.937 / 0.808 / 0.900 / 0.787,
±0.005 per split. Outside the band you are measuring the harness.

## Conventions

- Develop on `claude/cool-rubin-kd7m6s`; `main` is merged via PR.
- Commit messages say what changed and **why**, including defects found by
  running the code. Past tense, no model identifiers.
- Tests name the behaviour, not the function. A test that cannot be built
  honestly on the fixture gets pinned at the level where it can be stated
  exactly, rather than tuning the fixture until it agrees.
- When a tool reports a problem, it should say which *kind* — the fixes differ.
  `differs (-162)`, `76 duplicates`, `1288 mismatches` were each split into
  classified diagnoses for this reason.
