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
- **Never read C1 as "mean ≤ 0 and CI straddles zero".** It is "not
  significantly positive": a significant ETIS *loss* triggers it, and so does
  a positive mean whose interval includes zero. The earlier form let both pass.

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

**The training labels are the user's own, re-annotated set (decided
2026-11-04).** `TrainDataset/masks/` and `ValidationDataset/gt/` hold improved
masks (mode `1`), and their positive fractions differ from the distributed
labels both ways, by up to 0.47 of an image. That is intentional, not drift.
Two consequences: the published Polyp-PVT numbers are **not** a reference for
any run on this split, and every comparison model must be retrained on this
same manifest before it is compared. The five test splits are untouched and
still match the distributed labels pixel for pixel.

**Layout that must not be renamed** — other repositories read it:
`TrainDataset/images -> gt` (the photos, under a second name) and, in every
test split, `masks -> gt` (the labels, under a second name). The mask resolver
looks through symlinks, so `gt/` beside an `images` link to it is never read
as ground truth, and `prepare_data.py --check` refuses any mask directory that
decodes to photographs. If `TrainDataset` ever reports "Reading gt/", stop.

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
| `w_floor_frac` | share of the pooled weights the floor lifted to zero. At ξ̂ ≈ −0.5 expect 0.2–0.35: A1 is then weighting the top two-thirds of the pool, and the card's "weights rise with rank" claim holds on that part only. Report it. |

Timing: ~65 s/epoch at batch 16 over 1288 images on a 3080 Ti (fp16), so ~2 h
per 100-epoch run, ~18 h for three seeds of A0+A1+A2. Determinism: a fixed seed
reproduces a run **to tolerance, not bit-exactly** — bilinear upsample backward
has no deterministic CUDA kernel. Quote a tolerance, never equality.

## The gate

The published-number gate in the README does not apply here: the training
labels are re-annotated, so no run on this manifest can land in the Polyp-PVT
band and a miss says nothing about the harness. The harness check is instead
A0 against the **retrained** comparison models on the same manifest, and the
internal A0-vs-A1 contrast, where both arms read identical data. Record this
in any table caption: "trained on the re-annotated 1288/162 split; published
rows are not comparable".

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
