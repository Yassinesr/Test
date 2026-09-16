# The experiment ladder

Run in this order. Each rung is a precondition for the next; skipping one does
not save time, it just moves where the result becomes uninterpretable.

## 0. Before any GPU time

```bash
pytest -q                                                  # ~25 s, CPU
python tools/make_smoke_data.py --out ./_smoke_data
python tools/train.py --config configs/smoke.yaml          # ~30 s, CPU
python tools/check_memory.py --config configs/a1_pottc.yaml
```

Then freeze and audit the data (`docs/PROTOCOL.md` §2). Commit
`manifests/`.

## 1. A0 — reproduce the baseline

```bash
python tools/run_ablation.py --configs configs/a0_baseline.yaml --seeds 0 1 2 --out-dir runs/main
```

**Acceptance gate:** `dice_sweep` within ±0.5 mDice of Polyp-PVT's own table
*and* of the independent re-evaluation, on all five test sets. See
`docs/REPRODUCIBILITY.md` §3 for the check script and the five things that
usually explain a miss.

Nothing below this line is worth running until A0 passes. The reason is not
pedantry: §1.3 W5 of the brief puts the cross-paper noise floor at 0.3-0.8
mDice, so a harness that is 1 point off is already producing differences larger
than anything POT-TC might plausibly add.

This run also produces blocking item 6 — the baseline's own per-dataset seed
standard deviation — which is what makes any later delta interpretable.

## 2. A1 + A2 — the candidate and the ablation that can kill it

```bash
python tools/run_ablation.py \
    --configs configs/a1_pottc.yaml configs/a2_cvar.yaml \
    --seeds 0 1 2 --out-dir runs/main --auto-budget a0_baseline a1_pottc

python tools/analyze.py --runs runs/main --baseline a0_baseline --method a1_pottc \
    --a2 a2_cvar --seeds 0 1 2 --out reports/main.md
```

A2 replaces the GPD return level with the **empirical** `(1−α)` tail average
over the same exceedance pool — same threshold, same buffer, same schedule,
same `λ`. If A1 − A2 has a CI containing zero on every external dataset, the
extrapolation contributes nothing and POT-TC is a re-parameterisation of CVaR.
That is rejection condition **C2**, and it is cheap, so run it early.

Report A2's update count alongside its score. At `α = 0.02` the empirical tail
is the top 13% of the pool and a batch of 16 supplies about two exceedances, so
A2 gets a gradient on roughly a third of steps while A1 gets one on nearly all
of them. That gap in effective sample size *is* the extreme-value argument, so
also run the cross-check that A2 is not merely losing on update count:

```bash
python tools/run_ablation.py --configs configs/a2_cvar.yaml --seeds 0 1 2 \
    --out-dir runs/main pot.tail.cvar_selection=live
```

### Read the tail diagnostics, not only the score

From `train_metrics.jsonl`, per epoch:

| field | what a bad value means |
|---|---|
| `tail_xi_clamped` | must stay near 0. If it binds, the weighting is not the advertised one — the trainer also warns above 20%. |
| `tail_xi` | the fitted shape. Its distance from 0 predicts how far A1 can differ from A3 at all. |
| `tail_degenerate` | the PWM denominator collapsed and the exponential fallback was used. Rare is fine; common means `p` is too small for the batch. |
| `tail_n_live_active` | images per step actually receiving tail gradient. Near zero means the term is inactive, not gentle. |
| `tail_active_frac` | fraction of steps where the term fired at all. |

## 3. A3 — shape versus difficulty

```bash
python tools/run_ablation.py --configs configs/a3_ohem.yaml --seeds 0 1 2 --out-dir runs/main
python tools/analyze.py --runs runs/main --baseline a3_ohem --method a1_pottc \
    --seeds 0 1 2 --out reports/a1_vs_a3.md
```

Uniform weight on the whole exceedance pool: online hard-example mining with
identical plumbing. A1 − A3 isolates the *shape* of the EVT weighting. Test
the sub-hypothesis while you are here: the A1 − A3 gap should track the mean
`tail_xi` distance from zero. If ξ̂ hovers at zero all run and A1 ≈ A3, the
honest conclusion is that on this data the EVT content reduces to hard-example
mining — and that is a publishable negative result, not a failed experiment.

## 4. A4 — sensitivity, on a training-split fold

See `configs/sweeps/README.md`. `λ ∈ {0.1, 0.3, 1.0}` and
`p ∈ {0.05, 0.15, 0.30}`, selected on `data.val_frac: 0.1` with
`run.select: val_dice`, never on the test sets. `α` stays fixed at 0.02: it
sets how far past the data the GPD extrapolates, so sweeping it would confound
"the method works" with "this extrapolation distance suits this dataset", and
it is the one number A2 is defined against.

Re-run the chosen setting with `val_frac: 0.0` and `select: last` for the
reported table, so the reported run uses all 1450 training images exactly like
the baseline.

## 5. A5 — equal compute

POT-TC adds no parameters, so the equal-parameter control is exact by
construction (`tests/test_train_smoke.py` asserts it). The control that has to
be *run* is equal wall-clock:

```bash
# --auto-budget in step 2 printed the epoch count; use it here.
python tools/run_ablation.py --configs configs/a5_equal_budget.yaml \
    --seeds 0 1 2 --out-dir runs/main optim.epochs=<printed value>
```

## 6. A6 — label-noise robustness

```bash
python tools/run_ablation.py --configs configs/a6_capped.yaml --seeds 0 1 2 --out-dir runs/main
```

Read it together with `per_image_deficits.csv`, which names the training images
that populated the tail each epoch:

```bash
python - <<'PY'
import collections, csv, pathlib
rows = list(csv.DictReader(open("runs/main/a1_pottc/seed0/per_image_deficits.csv")))
by_epoch = collections.defaultdict(list)
for r in rows:
    by_epoch[int(r["epoch"])].append((float(r["deficit"]), r["stem"]))
worst = collections.Counter()
for ep, vals in by_epoch.items():
    for _, stem in sorted(vals, reverse=True)[:20]:
        worst[stem] += 1
print("training images most often in the worst 20, and how many epochs:")
for stem, n in worst.most_common(15):
    print(f"  {stem:<20} {n}/{len(by_epoch)}")
PY
```

If a handful of stems dominate every epoch, open those images before believing
anything else in the table. The card lists label-noise amplification as failure
mode #1 and this is the cheapest way to check it.

## 7. A7 — backbone independence

```bash
python tools/run_ablation.py --configs configs/a7_pranet_a0.yaml configs/a7_pranet_a1.yaml \
    --seeds 0 1 2 --out-dir runs/main
python tools/analyze.py --runs runs/main --baseline a7_pranet_a0 --method a7_pranet_a1 \
    --seeds 0 1 2 --out reports/a7_pranet.md
```

PraNet's own recipe (Adam, 20 epochs, Res2Net-50-v1b) rather than Polyp-PVT's.
If POT-TC helps on a PVT encoder and not on a Res2Net one, the claim is "this
helps this architecture", which is much weaker and must be stated that way.

---

## Reading the verdict

`tools/analyze.py` prints a **pre-registered** verdict. The three rejection
conditions and their thresholds live in `polyptail/stats/paired.py`, fixed in
code. If it says REJECT, the candidate is rejected. Editing a threshold after
seeing results converts a pre-registered test into a post-hoc one and the
result stops meaning anything.

A `NOT REJECTED` verdict is also not a positive result. It means the candidate
survived three specific ways of being wrong. The brief's §6.2 lists eleven
blocking items before *any* superiority claim is licensed;
`docs/PROTOCOL.md` §6 tracks which of them this repository discharges and
which need your GPU.
