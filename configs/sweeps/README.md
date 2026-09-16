# Sweeps (ablation A4)

`lambda` and `p` are the only two hyper-parameters POT-TC introduces that
plausibly matter. Both are swept **on a fold held out of the training split**
(`data.val_frac: 0.1`, `run.select: val_dice`) and never on the test sets.
Selecting on the test sets is what the released `Train.py` does and it is the
single easiest way to manufacture a result that does not replicate.

Run a sweep, pick the value, then re-run the chosen setting with
`data.val_frac: 0.0` and `run.select: last` for the reported table, so that the
reported run uses all 1450 training images exactly like the baseline.

```bash
# lambda: how hard the tail term pulls
for lam in 0.1 0.3 1.0; do
  python tools/train.py --config configs/a1_pottc.yaml \
    data.val_frac=0.1 run.select=val_dice \
    pot.tail.lam=$lam run.name=sweep/lam$lam run.seed=0
done

# p: what fraction of images counts as "the tail"
for p in 0.05 0.15 0.30; do
  python tools/train.py --config configs/a1_pottc.yaml \
    data.val_frac=0.1 run.select=val_dice \
    pot.tail.p=$p run.name=sweep/p$p run.seed=0
done
```

`alpha` stays at 0.02 throughout. It sets *how far past the data the GPD
extrapolates*, so sweeping it would confound "the method works" with "this
extrapolation distance happens to suit this dataset" -- and it is the one
number that ablation A2 is defined against.

Two diagnostics to read alongside the val score, from `train_metrics.jsonl`:

* `tail_xi_clamped` must stay near 0. If the shape clamp binds, the weighting
  is not the one the method describes (see `polyptail/evt.py`).
* `tail_n_live_active` tells you how many images per step actually receive
  tail gradient. If it is near zero, `p` is too small for your batch size and
  the term is mostly inactive rather than mostly gentle.
