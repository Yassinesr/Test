# Reproducing, and what "reproduced" has to mean

## 0. The bar

A reproduction is not "I ran the code and got a number". It is: **the harness
lands within ±0.5 mDice of Polyp-PVT's own table AND of the independent
re-evaluation, on all five test sets.** Outside that band you are measuring
the harness, not the method, and nothing downstream is interpretable.

Targets, from the brief's §2.1 (compare against **`dice_sweep`**, the
threshold-averaged number — the published values are threshold-averaged, and
comparing them to a fixed-0.5 number compares two different quantities):

| dataset | Polyp-PVT paper | independent re-evaluation |
|---|---|---|
| Kvasir | 0.917 | — |
| CVC-ClinicDB | 0.937 | — |
| CVC-ColonDB | 0.808 | 0.811 |
| CVC-300 | 0.900 | 0.904 |
| ETIS-LaribPolypDB | 0.787 | 0.790 |

The reproduction gap between those two columns is ≤ 0.4 points, which is what
makes them usable as a target at all.

## 1. Setup

```bash
git clone <this repo> && cd Test
conda env create -f environment.yml    # environment-cpu.yml with no GPU,
                                       # environment-cn.yml behind the TUNA mirrors
conda activate polyptail
pytest -q                              # 304 tests, ~25 s on CPU
```

`environment.yml` pins `torch==2.0.1+cu117` and caps NumPy below 2.0 (torch
2.0.1 predates NumPy 2 C-API support). Pinning the environment is part of the
reproduction, not housekeeping: every run records its resolved torch, NumPy,
CUDA and cuDNN versions in `environment.json`, and a torch-version change is
enough to move the last decimals.

Data and backbones, from inside the environment. If the datasets are already
on the machine — a Polyp-PVT checkout, a shared volume — point at them rather
than fetching a second copy:

```bash
python tools/prepare_data.py --link ../Polyp-PVT/dataset   # or --root <path>, or data.root=<path>
python tools/prepare_data.py --check
```

Only if you have no copy: `python tools/prepare_data.py --download --pretrained`.

`--check` validates `./dataset/` against what the reference implementation
hard-codes and names the actual problem — a renamed split, an archive unzipped
one level too deep, an unpaired mask, an extension the reference silently
skips. Run it until it is clean; everything after this assumes it is. See
`docs/RUNBOOK.md` step 3 for the layout diagram.

## 2. Freeze and audit the data — before any training

```bash
python tools/freeze_manifest.py --root ./dataset --out manifests/pranet_protocol.json
python tools/hash_collisions.py --root ./dataset --manifest manifests/pranet_protocol.json \
    --out manifests/collisions.json
git add manifests/ && git commit -m "Freeze dataset manifest and publish collision matrix"
```

Commit both. They are the protocol, and they are two of the blocking items in
§6.2 of the brief that the literature currently does not supply. Read
`docs/PROTOCOL.md` §2.2 for how to act on a non-empty collision matrix.

## 3. Reproduce the baseline

```bash
python tools/run_ablation.py --configs configs/a0_baseline.yaml --seeds 0 1 2 --out-dir runs/main
```

Then check the band:

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

If a split is out of band, the causes are, in order of how often they are the
answer:

1. **The pretrained backbone did not load.** Strict loading should have raised,
   but check `train.log` for the `keys loaded` line.
2. **You compared `dice` against a published number.** Use `dice_sweep`.
3. **The manifest does not match the distributed archives.**
   `tools/freeze_manifest.py` prints count mismatches; re-read its output.
4. **`optim.structure_variant`.** `legacy` reproduces the released artefact;
   `weighted` implements what the paper describes and gives different numbers.
5. **AMP.** Try `optim.amp=off` for one seed as a numerics check.

Nothing else in this repository is worth running until A0 lands in the band.

## 4. Run the candidate and its decisive ablation

```bash
python tools/run_ablation.py \
    --configs configs/a1_pottc.yaml configs/a2_cvar.yaml configs/a3_ohem.yaml \
    --seeds 0 1 2 --out-dir runs/main \
    --auto-budget a0_baseline a1_pottc

python tools/analyze.py --runs runs/main --baseline a0_baseline --method a1_pottc \
    --a2 a2_cvar --seeds 0 1 2 --out reports/main.md
```

`--auto-budget` prints the epoch count that equalises A1's wall-clock for A5.
`analyze.py` produces the paired intervals and evaluates the pre-registered
falsification criterion mechanically. If it says REJECT, the candidate is
rejected.

## 5. What every run leaves behind

```
runs/main/<arm>/seed<k>/
  config.yaml              fully resolved config, including every default
  environment.json         python/torch/CUDA/GPU/cudnn versions, git commit and dirty flag
  model_info.json          parameter count and evaluation head
  train.log                full console log
  train_metrics.jsonl      per-epoch losses and tail diagnostics (u, xi, beta, q_hat, clamp rate)
  per_image_deficits.csv   which training images populated the tail, per epoch
  results.json             per-dataset aggregates, pooled external, wall-clock
  per_image_metrics.json   every per-image score, keyed by stem
  predictions/<split>/     8-bit probability maps at native resolution
  last.pth                 weights plus the tail estimator's state
```

Re-scoring a checkpoint must give the same numbers as the run that produced it:

```bash
python tools/evaluate.py --run runs/main/a1_pottc/seed0 --compare
```

If that fails, the evaluation path is not deterministic and nothing downstream
is trustworthy.

## 6. Known sources of run-to-run variation

* **cuDNN autotuning** is on by default (`run.deterministic: false`). Two runs
  at the same seed will differ in the last decimals. Set
  `run.deterministic: true` for bit-comparable runs, at 10-20% throughput.
* **AMP** changes numerics. Keep it constant across arms you intend to compare —
  every config here inherits it from `base.yaml` for that reason.
* **Dataloader workers** are seeded from the run seed via `worker_init_fn`, so
  `num_workers` does not change the data order. It does change nothing else
  either; vary it freely for throughput.
* **The tail estimator is stateful.** `last.pth` carries the buffer, the
  threshold and the step counter, so resuming resumes the estimator rather than
  silently restarting its warm-up.
