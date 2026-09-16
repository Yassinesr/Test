# Running this on a 12 GB RTX 3080 Ti with CUDA 11.4

## 1. Install

CUDA 11.4 means a ~470 driver. CUDA **minor version compatibility** applies
inside the 11.x series: any `cu11x` PyTorch build runs on a driver ≥ 450.80.02,
and sm_86 (Ampere, which the 3080 Ti is) has been natively compiled into every
CUDA build since 11.1. So you are not restricted to `cu113`.

```bash
python -m venv .venv && source .venv/bin/activate

# Recommended: torch 2.0.1 + cu117. Newer kernels, same driver requirement.
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117

# Conservative alternative if anything looks odd:
# pip install torch==1.13.1 --index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt
```

There is **no torchvision and no timm dependency**. Images are decoded and
transformed through Pillow, and the PVTv2 backbone vendors the three timm
helpers it needs (`to_2tuple`, `trunc_normal_`, `DropPath`). That is deliberate:
in the torch 1.12–2.0 range that CUDA 11.4 pins you to, timm and torchvision
compatibility tables are a recurring source of silent breakage, and `timm`
moved `timm.models.layers` to `timm.layers` in a way that breaks the official
Polyp-PVT source outright.

Confirm the GPU is actually visible before anything else:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The code handles both torch generations transparently: `weights_only` in
`torch.load` (added in 1.13) and the `GradScaler` spelling change (2.4) are
both version-guarded in `polyptail/utils/io.py`.

## 2. Pretrained weights

```bash
mkdir -p pretrained_pth
# PVTv2-B2 for Polyp-PVT: from the Polyp-PVT release
#   https://github.com/DengPingFan/Polyp-PVT  -> pretrained_pth/pvt_v2_b2.pth
# Res2Net-50-v1b for the PraNet control (A7 only):
#   https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
```

Both loaders are **strict**: a checkpoint that does not line up raises instead
of loading nothing. The reference implementation filters mismatched keys
silently, so pointing it at the wrong file trains from scratch and never says
so — which produces a "reproduction gap" that has nothing to do with the
method.

## 3. Will batch 16 fit in 12 GB?

Measure it, do not guess — it depends on the multi-scale schedule, and the
1.25 scale (448×448, about 1.6× the activation volume of 352×352) is the step
that decides whether you OOM on epoch 1 or on epoch 40:

```bash
python tools/check_memory.py --config configs/a1_pottc.yaml
```

Defaults are set for this card: `optim.amp: fp16` is on, which roughly halves
activation memory. If a scale does not fit, the tool prints this ladder, in
order of preference:

| rung | change | effective batch | comparability |
|---|---|---|---|
| 1 | `optim.amp=fp16` | 16 | unchanged |
| 2 | `optim.batch_size=8 optim.grad_accum=2` | 16 | unchanged |
| 3 | `optim.batch_size=4 optim.grad_accum=4` | 16 | unchanged |
| 4 | `optim.scales=[0.75,1.0]` | 16 | **changes the recipe — say so** |

**Never lower `batch_size` without raising `grad_accum`.** That changes the
effective batch and makes the run incomparable to the reproduced baseline,
which is the only comparison this project licenses.

Gradient accumulation and the tail term interact in one way worth knowing:
the tail module is called once per micro-batch, so `buffer_size` and
`ema_momentum` are in units of *tail calls*, not optimiser steps. With
`grad_accum=2` the buffer fills twice as fast in step terms. The gradient
magnitude is unaffected (`rescale_live: true` normalises it).

`optim.amp: bf16` is also available and is numerically better behaved than
fp16 (no loss scaling), but it is slower than fp16 on Ampere consumer cards
and needs torch ≥ 1.10. Use `off` for an exact-fp32 reference run.

## 4. Time budget

One arm is 100 epochs × 91 steps × 3 scales. On a single 3080 Ti with fp16,
budget **roughly 8–14 hours per run**, so the full three-seed A0 + A1 + A2 set
is about a week of wall-clock. Plan accordingly:

```bash
# The two arms that matter most, three seeds each. Sequential on purpose:
# runs do not fit side by side on 12 GB, and interleaving them would make the
# wall-clock measurement that A5 depends on meaningless.
python tools/run_ablation.py \
    --configs configs/a0_baseline.yaml configs/a1_pottc.yaml \
    --seeds 0 1 2 --out-dir runs/main
```

The runner skips any `(config, seed)` that already has a `results.json`, so an
interrupted sweep resumes by re-running the same command.

Cheaper sanity passes before you commit a week:

```bash
# 1. Does the whole pipeline run at all?  ~30 seconds on CPU.
python tools/make_smoke_data.py --out ./_smoke_data
python tools/train.py --config configs/smoke.yaml

# 2. Does one real epoch run, and how long does it take?
python tools/train.py --config configs/a1_pottc.yaml \
    optim.epochs=1 eval.hd95=false run.name=timing_probe
```

Multiply the reported `epoch_seconds` by 100 for the real estimate. Note that
final evaluation with HD95 on all five test sets adds a few minutes on its own
(ETIS is 196 images at 966×1225, and the distance transform is CPU-bound).

## 5. Practical notes

* `data.verify: hash` re-hashes 2248 files at every run start, about a minute.
  Keep it on for reportable runs; use `exists` while iterating.
* `run.deterministic: true` disables cuDNN autotuning and costs roughly 10-20%
  throughput. Worth it for the exact-reproduction check, not for the sweeps.
* `num_workers: 4` is reasonable on a desktop; if the GPU shows gaps in
  `nvidia-smi dmon`, raise it. Decoding 1920×1072 Kvasir PNGs is not free.
* Each run writes its own `predictions/` directory: about 800 PNGs per run at
  native resolution, roughly 100–300 MB. Three seeds × three arms is a few GB.
  Budget the disk, and do not commit them.
