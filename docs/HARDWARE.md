# Running this on a 12 GB RTX 3080 Ti with CUDA 11.4

## 1. Install

### Recommended: conda for the environment, pip for torch

```bash
conda env create -f environment.yml
conda activate polyptail
```

`environment.yml` brings up Python 3.10 and the scientific stack from
conda-forge, then installs `torch==2.0.1+cu117` from the official PyTorch
wheel index. The split is deliberate:

* **PyTorch deprecated its own Anaconda channel.** `conda install pytorch`
  now resolves against a frozen archive that receives no updates and may
  eventually be withdrawn. The pip wheels at `download.pytorch.org` are the
  supported distribution path.
* **The pip wheels bundle their own CUDA runtime.** That means the cu117
  build runs against your CUDA 11.4 driver with no `cudatoolkit` package to
  reconcile, and no chance of conda solving to a CUDA build your driver
  cannot load.

CUDA 11.4 means a ~470 driver. CUDA **minor version compatibility** applies
inside the 11.x series: any `cu11x` PyTorch build runs on a driver
>= 450.80.02, and sm_86 (Ampere, which the 3080 Ti is) has been natively
compiled into every CUDA build since 11.1. So you are not restricted to
`cu113`.

**NumPy is capped below 2.0**, and this one bites people. torch 2.0.1 was
compiled against the NumPy 1.x C API and fails at import under NumPy 2 with
*"A module that was compiled using NumPy 1.x cannot be run in NumPy 2"*.
NumPy 2 support arrived in torch 2.4, which wants a newer driver than CUDA
11.4 offers. Keep the ceiling.

If anything misbehaves, `torch==1.13.1+cu117` from the same index is the most
conservative combination that still supports your card — edit the pip block
in `environment.yml` and re-create the environment.

### Restricted networks, proxies and mirrors

**First, separate the two failure modes — they need opposite fixes.**

```
ProxyError: Conda cannot proceed due to an error in your proxy configuration.
  ... Cannot connect to proxy ... [Errno 111] Connection refused
```

"Connection refused" from a *proxy* means a proxy is configured and nothing is
listening on it. That is a broken proxy setting, not a blocked upstream, and a
mirror will not fix it: if you genuinely need a proxy to reach the internet,
you need it to reach the mirror too. Triage before changing anything:

```bash
env | grep -i -E '_proxy|_PROXY'          # stale vars are the usual culprit
conda config --show proxy_servers channels default_channels
cat ~/.condarc 2>/dev/null
cat ~/.netrc  2>/dev/null                 # conda reads this too
```

* **No proxy is actually needed** (the vars are stale, e.g. inherited from a
  `.bashrc` written for a different network):

  ```bash
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
  conda config --remove-key proxy_servers    # only if it is set
  ```

* **A proxy is required.** Set it for conda *and* pip; they do not share
  configuration:

  ```bash
  conda config --set proxy_servers.http  http://user:pass@proxy:port
  conda config --set proxy_servers.https http://user:pass@proxy:port
  pip config set global.proxy http://user:pass@proxy:port
  ```

Note that the error in that traceback is about `repo.anaconda.com/pkgs/main`
and `/pkgs/r` — the Anaconda `defaults` channels, which **this project does not
use**. Conda appends them anyway unless told not to, which is why every
environment file here declares `nodefaults`. If you are on an older checkout,
add it:

```yaml
channels:
  - conda-forge
  - nodefaults
```

That alone removes one whole class of failure, and it sidesteps Anaconda's
commercial terms for the `defaults` channels, which many organisations block
deliberately.

#### Tsinghua (TUNA) mirrors

If upstream is reachable but slow or blocked — and the mirror is reachable
*without* a proxy — use the mirrored environment:

```bash
conda env create -f environment-cn.yml
conda activate polyptail
```

Check the mirrors respond before blaming them:

```bash
curl -sI https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/noarch/repodata.json | head -1
curl -sI https://pypi.tuna.tsinghua.edu.cn/simple/torch/ | head -1
```

`environment-cn.yml` is the same environment as `environment.yml`; only the
sources differ. It has **one substantive difference**: the pin is
`torch==2.0.1`, not `torch==2.0.1+cu117`. A `+local` version exists only on
the PyTorch index and no PyPI mirror can serve it. That costs nothing, because
the default PyPI wheel for torch 2.0.1 *is* the CUDA 11.7 build — pytorch.org's
previous-versions page gives `pip install torch==2.0.1` with no index URL as
the CUDA 11.7 install. Confirm after installing:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
# expect: 2.0.1+cu117 11.7
```

If it reports 11.8 or 12.x, the mirror served a different build; fall back to
`environment.yml`, or pin explicitly against the PyTorch index.

To set the mirrors system-wide instead of per-file:

```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

TUNA has had to restrict its `anaconda/pkgs/*` mirrors under Anaconda's terms.
Its `cloud/conda-forge` mirror is unencumbered, and with `nodefaults` that is
the only conda channel this project needs.

#### The path that needs no network at all

If a working environment already exists on the machine — the one you run
Polyp-PVT with, for instance — reuse it. This project's runtime dependencies
are torch, numpy, scipy, pillow and pyyaml, all of which a Polyp-PVT
environment already has bar possibly `scipy` and `pyyaml`:

```bash
conda activate <your-existing-env>
python -c "import torch, numpy, scipy, PIL, yaml; print('all present')"
pytest -q
```

Anything missing is one small package away, and `pytest`/`gdown` are optional
(tests and the dataset downloader respectively). Two things to check before
relying on it: that `torch.cuda.is_available()` is true, and that
`numpy.__version__` is below 2 if torch is older than 2.4 — see the NumPy note
above.

Worth knowing while you are unblocking the network: **the dataset step needs
no network either.** `tools/prepare_data.py --link` and `--check` are entirely
local, and the manifest and duplicate-audit tools never touch the network. The
only downloads this project ever needs are the Python packages and, if you do
not already have them, the dataset archive and the pretrained backbone.

### CPU-only environment

For the unit tests, the synthetic smoke run, or `tools/analyze.py` on results
produced elsewhere:

```bash
conda env create -f environment-cpu.yml
conda activate polyptail-cpu
```

### Pure-conda alternative

If you would rather conda managed CUDA as well, this still resolves today,
from the frozen `pytorch` channel:

```bash
conda create -n polyptail python=3.10
conda activate polyptail
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge "numpy<2" scipy pillow pyyaml pytest
```

It works, but you are depending on a channel that is no longer maintained.
Prefer `environment.yml` unless you have a specific reason not to.

### pip and virtualenv

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements-dev.txt
```

`requirements.txt` carries the same `numpy<2` ceiling.

### No torchvision, no timm

Images are decoded and transformed through Pillow, and the PVTv2 backbone
vendors the three timm helpers it needs (`to_2tuple`, `trunc_normal_`,
`DropPath`). That is deliberate: in the torch 1.12-2.0 range that CUDA 11.4
pins you to, timm and torchvision compatibility tables are a recurring source
of silent breakage, and `timm` moved `timm.models.layers` to `timm.layers` in
a way that breaks the official Polyp-PVT source outright.

### Check before anything else

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `2.0.1+cu117 True NVIDIA GeForce RTX 3080 Ti`. If `torch.cuda.is_available()`
is `False`, stop: the usual causes are a driver older than 450.80.02, or a CPU
wheel installed by omitting the index URL.

The code handles both torch generations transparently: `weights_only` in
`torch.load` (added in 1.13) and the `GradScaler` spelling change (2.4) are
both version-guarded in `polyptail/utils/io.py`.

## 2. Data and pretrained weights

**If the data is already on this machine, do not download it again.** A
Polyp-PVT checkout next door already has `dataset/TrainDataset` and
`dataset/TestDataset`:

```bash
python tools/prepare_data.py --link ../Polyp-PVT/dataset   # symlink, nothing copied
python tools/prepare_data.py --check                       # validate, no network
```

`--link` validates the target before creating the symlink. Alternatively leave
the data where it is and pass `--root <path>` to the tools and
`data.root=<path>` to the training commands. Manifests store paths relative to
the root, so the same frozen manifest verifies against either location.

Only if you have no copy, with the environment active (it carries `gdown`):

```bash
python tools/prepare_data.py --download --pretrained   # fetch, unpack, validate
```

Sources, if you would rather fetch by hand — all from the Polyp-PVT README:

* datasets: Google Drive file `1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC` (Baidu code
  `sydz`) → unpack into `./dataset/`
* PVTv2-B2: Google Drive folder `1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV` (Baidu
  code `w4vk`) → `./pretrained_pth/pvt_v2_b2.pth`
* Res2Net-50-v1b (A7 control only):
  <https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth>

Then re-run `--check`. See `docs/RUNBOOK.md` step 3 for the layout diagram and
what each diagnostic means.

Both checkpoint loaders are **strict**: a checkpoint that does not line up
raises instead of loading nothing. The reference implementation filters
mismatched keys silently, so pointing it at the wrong file trains from scratch
and never says so — which produces a "reproduction gap" that has nothing to do
with the method.

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
