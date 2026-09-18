#!/usr/bin/env python3
"""Lay out ./dataset and ./pretrained_pth, and check they are right.

Run inside the conda environment (``conda activate polyptail``), which is
where ``gdown`` lives:

    python tools/prepare_data.py --check                 # validate, no network
    python tools/prepare_data.py --download              # fetch + unpack, then validate
    python tools/prepare_data.py --download --pretrained # also fetch the backbones

``--check`` is the part that matters and the part that is tested. It compares
what you have against what the reference implementation hard-codes, and
reports in terms you can act on -- a wrong directory name, an archive unzipped
one level too deep, an unpaired mask, a file extension the reference silently
skips -- instead of failing on the first surprise several steps later.

``--download`` is a convenience wrapper around ``gdown``. Google Drive rate
limits large public files and periodically changes its confirmation flow, so
treat it as best-effort: if it fails, download by hand from the links it
prints and re-run ``--check``. Nothing downstream cares how the bytes arrived.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from polyptail.data.layout import check_layout  # noqa: E402

# Sources, from the Polyp-PVT README (sections 4.2 and 4.3).
DATASET_DRIVE_ID = "1pFxb9NbM8mj_rlSawTlcXG1OdVGAbRQC"
DATASET_URL = f"https://drive.google.com/file/d/{DATASET_DRIVE_ID}/view"
PVT_FOLDER_URL = "https://drive.google.com/drive/folders/1Eu8v9vMRvt-dyCH0XSV2i77lAd62nPXV"
RES2NET_URL = (
    "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/"
    "res2net50_v1b_26w_4s-3cf99910.pth"
)


def _need_gdown() -> str:
    exe = shutil.which("gdown")
    if exe:
        return exe
    print("gdown is not on PATH.\n"
          "  conda activate polyptail     (environment.yml installs it)\n"
          "  or: pip install gdown\n"
          "Or download by hand -- see the links below -- and re-run with --check.",
          file=sys.stderr)
    raise SystemExit(2)


def _unpack(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"unpacking {archive.name} -> {dest}/")
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        shutil.unpack_archive(str(archive), str(dest))
    _flatten_single_nesting(dest)


def _flatten_single_nesting(dest: Path) -> None:
    """Undo the common 'archive contains one top-level folder' case.

    The distributed zip sometimes expands to ``dataset/dataset/TrainDataset``
    depending on how it was repacked, which then fails every path in the
    reference code with an unhelpful error.
    """
    entries = [p for p in dest.iterdir() if not p.name.startswith(".")]
    if len(entries) != 1 or not entries[0].is_dir():
        return
    inner = entries[0]
    if not any((inner / d).is_dir() for d in ("TrainDataset", "TestDataset")):
        return
    print(f"  flattening redundant top-level directory {inner.name}/")
    for child in list(inner.iterdir()):
        shutil.move(str(child), str(dest / child.name))
    inner.rmdir()


def download_datasets(root: Path, keep_archive: bool) -> None:
    gdown = _need_gdown()
    archive = root.parent / "polyp_datasets.zip"
    if archive.is_file():
        print(f"reusing existing {archive}")
    else:
        print(f"downloading datasets from {DATASET_URL}")
        rc = subprocess.call([gdown, "--id", DATASET_DRIVE_ID, "-O", str(archive)])
        if rc != 0 or not archive.is_file():
            print(f"\ngdown failed (exit {rc}). Download this by hand and re-run with --check:\n"
                  f"  {DATASET_URL}\n"
                  f"  Baidu mirror: https://pan.baidu.com/s/1BTgT27VxvOgKpHrigwm7Bw  code: sydz\n"
                  f"Unpack it so that {root}/TrainDataset/images/ exists.", file=sys.stderr)
            raise SystemExit(1)
    _unpack(archive, root)
    if not keep_archive:
        archive.unlink(missing_ok=True)


def download_pretrained(dest: Path) -> None:
    gdown = _need_gdown()
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "pvt_v2_b2.pth").is_file():
        print("pvt_v2_b2.pth already present")
    else:
        print(f"downloading PVTv2-B2 from {PVT_FOLDER_URL}")
        rc = subprocess.call([gdown, "--folder", PVT_FOLDER_URL, "-O", str(dest)])
        if rc != 0:
            print(f"\ngdown failed (exit {rc}). Fetch pvt_v2_b2.pth by hand from\n"
                  f"  {PVT_FOLDER_URL}\n"
                  f"and place it at {dest}/pvt_v2_b2.pth", file=sys.stderr)

    res2net = dest / "res2net50_v1b_26w_4s-3cf99910.pth"
    if res2net.is_file():
        print("res2net checkpoint already present")
    else:
        print(f"downloading Res2Net-50-v1b (only needed for the A7 control)")
        curl = shutil.which("curl")
        if curl:
            subprocess.call([curl, "-L", "--fail", "-o", str(res2net), RES2NET_URL])
        else:
            print(f"curl not found; fetch {RES2NET_URL} by hand into {dest}/", file=sys.stderr)

    # Flatten in case gdown --folder created a nested directory.
    for sub in [p for p in dest.iterdir() if p.is_dir()]:
        for f in sub.glob("*.pth"):
            shutil.move(str(f), str(dest / f.name))
        if not any(sub.iterdir()):
            sub.rmdir()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("./dataset"))
    ap.add_argument("--pretrained-dir", type=Path, default=Path("./pretrained_pth"))
    ap.add_argument("--download", action="store_true", help="fetch and unpack the datasets")
    ap.add_argument("--pretrained", action="store_true", help="also fetch the backbone weights")
    ap.add_argument("--check", action="store_true", help="validate the layout only (the default)")
    ap.add_argument("--keep-archive", action="store_true", help="do not delete the downloaded zip")
    args = ap.parse_args()

    if args.download:
        download_datasets(args.root, args.keep_archive)
    if args.pretrained:
        download_pretrained(args.pretrained_dir)

    rep = check_layout(args.root)
    print()
    print(rep.summary())

    pvt = args.pretrained_dir / "pvt_v2_b2.pth"
    print()
    if pvt.is_file():
        print(f"pretrained backbone: {pvt} ({pvt.stat().st_size / 2**20:.1f} MiB)")
    else:
        print(f"pretrained backbone MISSING: expected {pvt}\n"
              f"  get it from {PVT_FOLDER_URL} (Baidu code: w4vk),\n"
              f"  or run: python tools/prepare_data.py --pretrained")

    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
