import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def synthetic_dataset(tmp_path):
    """A tiny PraNet-layout dataset plus its frozen manifest."""
    import numpy as np
    from PIL import Image

    from polyptail.data.manifest import build_manifest, write_manifest

    rng = np.random.default_rng(0)
    splits = {"TrainDataset": 12, "ValidationDataset": 5, "TestDataset/Fake": 6}
    for split, n in splits.items():
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "masks").mkdir(parents=True)
        for i in range(n):
            h, w = 40 + i % 3, 48 + i % 5
            yy, xx = np.mgrid[0:h, 0:w]
            mask = ((yy - h / 2) ** 2 / (h / 5) ** 2 + (xx - w / 2) ** 2 / (w / 5) ** 2) <= 1
            img = np.clip(rng.normal(120, 20, (h, w, 3)) + mask[..., None] * 60, 0, 255).astype(np.uint8)
            # The validation stems are prefixed so that an overlap with the
            # training split has to be created deliberately by a test, never
            # by the fixture numbering both from zero.
            stem = f"v{i:03d}" if split == "ValidationDataset" else f"{i:03d}"
            Image.fromarray(img).save(tmp_path / split / "images" / f"{stem}.png")
            Image.fromarray(mask.astype(np.uint8) * 255).save(tmp_path / split / "masks" / f"{stem}.png")
    manifest = build_manifest(tmp_path, list(splits))
    write_manifest(manifest, tmp_path / "manifest.json", tmp_path / "manifest.sha256")
    return tmp_path, manifest
