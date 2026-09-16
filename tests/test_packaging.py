"""The three install paths must agree.

There are three ways to build an environment here -- ``environment.yml``,
``environment-cpu.yml`` and ``requirements.txt`` -- and a constraint that
exists in one but not the others is worse than no constraint at all, because
it fails for some users and not others.

The NumPy ceiling is the one that matters today: torch 2.0.1, the build
recommended for a CUDA 11.4 driver, was compiled against the NumPy 1.x C API
and fails at import under NumPy 2. The code itself works on both majors (the
``np.bitwise_count`` fallback in ``polyptail/data/phash.py`` exists for
exactly that reason, and CI runs the suite under each), so the ceiling is a
torch constraint, not ours -- but it has to be stated identically everywhere.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = ["environment.yml", "environment-cpu.yml"]


def conda_deps(name: str) -> tuple[list[str], list[str]]:
    blob = yaml.safe_load((ROOT / name).read_text())
    conda, pip = [], []
    for item in blob["dependencies"]:
        if isinstance(item, dict) and "pip" in item:
            pip.extend(item["pip"])
        else:
            conda.append(item)
    return conda, pip


def spec_for(deps: list[str], package: str) -> str | None:
    for d in deps:
        if re.match(rf"^{re.escape(package)}\b", d.strip()):
            return d.strip()
    return None


@pytest.mark.parametrize("name", ENV_FILES)
class TestEnvironmentFiles:
    def test_parses_and_names_an_environment(self, name):
        blob = yaml.safe_load((ROOT / name).read_text())
        assert blob["name"].startswith("polyptail")
        assert "conda-forge" in blob["channels"]

    def test_installs_torch_through_pip_not_conda(self, name):
        """PyTorch's own Anaconda channel is deprecated; the wheels are the
        supported path, and they bundle their own CUDA runtime."""
        conda, pip = conda_deps(name)
        assert spec_for(conda, "pytorch") is None
        assert any(p.startswith("torch") for p in pip)
        assert any("download.pytorch.org" in p for p in pip)

    def test_the_gpu_pin_names_its_cuda_build_explicitly(self, name):
        """`torch==2.0.1` alone is ambiguous under --extra-index-url: PyPI
        carries that version number too, so pip could serve either wheel. The
        `+cu117` local version exists only on the PyTorch index, which makes
        the pin mean what the docs say it means."""
        if name != "environment.yml":
            pytest.skip("CPU environment intentionally tracks the current wheel")
        _, pip = conda_deps(name)
        torch_spec = next(p for p in pip if p.startswith("torch"))
        index = next(p for p in pip if "download.pytorch.org" in p)
        cuda_tag = index.rstrip("/").rsplit("/", 1)[-1]          # e.g. "cu117"
        assert f"+{cuda_tag}" in torch_spec, (
            f"{torch_spec!r} does not pin the {cuda_tag} build served by {index!r}"
        )

    def test_caps_numpy_below_two(self, name):
        conda, _ = conda_deps(name)
        spec = spec_for(conda, "numpy")
        assert spec is not None, "numpy must be pinned explicitly, not left to the solver"
        assert "<2" in spec, f"{name} must cap numpy below 2.0, got {spec!r}"

    def test_pins_a_python_version(self, name):
        conda, _ = conda_deps(name)
        assert spec_for(conda, "python") is not None


class TestConsistency:
    def test_numpy_ceiling_is_identical_in_all_four_install_paths(self):
        req = (ROOT / "requirements.txt").read_text()
        line = next(l for l in req.splitlines() if l.strip().startswith("numpy")).strip()
        assert "<2" in line, f"requirements.txt must carry the ceiling, got {line!r}"
        for name in ENV_FILES:
            conda, _ = conda_deps(name)
            assert spec_for(conda, "numpy").replace(" ", "") == line.replace(" ", "")
        pyproject = (ROOT / "pyproject.toml").read_text()
        numpy_line = next(l for l in pyproject.splitlines() if '"numpy' in l).strip()
        assert "<2" in numpy_line, f"pyproject.toml must carry the ceiling, got {numpy_line!r}"

    def test_runtime_dependencies_agree_between_pip_and_conda(self):
        req = {
            l.split(">=")[0].split("<")[0].split("==")[0].strip()
            for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        }
        conda, pip = conda_deps("environment.yml")
        names = {c.split(">=")[0].split("<")[0].split("=")[0].strip() for c in conda}
        names |= {p.split("==")[0].strip() for p in pip if not p.startswith("--")}
        missing = req - names
        assert not missing, f"environment.yml is missing runtime deps: {sorted(missing)}"

    def test_pyproject_declares_the_same_runtime_packages(self):
        text = (ROOT / "pyproject.toml").read_text()
        block = text[text.index("dependencies = ["):]
        block = block[: block.index("]")]
        declared = {
            re.split(r"[><=]", l.strip().strip('",'))[0]
            for l in block.splitlines()[1:]
            if l.strip() and not l.strip().startswith("#")
        }
        req = {
            re.split(r"[><=]", l.strip())[0]
            for l in (ROOT / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.strip().startswith("#")
        }
        assert declared == req, f"pyproject {sorted(declared)} != requirements {sorted(req)}"
