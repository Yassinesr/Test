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
ENV_FILES = ["environment.yml", "environment-cpu.yml", "environment-cn.yml"]
PYTORCH_INDEX = "download.pytorch.org"


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
        # conda-forge, directly or through a mirror. Nothing here lives
        # anywhere else, which is what makes `nodefaults` safe.
        assert any("conda-forge" in str(c) for c in blob["channels"])

    def test_installs_torch_through_pip_not_conda(self, name):
        """PyTorch's own Anaconda channel is deprecated; the wheels are the
        supported path, and they bundle their own CUDA runtime."""
        conda, pip = conda_deps(name)
        assert spec_for(conda, "pytorch") is None
        assert any(p.startswith("torch") for p in pip)
        assert any(p.startswith(("-i", "--index-url", "--extra-index-url")) for p in pip), \
            "the pip section must state which index it installs torch from"

    def test_never_falls_back_to_the_anaconda_default_channels(self, name):
        """Without `nodefaults`, conda appends repo.anaconda.com even when the
        file names only conda-forge -- which fails on a restricted network and
        pulls in Anaconda's commercial terms for nothing."""
        blob = yaml.safe_load((ROOT / name).read_text())
        assert "nodefaults" in blob["channels"], f"{name} must declare nodefaults"
        assert not any("repo.anaconda.com" in str(c) for c in blob["channels"])

    def test_no_environment_pins_a_local_cuda_version(self, name):
        """A `+cuXXX` local version exists only on download.pytorch.org, so
        pinning one makes that single host a hard requirement. It is slow or
        blocked from much of the world, and the failure mode is a five-minute
        timeout ending in "No matching distribution found" -- which reads like
        a missing package rather than a network problem.

        This inverts an earlier decision in this repository. The local pin was
        added to stop pip serving the PyPI build of the same version number;
        that ambiguity is theoretical for torch 2.0.1, whose default PyPI
        wheel *is* the CUDA 11.7 build, while the outage it caused was real.
        """
        _, pip = conda_deps(name)
        torch_spec = next(p for p in pip if p.startswith("torch"))
        assert "+" not in torch_spec, (
            f"{name} pins {torch_spec!r}; a local version can only be served by "
            "the PyTorch index, making that host a single point of failure"
        )

    def test_the_torch_version_is_pinned_exactly(self, name):
        """Dropping the local tag must not mean dropping the version: the CUDA
        build you get is a property of *which version* you ask for."""
        _, pip = conda_deps(name)
        torch_spec = next(p for p in pip if p.startswith("torch"))
        if name == "environment-cpu.yml":
            pytest.skip("CPU environment intentionally tracks the current wheel")
        assert "==" in torch_spec, f"{name} must pin torch exactly, got {torch_spec!r}"

    def test_states_the_cuda_build_and_how_to_check_it(self, name):
        """Because the pin no longer names the build, the file has to say what
        it resolves to and how to confirm it."""
        if name == "environment-cpu.yml":
            pytest.skip("no CUDA build to state")
        text = (ROOT / name).read_text()
        assert "11.7" in text or "cu117" in text
        assert "torch.version.cuda" in text, (
            f"{name} must tell the reader how to verify the CUDA build it got"
        )


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
