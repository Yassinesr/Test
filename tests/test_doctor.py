"""The setup doctor must run where nothing else does.

Its whole premise is that it works before any environment exists -- in conda's
`base`, on a machine where you cannot install anything, which is exactly the
situation it is for. So the load-bearing property is not what it prints but
what it imports: nothing outside the standard library.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCTOR = ROOT / "tools" / "doctor.py"

#: Imported deliberately, inside try/except, to *report* their absence.
PROBED = {"torch", "numpy", "scipy", "PIL", "yaml", "pytest", "gdown"}


def top_level_imports(path: Path) -> set[str]:
    """Modules imported unconditionally at module scope."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:                      # module scope only
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


class TestStdlibOnly:
    def test_exists_and_is_executable_as_a_script(self):
        assert DOCTOR.is_file()
        assert DOCTOR.read_text().startswith("#!")

    @pytest.mark.skipif(sys.version_info < (3, 10),
                        reason="sys.stdlib_module_names needs Python 3.10+")
    def test_imports_nothing_outside_the_standard_library(self):
        third_party = top_level_imports(DOCTOR) - set(sys.stdlib_module_names) - PROBED
        assert not third_party, (
            f"tools/doctor.py must run before any environment exists, but imports "
            f"{sorted(third_party)}"
        )

    def test_does_not_import_the_project_itself(self):
        """Importing polyptail would defeat the point: the package needs the
        very dependencies the doctor exists to report as missing.

        Checked against the import graph, not against the word appearing in
        the file -- the doctor legitimately prints `conda activate polyptail`.
        """
        tree = ast.parse(DOCTOR.read_text())
        imported = set()
        for node in ast.walk(tree):             # anywhere, not just module scope
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and getattr(node.func, "id", "") == "__import__":
                if node.args and isinstance(node.args[0], ast.Constant):
                    imported.add(str(node.args[0].value).split(".")[0])
        assert "polyptail" not in imported


@pytest.fixture(scope="module")
def run():
    return subprocess.run([sys.executable, str(DOCTOR)], cwd=ROOT,
                          capture_output=True, text=True, timeout=180)


class TestRuns:

    def test_completes_without_traceback(self, run):
        assert "Traceback" not in run.stderr, run.stderr
        assert run.returncode in (0, 1), f"unexpected exit {run.returncode}"

    def test_reports_all_three_sections(self, run):
        for heading in ("1. Can this interpreter run the project?",
                        "2. Proxy and channel configuration",
                        "3. Which package sources are reachable?",
                        "What to do next"):
            assert heading in run.stdout, f"missing section: {heading}"

    def test_checks_every_runtime_dependency(self, run):
        for package in ("torch", "numpy", "scipy", "pillow", "pyyaml"):
            assert package in run.stdout, f"doctor does not check {package}"

    def test_always_ends_with_an_actionable_step(self, run):
        tail = run.stdout[run.stdout.index("What to do next"):]
        assert len(tail.strip().splitlines()) > 2, "the recommendation section is empty"

    def test_names_the_sources_it_probed(self, run):
        for host in ("conda-forge", "PyPI", "PyTorch wheels", "TUNA"):
            assert host in run.stdout


def load_doctor():
    """Import tools/doctor.py as a module without it being on sys.path."""
    spec = importlib.util.spec_from_file_location("doctor_under_test", DOCTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_torch(cuda_version, available, device="NVIDIA GeForce RTX 3080 Ti"):
    t = types.SimpleNamespace()
    t.version = types.SimpleNamespace(cuda=cuda_version)
    t.cuda = types.SimpleNamespace(
        is_available=lambda: available,
        get_device_name=lambda i: device,
    )
    return t


class TestCudaDiagnosis:
    """`torch.cuda.is_available()` collapses three unrelated problems into
    False. Each needs a different fix, so the doctor must tell them apart."""

    def test_working_cuda(self, monkeypatch):
        d = load_doctor()
        monkeypatch.setattr(d, "_nvidia_smi", lambda: "RTX 3080 Ti, 470.129.06")
        notes: list[str] = []
        ok, kind = d._check_cuda(fake_torch("11.7", True), notes)
        assert ok and kind == "ok" and not notes

    def test_cpu_only_wheel_is_named_as_such(self, monkeypatch):
        """The case no driver fix can solve -- a different package is needed."""
        d = load_doctor()
        monkeypatch.setattr(d, "_nvidia_smi", lambda: "RTX 3080 Ti, 470.129.06")
        notes: list[str] = []
        ok, kind = d._check_cuda(fake_torch(None, False), notes)
        assert not ok and kind == "cpu_only_wheel"
        assert any("CPU-ONLY" in n for n in notes)
        assert any("environment.yml" in n for n in notes)

    def test_cuda_wheel_without_a_driver(self, monkeypatch):
        d = load_doctor()
        monkeypatch.setattr(d, "_nvidia_smi", lambda: None)
        notes: list[str] = []
        ok, kind = d._check_cuda(fake_torch("11.7", False), notes)
        assert not ok and kind == "no_driver"
        assert any("nvidia-smi is absent" in n for n in notes)

    def test_driver_present_but_no_device_visible(self, monkeypatch):
        d = load_doctor()
        monkeypatch.setattr(d, "_nvidia_smi", lambda: "RTX 3080 Ti, 470.129.06")
        notes: list[str] = []
        ok, kind = d._check_cuda(fake_torch("11.7", False), notes)
        assert not ok and kind == "driver_but_no_device"
        assert any("CUDA_VISIBLE_DEVICES" in n for n in notes)

    def test_a_broken_torch_does_not_crash_the_doctor(self, monkeypatch):
        d = load_doctor()
        monkeypatch.setattr(d, "_nvidia_smi", lambda: None)
        t = types.SimpleNamespace(version=types.SimpleNamespace(cuda="11.7"))
        t.cuda = types.SimpleNamespace(
            is_available=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        notes: list[str] = []
        ok, _ = d._check_cuda(t, notes)
        assert not ok and any("boom" in n for n in notes)


class TestRecommendation:
    """A usable interpreter with no CUDA must not be reported as ready."""

    def _rec(self, capsys, iface, cfg=None, net=None):
        d = load_doctor()
        d.recommend(iface, cfg or {"env_proxies": {}, "condarc_proxies": []},
                    net or {"conda-forge": True, "PyPI": True})
        return capsys.readouterr().out

    def test_cuda_ready_interpreter_is_told_to_get_on_with_it(self, capsys):
        out = self._rec(capsys, {"usable": True, "cuda": True, "cuda_kind": "ok",
                                 "missing": [], "optional_missing": [], "notes": []})
        assert "CUDA included" in out
        assert "offline half" not in out

    def test_no_cuda_interpreter_is_told_what_still_works(self, capsys):
        out = self._rec(capsys, {"usable": True, "cuda": False,
                                 "cuda_kind": "cpu_only_wheel",
                                 "missing": [], "optional_missing": [], "notes": []})
        assert "offline half of the workflow works" in out
        assert "freeze_manifest" in out          # the work that is NOT blocked
        assert "CPU-ONLY build" in out           # and the reason it is
        assert "CUDA included" not in out


class TestProxySourceLocation:
    """Finding the file that sets the proxy is most of the fix; leaving the
    user to grep for it is a round trip the tool should not cost them."""

    def _home(self, tmp_path, body: str):
        (tmp_path / ".bashrc").write_text(body)
        return tmp_path

    def test_locates_exports_with_file_and_line(self, tmp_path, monkeypatch):
        home = self._home(tmp_path, "export PATH=/x\n"
                                    "export http_proxy=http://127.0.0.1:16041\n"
                                    "export https_proxy=http://127.0.0.1:16041\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        hits = load_doctor().find_proxy_exports()
        found = {(Path(p).name, n) for p, n, _ in hits}
        assert (".bashrc", 2) in found and (".bashrc", 3) in found

    def test_ignores_already_commented_lines(self, tmp_path, monkeypatch):
        """Otherwise it keeps reporting a proxy the user has already disabled."""
        home = self._home(tmp_path, "# export http_proxy=http://old:3128\n"
                                    "export https_proxy=http://127.0.0.1:16041\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        hits = load_doctor().find_proxy_exports()
        assert len(hits) == 1 and "https_proxy" in hits[0][2]

    def test_does_not_match_unrelated_exports(self, tmp_path, monkeypatch):
        home = self._home(tmp_path, "export PATH=/x\nexport NO_PROXY_SETTING=1\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        assert load_doctor().find_proxy_exports() == []

    def test_recommendation_quotes_the_lines_and_a_safe_sed(self, capsys, tmp_path):
        d = load_doctor()
        cfg = {
            "env_proxies": {"http_proxy": "http://127.0.0.1:16041"},
            "condarc_proxies": [],
            "exports": [(str(tmp_path / ".bashrc"), 142,
                         "export http_proxy=http://127.0.0.1:16041")],
            "proxy_is_local": True, "proxy_alive": False,
        }
        d.recommend({"usable": True, "cuda": False, "cuda_kind": "cpu_only_wheel",
                     "missing": [], "optional_missing": [], "notes": []},
                    cfg, {"conda-forge": True, "PyPI": True})
        out = capsys.readouterr().out
        assert ".bashrc:142" in out                 # names the exact line
        assert "LOOPBACK" in out                    # explains why it is dead
        assert "cp ~/.bashrc ~/.bashrc.bak" in out  # backs up before editing
        assert "sed -i -E" in out
