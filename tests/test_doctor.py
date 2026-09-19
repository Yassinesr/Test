"""The setup doctor must run where nothing else does.

Its whole premise is that it works before any environment exists -- in conda's
`base`, on a machine where you cannot install anything, which is exactly the
situation it is for. So the load-bearing property is not what it prints but
what it imports: nothing outside the standard library.
"""

from __future__ import annotations

import ast
import subprocess
import sys
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
        very dependencies the doctor exists to report as missing."""
        assert "polyptail" not in DOCTOR.read_text().replace("polyptail doctor", "")


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
