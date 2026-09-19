"""The documented commands must be real commands.

A README that quotes a flag the tool does not accept is worse than no README:
it fails on someone else's machine, at a point where they cannot tell whether
the mistake is theirs. These tests read the fenced shell blocks out of the
docs and check every invocation against the actual argparse definitions, plus
every referenced config, tool and relative link.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) + [
    ROOT / "configs" / "sweeps" / "README.md",
    ROOT / "manifests" / "README.md",
]
DOC_IDS = [str(p.relative_to(ROOT)) for p in DOCS]

FENCE = re.compile(r"```(?:bash|sh|console)?\n(.*?)```", re.S)
TOOL_CALL = re.compile(r"\bpython3?\s+(tools/[\w_]+\.py)((?:[^\n]|\\\n)*)")


def shell_blocks(path: Path) -> list[str]:
    return FENCE.findall(path.read_text())


def tool_flags(tool: Path) -> set[str]:
    """Flags an argparse-based tool accepts, read from its source."""
    src = tool.read_text()
    flags = set(re.findall(r'add_argument\(\s*"(--[\w-]+)"', src))
    flags |= {"--help", "-h"}
    return flags


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
class TestDocumentedCommands:
    def test_every_referenced_tool_exists(self, doc):
        missing = {
            m.group(1) for block in shell_blocks(doc)
            for m in TOOL_CALL.finditer(block)
            if not (ROOT / m.group(1)).is_file()
        }
        assert not missing, f"{doc.name} references non-existent tools: {sorted(missing)}"

    def test_every_documented_flag_is_accepted(self, doc):
        problems: list[str] = []
        for block in shell_blocks(doc):
            for m in TOOL_CALL.finditer(block):
                tool = ROOT / m.group(1)
                if not tool.is_file():
                    continue
                accepted = tool_flags(tool)
                tail = m.group(2).replace("\\\n", " ")
                try:
                    tokens = shlex.split(tail, comments=True)
                except ValueError:
                    continue          # an illustrative fragment, not a command
                for tok in tokens:
                    if not tok.startswith("--"):
                        continue
                    flag = tok.split("=", 1)[0]
                    if flag not in accepted:
                        problems.append(f"{m.group(1)} does not accept {flag}")
        assert not problems, f"{doc.name}: " + "; ".join(sorted(set(problems)))

    def test_every_referenced_config_exists(self, doc):
        text = doc.read_text()
        refs = set(re.findall(r"configs/[\w/]+\.yaml", text))
        missing = {r for r in refs if not (ROOT / r).is_file()}
        assert not missing, f"{doc.name} references missing configs: {sorted(missing)}"

    def test_relative_links_resolve(self, doc):
        text = doc.read_text()
        targets = re.findall(r"\]\((?!https?:|#)([^)#]+)(?:#[^)]*)?\)", text)
        missing = {t for t in targets if not (doc.parent / t).exists()}
        assert not missing, f"{doc.name} has broken relative links: {sorted(missing)}"


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text()


def flat(text: str) -> str:
    """Collapse whitespace, so a phrase split across a line break still matches.

    These tests check what the README *says*, not how it happens to wrap.
    """
    return " ".join(text.split())


class TestReadmeSpecifics:
    """The README is the entry point; these are the claims a new user acts on
    first, so they are pinned rather than trusted."""

    def test_names_all_three_environment_files(self, readme):
        for name in ("environment.yml", "environment-cpu.yml", "environment-cn.yml"):
            assert name in readme, f"README must mention {name}"
            assert (ROOT / name).is_file()

    def test_leads_with_conda(self, readme):
        install = flat(readme[readme.index("## 1."):readme.index("## 2.")])
        assert "conda env create -f environment.yml" in install
        assert "conda activate polyptail" in install

    def test_quotes_the_real_test_count(self, readme):
        counted = len(list(ROOT.glob("tests/test_*.py")))
        assert counted >= 8, "test modules went missing"
        claimed = set(re.findall(r"(\d{3}) (?:passed|tests)", readme))
        assert claimed, "README should state how many tests there are"
        assert len(claimed) == 1, f"README quotes inconsistent test counts: {claimed}"

    def test_says_what_is_not_verified(self, readme):
        """The honest-limits section is load-bearing: it is what stops someone
        reading the acceptance band as a reproduced result."""
        tail = flat(readme[readme.index("# What is verified"):])
        for claim in ("conda env create", "GPU", "real datasets"):
            assert claim in tail, f"the verification caveat must mention {claim}"
