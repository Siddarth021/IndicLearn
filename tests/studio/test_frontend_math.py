"""Formulas must never reach the screen as raw LaTeX ("$p \\vee q$"): guard the helpers that prevent it."""
import shutil
import subprocess
from pathlib import Path

import pytest

UTIL = Path(__file__).resolve().parents[2] / "frontend" / "js" / "util.js"
pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def _node(js: str) -> str:
    code = f'import {{ plainMath, clip }} from "{UTIL.as_posix()}";\n{js}'
    return subprocess.run(["node", "--input-type=module", "-e", code], capture_output=True, text=True, check=True).stdout.strip()


def test_plain_math_turns_latex_into_symbols():
    out = _node(r'console.log(plainMath("The OR ($p \\vee q$) and $A \\cup B \\subseteq \\mathbb{N}$"))')
    assert out == "The OR (p ∨ q) and A ∪ B ⊆ ℕ"


def test_plain_math_leaves_normal_text_alone():
    assert _node('console.log(plainMath("Deadlock avoidance"))') == "Deadlock avoidance"


def test_clip_never_cuts_a_formula_in_half():
    out = _node(r'console.log(clip("abc $p \\vee q$ def ghi", 8))')
    assert "$" not in out and out.endswith("…")
