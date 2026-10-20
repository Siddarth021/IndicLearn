"""LaTeX (as written in lessons, questions and solved answers) -> readable Unicode text, for PDFs and speech.

Only the small subset this app writes: logic and set symbols, \\frac, \\binom, \\sqrt, ^ and _, bmatrix, \\text.
Matrices come back as [a, b; c, d] inline, or as row lists via split_matrices() so the PDF can draw real grids.
"""

from __future__ import annotations

import re

SYMS = {
    "neg": "¬", "lnot": "¬", "wedge": "∧", "land": "∧", "vee": "∨", "lor": "∨", "rightarrow": "→", "to": "→", "leftrightarrow": "↔", "Rightarrow": "⇒",
    "Leftrightarrow": "⇔", "iff": "⇔", "implies": "⇒", "leftarrow": "←", "forall": "∀", "exists": "∃", "in": "∈", "notin": "∉", "cup": "∪", "cap": "∩",
    "bigcup": "∪", "bigcap": "∩", "oplus": "⊕", "circ": "∘", "odot": "⊙", "subseteq": "⊆", "subset": "⊂", "supseteq": "⊇", "emptyset": "∅", "varnothing": "∅",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "equiv": "≡", "approx": "≈", "cdot": "·", "times": "×", "div": "÷", "pm": "±",
    "lambda": "λ", "Lambda": "Λ", "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "epsilon": "ε", "theta": "θ", "mu": "μ", "pi": "π",
    "sigma": "σ", "Sigma": "Σ", "Pi": "Π", "phi": "φ", "omega": "ω", "infty": "∞", "sum": "Σ", "prod": "Π", "perp": "⊥", "mid": "|", "nmid": "∤",
    "therefore": "∴", "vdash": "⊢", "models": "⊨", "top": "⊤", "bot": "⊥", "ell": "ℓ", "dots": "…", "cdots": "⋯", "ldots": "…", "vdots": "⋮",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉", "langle": "⟨", "rangle": "⟩", "quad": "  ", "qquad": "    ", "|": "‖", "{": "{", "}": "}",
    ",": " ", ";": " ", ":": " ", "!": "", " ": " ", "colon": ":", "triangle": "△", "star": "*", "prime": "′", "partial": "∂", "nabla": "∇", "sim": "~",
    "uparrow": "↑", "downarrow": "↓", "mapsto": "↦", "setminus": "∖", "%": "%", "#": "#", "&": "&", "_": "_", "$": "$",
}
SUP = str.maketrans("0123456789+-=()nkTim*", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿᵏᵀⁱᵐ*")
SUB = str.maketrans("0123456789+-=()aeijknmrstx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₙₘᵣₛₜₓ")
BB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ", "C": "ℂ"}


def _group(s: str, i: int) -> tuple[str, int]:
    """The {...} group (or single token) starting at s[i]; returns (content, index after it)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            if s[j] == "\\":
                j += 2
                continue
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    return s[i + 1 : j], j + 1
            j += 1
        return s[i + 1 :], len(s)
    if s[i] == "\\":
        m = re.match(r"\\([A-Za-z]+|.)", s[i:])
        return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _wrap(x: str) -> str:
    x = x.strip()
    return x if re.fullmatch(r"-?\d+(?:\.\d+)?|√?[A-Za-zα-ω]|√\d+", x) else f"({x})"


def _script(x: str, table: dict, mark: str) -> str:
    t = x.strip()
    if t and all(ord(ch) in table for ch in t):
        return t.translate(table)
    return mark + (t if len(t) == 1 else f"({t})")


def plain(s: str) -> str:
    """Convert one LaTeX math string (without the $ signs)."""
    out, i = [], 0
    s = re.sub(r"\\(?:left|right|big|Big)(?=[()\[\].|]|\\[{}|])", "", s).replace("\\displaystyle", "").replace("\\not\\equiv", "≢")
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+|.)", s[i:])
            name = m.group(1)
            i += len(m.group(0))
            if name in ("frac", "dfrac", "tfrac"):
                a, i = _group(s, i)
                b, i = _group(s, i)
                out.append(f"{_wrap(plain(a))}/{_wrap(plain(b))}")
            elif name == "binom":
                a, i = _group(s, i)
                b, i = _group(s, i)
                out.append(f"C({plain(a)}, {plain(b)})")
            elif name == "sqrt":
                a, i = _group(s, i)
                out.append("√" + _wrap(plain(a)))
            elif name in ("text", "mathrm", "operatorname", "mathbf", "textbf", "mathit", "boldsymbol"):
                a, i = _group(s, i)
                out.append(plain(a) if name != "text" else a)
            elif name == "mathbb":
                a, i = _group(s, i)
                out.append(BB.get(a.strip(), a))
            elif name in ("hat", "bar", "overline", "tilde", "vec"):
                a, i = _group(s, i)
                mark = {"hat": "\u0302", "bar": "\u0304", "overline": "\u0305", "tilde": "\u0303", "vec": "\u20d7"}[name]
                p = plain(a)
                out.append(p + mark if len(p) == 1 else f"({p})" + ("′" if name == "overline" else mark))
            elif name == "pmod":
                a, i = _group(s, i)
                out.append(f" (mod {plain(a)})")
            elif name == "begin":
                env, i = _group(s, i)
                end = s.find("\\end{" + env + "}", i)
                body = s[i:end if end >= 0 else len(s)]
                i = (end + len("\\end{" + env + "}")) if end >= 0 else len(s)
                rows = [r for r in re.split(r"\\\\", body) if r.strip()]
                if env in ("bmatrix", "pmatrix", "matrix", "vmatrix"):
                    lb, rb = {"pmatrix": "()", "vmatrix": "||"}.get(env, "[]")
                    out.append(lb + "; ".join(", ".join(plain(c).strip() for c in r.split("&")) for r in rows) + rb)
                else:  # aligned, cases ...
                    out.append("; ".join(plain(r.replace("&", "")).strip() for r in rows))
            elif name == "\\":
                out.append("; ")
            else:
                out.append(SYMS.get(name, name))
        elif c in "^_":
            a, i = _group(s, i + 1)
            out.append(_script(plain(a), SUP if c == "^" else SUB, c))
        elif c in "{}":
            i += 1
        else:
            out.append(c)
            i += 1
    return re.sub(r"[ ]{2,}", " ", "".join(out)).replace("( ", "(").replace(" )", ")").replace("¬ ", "¬")


_SEG = re.compile(r"\$\$(.+?)\$\$|\$([^$]+?)\$|\\\((.+?)\\\)", re.S)


def text(s: str) -> str:
    """Text with $...$ math -> plain readable text."""
    return _SEG.sub(lambda m: plain(m.group(1) or m.group(2) or m.group(3) or ""), str(s or ""))


_MAT = re.compile(r"\\begin\{([bpv]?matrix)\}(.*?)\\end\{\1\}", re.S)


def split_matrices(latex: str) -> list[tuple[str, object]]:
    """A display formula -> [("text", "A ="), ("matrix", [["1","2"],["3","4"]]), ...] so a PDF can draw real matrices."""
    out: list[tuple[str, object]] = []
    pos = 0
    for m in _MAT.finditer(latex):
        if m.start() > pos:
            out.append(("text", plain(latex[pos : m.start()])))
        rows = [[plain(c).strip() for c in r.split("&")] for r in re.split(r"\\\\", m.group(2)) if r.strip()]
        out.append(("matrix", rows))
        pos = m.end()
    if pos < len(latex):
        out.append(("text", plain(latex[pos:])))
    return [(k, v) for k, v in out if k == "matrix" or str(v).strip()]
