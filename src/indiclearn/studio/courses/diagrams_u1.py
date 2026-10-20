"""Figures for DSMA Unit 1 (Mathematical Logic).

One builder per topic: propositions, conditionals, equivalence, tautology, predicates, inference,
proofs, normal-forms. Every truth value, minterm and maxterm in these figures is evaluated here in
Python (see `_rows` / `_col`) instead of being typed out, so a figure cannot disagree with the maths.
"""

from __future__ import annotations

import itertools
from typing import Callable, Sequence

from indiclearn.studio.courses.diagrams import (
    BAD,
    BAD_SOFT,
    GOLD,
    GOLD_SOFT,
    GOLD_WASH,
    GOOD,
    GOOD_SOFT,
    INK,
    INK2,
    LINE,
    MUTED,
    SURFACE,
    SURFACE2,
    arrow,
    box,
    caption,
    diamond,
    line,
    path,
    rect,
    register,
    svg,
    table,
    text,
)

# ---------------------------------------------------------------- truth-value helpers

NOT, AND, OR, IMP, IFF = "¬", "∧", "∨", "→", "↔"
FORALL, EXISTS, EQUIV, NEQ, ENT = "∀", "∃", "≡", "≠", "⊢"
NB = "   "  # SVG collapses plain runs of spaces; these survive


def _assigns(n: int) -> list[tuple[bool, ...]]:
    """All 2**n assignments in the textbook order: all T first, down to all F."""
    return list(itertools.product([True, False], repeat=n))


def _tf(b: bool) -> str:
    return "T" if b else "F"


def _col(n: int, fn: Callable[..., bool]) -> list[bool]:
    """The truth column of `fn` over every assignment of n variables."""
    return [bool(fn(*a)) for a in _assigns(n)]


def _rows(n: int, cols: Sequence[Callable[..., bool]]) -> list[list[str]]:
    """Truth-table rows: the n variables, then one column per callable."""
    out = []
    for a in _assigns(n):
        out.append([_tf(v) for v in a] + [_tf(bool(f(*a))) for f in cols])
    return out


def _tf_fill(first_value_col: int) -> Callable[[int, int, str], str]:
    """Tint T green and F red, but only from column `first_value_col` on."""

    def fill(_i: int, j: int, v: str) -> str:
        if j < first_value_col:
            return ""
        return GOOD_SOFT if v == "T" else BAD_SOFT

    return fill


def _card(x: float, y: float, w: float, h: float, title: str, rows: Sequence[str],
          fill: str = SURFACE, stroke: str = LINE, title_fill: str = INK,
          size: float = 12, pad: float = 12) -> str:
    """A left-aligned box: one title line, then body lines."""
    out = rect(x, y, w, h, fill, stroke, 8)
    out += text(x + pad, y + 19, title, 12.5, title_fill, "start", weight=600)
    for i, r in enumerate(rows):
        out += text(x + pad, y + 36 + i * 15.5, r, size, INK2, "start")
    return out


def _bracket(x1: float, x2: float, y: float, label: str, color: str = MUTED) -> str:
    """A span bracket above a group of table columns, with its name above it."""
    out = path(f"M{x1:.1f} {y + 6:.1f}V{y:.1f}H{x2:.1f}V{y + 6:.1f}", "none", color, 1.1)
    out += text((x1 + x2) / 2, y - 6, label, 12, color)
    return out


# ---------------------------------------------------------------- propositions
@register("dsma", "propositions")
def _propositions() -> str:
    """Structured truth table: input columns, sub-expression columns, highlighted main connective."""
    cols = [f"{NOT}q", f"p {AND} {NOT}q", f"p {OR} q", f"(p {AND} {NOT}q) {IMP} (p {OR} q)"]
    fns = [
        lambda p, q: not q,
        lambda p, q: p and not q,
        lambda p, q: p or q,
        lambda p, q: (not (p and not q)) or (p or q),
    ]
    rows = _rows(2, fns)
    final = _col(2, fns[-1])

    widths = [48, 48, 58, 104, 92, 176]
    x0 = (700 - sum(widths)) / 2
    ty, rh = 62, 28
    body = table(x0, ty, ["p", "q"] + cols, rows, widths, rh, highlight=5)

    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)
    body += _bracket(xs[0], xs[2], ty - 14, "input variables")
    body += _bracket(xs[2], xs[5], ty - 14, "sub-expressions, in precedence order")
    body += _bracket(xs[5], xs[6], ty - 14, "main connective", GOLD)

    bottom = ty + rh * (len(rows) + 1)
    verdict = ("every row is T, so this proposition is a tautology" if all(final)
               else "the output column mixes T and F, so this is a contingency")
    body += caption(350, bottom + 22, f"{NOT} binds tightest, then {AND}, then {OR}, "
                                     f"then {IMP}, then {IFF}")
    body += text(350, bottom + 42, verdict, 12, GOOD if all(final) else INK2, weight=500)
    return svg(700, int(bottom + 58), body,
               f"truth table for (p and not q) implies (p or q), with sub-expression columns")


# ---------------------------------------------------------------- conditionals
@register("dsma", "conditionals")
def _conditionals() -> str:
    """The four conditional forms side by side; equivalent pairs sit next to each other."""
    fns = [
        lambda p, q: (not p) or q,            # p -> q
        lambda p, q: (not (not q)) or (not p),  # ~q -> ~p   (contrapositive)
        lambda p, q: (not q) or p,            # q -> p      (converse)
        lambda p, q: (not (not p)) or (not q),  # ~p -> ~q  (inverse)
    ]
    heads = [f"p {IMP} q", f"{NOT}q {IMP} {NOT}p", f"q {IMP} p", f"{NOT}p {IMP} {NOT}q"]
    rows = _rows(2, fns)
    c = [_col(2, f) for f in fns]
    assert c[0] == c[1] and c[2] == c[3] and c[0] != c[2]  # the point of the figure

    widths = [46, 46] + [138] * 4
    x0 = (700 - sum(widths)) / 2
    ty, rh = 70, 28
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)

    body = _bracket(xs[2], xs[4], ty - 16, f"statement {EQUIV} contrapositive", GOOD)
    body += _bracket(xs[4], xs[6], ty - 16, f"converse {EQUIV} inverse", GOLD)
    body += table(x0, ty, ["p", "q"] + heads, rows, widths, rh, cell_fill=_tf_fill(2))

    bottom = ty + rh * (len(rows) + 1)
    body += text(xs[3], bottom + 22, "identical columns", 12, GOOD, weight=500)
    body += text(xs[5], bottom + 22, "identical columns", 12, GOLD, weight=500)
    body += caption(350, bottom + 44,
                    f"the two groups differ, so p {IMP} q is NOT equivalent to its converse or "
                    f"its inverse")
    return svg(700, int(bottom + 60), body,
               "truth table of a conditional with its converse, inverse and contrapositive")


# ---------------------------------------------------------------- equivalence
@register("dsma", "equivalence")
def _equivalence() -> str:
    """Rewriting chain for not(p -> q), checked against a truth table beside it."""
    steps = [
        (f"{NOT}(p {IMP} q)", ""),
        (f"{NOT}({NOT}p {OR} q)", f"Implication Law:  p {IMP} q {EQUIV} {NOT}p {OR} q"),
        (f"{NOT}{NOT}p {AND} {NOT}q", f"De Morgan:  {NOT}(a {OR} b) {EQUIV} {NOT}a {AND} {NOT}b"),
        (f"p {AND} {NOT}q", f"Double Negation:  {NOT}{NOT}p {EQUIV} p"),
    ]
    bx, bw, bh, step = 30, 300, 44, 86
    body = ""
    for i, (expr, law) in enumerate(steps):
        y = 28 + i * step
        last = i == len(steps) - 1
        body += box(bx, y, bw, bh, expr, GOLD_WASH if last else SURFACE,
                    GOLD if last else LINE, 15, 600 if last else 500)
        if law:
            body += arrow(bx + 46, y - step + bh, bx + 46, y - 3, GOLD, 1.4)
            body += text(bx + 60, y - step + bh + 27, law, 12, GOLD, "start")

    tx, ty, tw = 390, 104, [44, 44, 102, 90]
    rows = _rows(2, [lambda p, q: not ((not p) or q), lambda p, q: p and not q])
    body += table(tx, ty, ["p", "q", f"{NOT}(p{IMP}q)", f"p {AND} {NOT}q"], rows, tw, 28,
                  cell_fill=_tf_fill(2))
    body += caption(tx + sum(tw) / 2, ty - 12, "check: the two columns agree")
    assert _col(2, lambda p, q: not ((not p) or q)) == _col(2, lambda p, q: p and not q)
    body += text(tx + sum(tw) / 2, ty + 172, f"{NOT}(p {IMP} q){NB}{EQUIV}{NB}p {AND} {NOT}q", 13,
                 GOOD, weight=600)
    body += caption(tx + sum(tw) / 2, ty + 196, "so the negation of an implication")
    body += caption(tx + sum(tw) / 2, ty + 212, "is not an implication at all")

    bottom = 28 + (len(steps) - 1) * step + bh
    body += caption(350, bottom + 26, "one law per step, named: this is how the proof is written "
                                      "in an exam")
    return svg(700, int(bottom + 42), body,
               "rewriting not(p implies q) to p and not q by the implication, De Morgan and "
               "double-negation laws")


# ---------------------------------------------------------------- tautology
@register("dsma", "tautology")
def _tautology() -> str:
    """Universe of propositions split into tautology, contingency, contradiction."""
    taut = (f"(p {AND} q) {OR} {NOT}(p {AND} q)", lambda p, q: (p and q) or not (p and q))
    cont = (f"p {IMP} q", lambda p, q: (not p) or q)
    contra = (f"(p {OR} q) {AND} {NOT}(p {OR} q)", lambda p, q: (p or q) and not (p or q))
    assert all(_col(2, taut[1])) and not any(_col(2, contra[1]))
    assert any(_col(2, cont[1])) and not all(_col(2, cont[1]))

    body = rect(24, 34, 652, 212, SURFACE2, LINE, 12)
    body += text(36, 54, "universe: every compound proposition", 12, MUTED, "start")
    body += rect(40, 66, 420, 166, SURFACE, GOLD, 10)
    body += text(52, 86, f"Satisfiable — true in at least one row", 12.5, GOLD, "start",
                 weight=600)

    def strip(cx: float, y: float, name: str, rule: str, expr: str,
              fn: Callable[..., bool], color: str) -> str:
        out = text(cx, y, name, 13, color, weight=600)
        out += text(cx, y + 18, rule, 12, INK2)
        out += text(cx, y + 36, expr, 12.5, INK, weight=500)
        vals = _col(2, fn)
        cw, w = 26, 26 * 4
        for k, v in enumerate(vals):
            xx = cx - w / 2 + k * cw
            out += rect(xx, y + 48, cw, 22, GOOD_SOFT if v else BAD_SOFT, LINE, 0, 1)
            out += text(xx + cw / 2, y + 63, _tf(v), 12, GOOD if v else BAD, weight=600)
        out += text(cx, y + 86, "final column", 11.5, MUTED)
        return out

    body += rect(54, 98, 190, 122, GOOD_SOFT, GOOD, 8, 1)
    body += strip(149, 118, "Tautology", "always T", taut[0], taut[1], GOOD)
    body += rect(256, 98, 190, 122, GOLD_SOFT, GOLD, 8, 1)
    body += strip(351, 118, "Contingency", "T in some rows, F in others", cont[0], cont[1], GOLD)
    body += rect(474, 66, 188, 166, BAD_SOFT, BAD, 10, 1)
    body += text(568, 86, "Unsatisfiable", 12.5, BAD, weight=600)
    body += strip(568, 118, "Contradiction", "always F", contra[0], contra[1], BAD)

    body += caption(350, 266, "the three classes are disjoint and cover everything; "
                              "satisfiable = tautology + contingency")
    return svg(700, 282, body,
               "Euler diagram: tautology and contingency inside satisfiable, contradiction outside")


# ---------------------------------------------------------------- predicates
@register("dsma", "predicates")
def _predicates() -> str:
    """Negating a nested quantified statement, one quantifier at a time."""
    steps = [
        (f"{NOT} ( {FORALL}x {EXISTS}y P(x, y) )", ""),
        (f"{EXISTS}x {NOT} ( {EXISTS}y P(x, y) )",
         f"{NOT}{FORALL}x  {EQUIV}  {EXISTS}x{NOT}      (flip the outer quantifier)"),
        (f"{EXISTS}x {FORALL}y {NOT} P(x, y)",
         f"{NOT}{EXISTS}y  {EQUIV}  {FORALL}y{NOT}      (flip the next one)"),
    ]
    bx, bw, bh, step = 30, 360, 46, 92
    body = ""
    for i, (expr, law) in enumerate(steps):
        y = 28 + i * step
        last = i == len(steps) - 1
        body += box(bx, y, bw, bh, expr, GOLD_WASH if last else SURFACE,
                    GOLD if last else LINE, 15, 600 if last else 500)
        if law:
            body += arrow(bx + 40, y - step + bh, bx + 40, y - 3, GOLD, 1.4)
            body += text(bx + 54, y - step + bh + 29, law, 12, GOLD, "start")

    body += _card(420, 28, 252, 152, "Worked instance", [
        f"{FORALL}x {EXISTS}y (x + y = 0)",
        "“every x has some y summing to 0”",
        "",
        f"negation:  {EXISTS}x {FORALL}y (x + y {NEQ} 0)",
        "“some x has no y at all that",
        "works for it”",
    ], GOLD_WASH, GOLD, GOLD)
    body += _card(420, 192, 252, 66, "Order is fixed", [
        f"x stays before y: {FORALL}x{EXISTS}y P is not",
        f"the same claim as {EXISTS}y{FORALL}x P",
    ], SURFACE2, LINE, INK)

    bottom = 28 + (len(steps) - 1) * step + bh
    body += caption(210, bottom + 26, f"each {NOT} moves one quantifier inward and swaps it")
    return svg(700, int(bottom + 42), body,
               "step-by-step negation of for-all x there-exists y P(x,y)")


# ---------------------------------------------------------------- inference
@register("dsma", "inference")
def _inference() -> str:
    """Two-column formal proof of p -> q, q -> not r, r therefore not p."""
    proof = [
        [f"p {IMP} q", "Premise"],
        [f"q {IMP} {NOT}r", "Premise"],
        [f"p {IMP} {NOT}r", "Hypothetical Syllogism (1, 2)"],
        ["r", "Premise"],
        [f"{NOT}{NOT}r", "Double Negation (4)"],
        [f"{NOT}p", f"Modus Tollens (3, 5):  from {NOT}({NOT}r) and p {IMP} {NOT}r"],
    ]
    rows = [[str(i + 1)] + r for i, r in enumerate(proof)]

    # the argument really is valid: no assignment makes every premise true and the conclusion false
    bad = [a for a in _assigns(3)
           if ((not a[0]) or a[1]) and ((not a[1]) or (not a[2])) and a[2] and a[0]]
    assert not bad

    widths = [38, 170, 412]
    x0 = (700 - sum(widths)) / 2
    body = rect(x0, 22, sum(widths), 38, GOLD_WASH, GOLD, 8)
    body += text(x0 + 14, 46, "Argument", 12, GOLD, "start", weight=600)
    body += text(x0 + 94, 46, f"p {IMP} q,{NB}q {IMP} {NOT}r,{NB}r{NB}{ENT}{NB}{NOT}p",
                 13.5, INK, "start", weight=600)

    ty, rh = 74, 30
    last = len(rows) - 1
    body += table(x0, ty, ["#", "Step / Expression", "Reason / Rule Applied"], rows, widths, rh,
                  cell_fill=lambda i, j, v: GOOD_SOFT if i == last else "", mono=False)
    bottom = ty + rh * (len(rows) + 1)
    body += text(x0 + 14, bottom + 24, "Conclusion reached from the premises alone",
                 12, GOOD, "start", weight=600)
    body += text(x0 + sum(widths) - 14, bottom + 24,
                 f"so (p{IMP}q) {AND} (q{IMP}{NOT}r) {AND} r {IMP} {NOT}p is a tautology: valid",
                 12, GOOD, "end")
    body += caption(350, bottom + 46, "name the rule and the line numbers on every step")
    return svg(700, int(bottom + 62), body,
               "two-column formal proof table deriving not p from three premises")


# ---------------------------------------------------------------- proofs
@register("dsma", "proofs")
def _proofs() -> str:
    """Decision flowchart over the five proof strategies."""
    body = diamond(350, 46, 320, 56, "Is the claim to be proved, or disproved?", GOLD_WASH, GOLD,
                   13)
    body += box(40, 96, 300, 34, f"PROVE{NB}p {IMP} q", GOLD_SOFT, GOLD, 13, 600)
    body += box(370, 96, 280, 34, f"DISPROVE{NB}{FORALL}x P(x)", BAD_SOFT, BAD, 13, 600)
    body += arrow(190, 48, 190, 92, INK2, 1.3)   # left vertex of the diamond, straight down
    body += arrow(510, 48, 510, 92, INK2, 1.3)   # right vertex

    methods = [
        ("Direct", [f"assume p is true, derive q by a chain", "of valid steps"]),
        ("Contrapositive", [f"assume {NOT}q, derive {NOT}p",
                            f"uses p {IMP} q {EQUIV} {NOT}q {IMP} {NOT}p"]),
        ("Contradiction", [f"assume p {AND} {NOT}q, derive an", "absurdity such as 1 = 0"]),
        ("Proof by Cases", ["split the domain into exhaustive cases",
                            "(n even / n odd) and prove each"]),
    ]
    y, h, gap = 146, 58, 8
    body += line(32, y, 32, y + len(methods) * (h + gap) - gap, GOLD, 2)
    for i, (name, rows) in enumerate(methods):
        yy = y + i * (h + gap)
        body += _card(40, yy, 300, h, name, rows, SURFACE, LINE, GOLD, 11.8)

    body += _card(370, y, 280, h, "Counterexample", ["one x in the domain with P(x) false",
                                                     "is already enough to disprove it"],
                  SURFACE, LINE, BAD, 11.8)
    body += _card(370, y + h + gap, 280, 2 * h + gap, "Worked: contrapositive", [
        f"claim: n² even {IMP} n even",
        f"assume {NOT}q: n is odd, n = 2k + 1",
        f"then n² = 2(2k² + 2k) + 1, which is odd",
        f"that is {NOT}p, so the claim holds",
    ], GOLD_WASH, GOLD, GOLD, 11.8)
    body += _card(370, y + 3 * (h + gap), 280, h, "Worked: counterexample",
                  ["“every prime is odd” is false:", "n = 2 is prime and even"],
                  BAD_SOFT, BAD, BAD, 11.8)

    bottom = y + len(methods) * (h + gap) - gap
    body += caption(350, bottom + 20, "say which strategy you are using before the first line of "
                                      "algebra")
    return svg(700, int(bottom + 36), body,
               "flowchart of the five proof strategies: direct, contrapositive, contradiction, "
               "cases and counterexample")


# ---------------------------------------------------------------- normal-forms
@register("dsma", "normal-forms")
def _normal_forms() -> str:
    """2x2 grid of F(p, q) = p -> q, each cell carrying its minterm or maxterm."""
    fn: Callable[[bool, bool], bool] = lambda p, q: (not p) or q
    names = ("p", "q")

    def lit(v: str, val: bool, neg_on_true: bool) -> str:
        return (f"{NOT}{v}" if val == neg_on_true else v)

    def minterm(a: tuple[bool, ...]) -> str:
        return f" {AND} ".join(lit(n, v, False) for n, v in zip(names, a))

    def maxterm(a: tuple[bool, ...]) -> str:
        return f" {OR} ".join(lit(n, v, True) for n, v in zip(names, a))

    def idx(a: tuple[bool, ...]) -> int:
        return int("".join("1" if v else "0" for v in a), 2)

    cells = {a: bool(fn(*a)) for a in _assigns(2)}
    ones = sorted(idx(a) for a, v in cells.items() if v)
    zeros = sorted(idx(a) for a, v in cells.items() if not v)
    pdnf = f" {OR} ".join(f"({minterm(a)})" for a in sorted(cells, key=idx) if cells[a])
    pcnf = f" {AND} ".join(f"({maxterm(a)})" for a in sorted(cells, key=idx) if not cells[a])

    gx, gy, cw, ch = 70, 66, 148, 92
    body = text(gx + cw / 2, gy - 28, "q = T", 12.5, MUTED, weight=600)
    body += text(gx + cw * 1.5, gy - 28, "q = F", 12.5, MUTED, weight=600)
    body += text(gx - 12, gy + ch / 2 + 4, "p = T", 12.5, MUTED, "end", weight=600)
    body += text(gx - 12, gy + ch * 1.5 + 4, "p = F", 12.5, MUTED, "end", weight=600)
    body += text(gx + cw, gy - 50, f"F(p, q) = p {IMP} q", 13.5, INK, weight=600)

    for r, pv in enumerate([True, False]):
        for c, qv in enumerate([True, False]):
            a = (pv, qv)
            v = cells[a]
            x, y = gx + c * cw, gy + r * ch
            body += rect(x, y, cw, ch, GOOD_SOFT if v else BAD_SOFT, GOOD if v else BAD, 0, 1.2)
            body += text(x + cw / 2, y + 26, _tf(v), 17, GOOD if v else BAD, weight=700)
            body += text(x + cw / 2, y + 48, f"row index {idx(a)}", 12, MUTED)
            body += text(x + cw / 2, y + 68, minterm(a) if v else maxterm(a), 13,
                         INK, weight=600)
            body += text(x + cw / 2, y + 84, "minterm" if v else "maxterm", 11.5,
                         GOOD if v else BAD)

    body += _card(386, 40, 286, 98, "T row → minterm (AND)", [
        f"x appears as x when x = T,", f"as {NOT}x when x = F", "the OR of them all is the PDNF",
    ], GOOD_SOFT, GOOD, GOOD)
    body += _card(386, 150, 286, 98, "F row → maxterm (OR)", [
        f"x appears as {NOT}x when x = T,", "as x when x = F",
        "the AND of them all is the PCNF",
    ], BAD_SOFT, BAD, BAD)

    sx = ", ".join(str(i) for i in ones)
    px = ", ".join(str(i) for i in zeros)
    body += rect(28, 272, 644, 38, GOOD_SOFT, GOOD, 8, 1)
    body += text(42, 296, "PDNF", 12.5, GOOD, "start", weight=700)
    body += text(104, 296, f"{pdnf}{NB}={NB}Σ m({sx})", 13, INK, "start", weight=500)
    body += rect(28, 318, 644, 38, BAD_SOFT, BAD, 8, 1)
    body += text(42, 342, "PCNF", 12.5, BAD, "start", weight=700)
    body += text(104, 342, f"{pcnf}{NB}={NB}Π M({px})", 13, INK, "start", weight=500)
    n_one, n_zero = len(ones), len(zeros)
    body += caption(350, 376, f"row index is the binary value of (p, q) with T = 1, F = 0; "
                              f"{n_one} T row{'s' if n_one != 1 else ''} give the minterms, "
                              f"{n_zero} F row{'s' if n_zero != 1 else ''} the maxterms")
    return svg(700, 392, body,
               "two by two grid of p implies q, mapping true rows to minterms and false rows to "
               "maxterms")
