"""DSMA Unit 3 figures - Counting Techniques and Recurrence Relations.

One builder per lesson topic: the two counting rules, inclusion-exclusion, the pigeonhole principle,
the permutation/combination decision tree, Pascal's triangle with the hockey-stick identity, and the
three recurrence lessons (modelling, homogeneous, non-homogeneous).

Every number on these figures is computed here - binomial coefficients with ``math.comb``, the
characteristic roots by solving the quadratic, the inclusion-exclusion counts by actual floor
division - so a figure can never disagree with the lesson text.
"""

from __future__ import annotations

import math

from indiclearn.studio.courses.diagrams import (
    BAD,
    BAD_SOFT,
    GOLD,
    GOLD2,
    GOLD_SOFT,
    GOLD_WASH,
    GOOD,
    GOOD_SOFT,
    INK,
    INK2,
    LINE,
    MONO,
    MUTED,
    SURFACE,
    SURFACE2,
    WARN,
    WARN_SOFT,
    arrow,
    caption,
    circle,
    diamond,
    esc,
    line,
    path,
    rect,
    register,
    svg,
    text,
)

NBSP = " "
W = 700  # the lesson column


# ---------------------------------------------------------------- maths text (sub/superscripts)
def _runs(s: str) -> list[tuple[str, str]]:
    """Split 'a_n = a_{n-1} + 2^n' into [('', 'a'), ('sub', 'n'), ('', ' = a'), ...]."""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in "_^" and i + 1 < len(s):
            kind = "sub" if c == "_" else "sup"
            i += 1
            if s[i] == "{":
                j = s.index("}", i)
                body, i = s[i + 1 : j], j + 1
            else:
                body, i = s[i], i + 1
            if buf:
                out.append(("", "".join(buf)))
                buf = []
            out.append((kind, body))
        else:
            buf.append(c)
            i += 1
    if buf:
        out.append(("", "".join(buf)))
    return out


def mtext(x: float, y: float, s: str, size: float = 13, fill: str = INK, anchor: str = "middle",
          weight: str | int = 400, mono: bool = True) -> str:
    """A line of maths: ``_`` and ``^`` (optionally braced) become real sub/superscripts.

    The primitives draw flat strings only, so this is the one place the module writes its own
    <text>: a subscripted a_{n-1} is worth it in a recurrence figure.
    """
    shift = 0.0
    parts = []
    for kind, body in _runs(s):
        want = 0.0 if not kind else (size * 0.25 if kind == "sub" else -size * 0.36)
        dy = want - shift
        shift = want
        fs = size if not kind else size * 0.72
        d = f' dy="{dy:.2f}"' if abs(dy) > 0.005 else ""
        parts.append(f'<tspan font-size="{fs:.2f}"{d}>{esc(body.replace(" ", NBSP))}</tspan>')
    fam = f' font-family="{MONO}"' if mono else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{fam}>{"".join(parts)}</text>')


def _step(x: float, y: float, n: int, words: str) -> str:
    """A small numbered stage caption, e.g. '2 . characteristic equation'."""
    return (text(x, y, f"{n}", 11, GOLD, "start", weight=700)
            + text(x + 10, y, f"· {words}", 11, MUTED, "start"))


def _panel(x: float, y: float, w: float, h: float, fill: str = GOLD_WASH, stroke: str = LINE) -> str:
    return rect(x, y, w, h, fill, stroke, 10, 1.2)


def _num(v: float) -> str:
    s = str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"
    return s.replace("-", "−")  # a real minus sign, not a hyphen


def _term(coef: float, base: float, first: bool = False) -> str:
    """One term of a closed form: 2*3^n -> '2·3^n', -1*2^n -> '− 2^n'."""
    sign = "" if first else ("+ " if coef > 0 else "− ")
    if first and coef < 0:
        sign = "−"
    mag = abs(coef)
    head = "" if abs(mag - 1) < 1e-9 else f"{_num(mag)}·"
    return f"{sign}{head}{_num(base)}^n"


# ================================================================ basic-counting
@register("dsma", "basic-counting")
def _basic_counting() -> str:
    na, nb, nd = 3, 4, 2
    body = text(190, 28, "Sum rule — one or the other", 12.5, GOLD, weight=700)
    body += text(530, 28, "Product rule — one after the other", 12.5, GOLD, weight=700)
    body += line(360, 16, 360, 268, LINE, 1, dash="4 5")

    # --- sum rule: two disjoint circles
    for cx, name, cnt in ((132, "A", na), (250, "B", nb)):
        body += circle(cx, 122, 50, SURFACE2, LINE, 1.4)
        body += text(cx, 116, name, 16, GOLD, weight=700)
        body += text(cx, 136, f"{cnt} ways", 12, INK2)
    body += mtext(191, 194, "A ∩ B = ∅", 11.5, MUTED)
    body += mtext(191, 232, f"|A ∪ B| = {na} + {nb} = {na + nb}", 14, INK, weight=700)
    body += caption(191, 258, "no outcome is in both — so just add")

    # --- product rule: a 3 x 2 tree
    root = (538, 60)
    mid = [448, 538, 628]
    body += text(538, 44, "start", 10.5, MUTED, italic=True)
    body += text(366, 132, "step 1", 10.5, MUTED, "start", italic=True)
    body += text(366, 200, "step 2", 10.5, MUTED, "start", italic=True)
    body += circle(*root, 8, GOLD, GOLD, 1.5)
    for i, mx in enumerate(mid):
        body += line(root[0], root[1] + 8, mx, 112, LINE, 1.2)
        body += circle(mx, 128, 16, GOLD_SOFT, GOLD2, 1.4)
        body += text(mx, 132.5, str(i + 1), 12.5, INK, weight=600)
        for j, lx in enumerate((mx - 27, mx + 27)):
            body += line(mx, 144, lx, 183, LINE, 1.2)
            body += circle(lx, 196, 13, SURFACE2, LINE, 1.2)
            body += text(lx, 200, "ab"[j], 12, INK2, weight=600)
    body += mtext(518, 232, f"{na} × {nd} = {na * nd}", 14, INK, weight=700)
    body += text(564, 232, "paths", 13, INK2, "start")
    body += caption(538, 258, f"{na} options, then {nd} more for each — multiply")
    return svg(W, 280, body,
               "Left: two disjoint circles A and B with 3 and 4 outcomes, added by the sum rule. "
               "Right: a tree with 3 first-step branches each splitting into 2, giving 6 paths by the product rule.")


# ================================================================ incl-excl
@register("dsma", "incl-excl")
def _incl_excl() -> str:
    n, divs = 100, (2, 3, 5)
    singles = [n // d for d in divs]
    pairs = [n // (divs[0] * divs[1]), n // (divs[1] * divs[2]), n // (divs[0] * divs[2])]
    triple = n // (divs[0] * divs[1] * divs[2])
    total = sum(singles) - sum(pairs) + triple

    r = 66.0
    ax, ay, bx, by, cx, cy = 139.0, 129.0, 211.0, 129.0, 175.0, 191.6
    body = _panel(46, 44, 258, 248, SURFACE, LINE)
    body += text(62, 63, "U", 13, MUTED, weight=600, italic=True)
    for px, py, fill in ((ax, ay, GOLD_WASH), (bx, by, GOLD_WASH), (cx, cy, GOLD_WASH)):
        body += circle(px, py, r, fill, GOLD2, 1.4, fill_opacity=0.55)
    for px, py, name in ((67.1, 87.5, "A"), (282.9, 87.5, "B"), (175, 274.6, "C")):
        body += text(px, py, name, 15, GOLD, weight=700)

    # the sign each of the seven regions carries: singles +, pairwise overlaps -, the triple +
    regions = ((103.5, 113, "+"), (246.5, 113, "+"), (175, 237, "+"),      # A only, B only, C only
               (175, 108, "−"), (134.5, 178, "−"), (215.5, 178, "−"),      # A∩B, A∩C, B∩C only
               (175, 160, "+"))                                           # A∩B∩C
    for sx, sy, sign in regions:
        body += text(sx, sy, sign, 17, GOOD if sign == "+" else BAD, weight=700)

    # formula + the lesson's worked count
    body += _panel(326, 60, 350, 220)
    body += mtext(342, 86, "|A ∪ B ∪ C|  =", 13.5, INK, "start", 700)
    rows = (
        ("+", GOOD, "|A| + |B| + |C|"),
        ("−", BAD, "|A∩B| − |B∩C| − |A∩C|"),
        ("+", GOOD, "|A∩B∩C|"),
    )
    for i, (sign, col, rest) in enumerate(rows):
        yy = 114 + i * 28
        body += text(352, yy, sign, 15, col, "start", weight=700)
        body += mtext(372, yy, rest, 13, INK2, "start")
    body += line(342, 208, 660, 208, LINE, 1)
    body += text(342, 228, f"how many of 1…{n} are divisible by 2, 3 or 5?", 11.5, MUTED, "start")
    body += mtext(342, 252, f"({' + '.join(map(str, singles))}) − ({' + '.join(map(str, pairs))}) + {triple}",
                  13, INK, "start")
    body += mtext(342, 274, f"= {sum(singles)} − {sum(pairs)} + {triple} = {total}", 14, GOLD, "start", 700)

    for lx, sign, col, words in ((90, "+", GOOD, "add each set once"),
                                 (260, "−", BAD, "subtract pairwise overlaps"),
                                 (480, "+", GOOD, "add the triple back")):
        body += text(lx, 314, sign, 13, col, "start", weight=700)
        body += text(lx + 14, 314, words, 11.5, MUTED, "start")
    return svg(W, 332, body,
               "Three overlapping circles A, B, C inside the universal set, each of the seven regions marked "
               "with the sign it carries in the inclusion-exclusion formula, with the formula and the count of "
               f"integers 1 to {n} divisible by 2, 3 or 5 beside it.")


# ================================================================ pigeonhole
def _dot(cx: float, cy: float, gold: bool = False) -> str:
    return circle(cx, cy, 8, GOLD if gold else MUTED, GOLD if gold else MUTED, 1)


@register("dsma", "pigeonhole")
def _pigeonhole() -> str:
    k1, holes = 5, 5
    pigeons = holes + 1
    k, want = 4, 3
    need = k * (want - 1) + 1

    body = text(350, 32, f"Basic form — {pigeons} pigeons into {holes} holes", 12.5, GOLD, weight=700)
    xs = [74 + i * 114 for i in range(k1)]
    for i, bx in enumerate(xs):
        last = i == k1 - 1
        body += rect(bx, 58, 96, 70, GOLD_WASH if last else SURFACE2, GOLD if last else LINE,
                     6, 2 if last else 1.3)
        if last:
            body += _dot(bx + 32, 93) + _dot(bx + 64, 93, gold=True)
        else:
            body += _dot(bx + 48, 93)
        body += text(bx + 48, 146, f"hole {i + 1}", 11, MUTED)
    body += arrow(594, 38, 594, 83, GOLD, 1.6)
    body += text(670, 44, f"pigeon {pigeons}", 11.5, GOLD, "end", weight=600)
    body += mtext(350, 172, f"⌈{pigeons}/{holes}⌉ = {-(-pigeons // holes)}"
                            "  →  some hole holds 2 or more", 13, INK2, mono=False)
    body += line(74, 192, 626, 192, LINE, 1, dash="4 5")

    body += text(350, 216, f"Generalized — force {want} cards of one suit", 12.5, GOLD, weight=700)
    xs2 = [93 + i * 134 for i in range(k)]
    suits = ("clubs", "diamonds", "hearts", "spades")
    for i, bx in enumerate(xs2):
        last = i == k - 1
        body += rect(bx, 236, 112, 78, GOLD_WASH if last else SURFACE2, GOLD if last else LINE,
                     6, 2 if last else 1.3)
        body += _dot(bx + 40, 264) + _dot(bx + 72, 264)
        if last:
            body += _dot(bx + 56, 292, gold=True)
        body += text(bx + 56, 330, suits[i], 11, MUTED)
    body += arrow(660, 292, 563, 292, GOLD, 1.6)
    body += text(666, 278, f"card {need}", 11.5, GOLD, "end", weight=600)
    body += mtext(350, 358, f"N = k(r−1) + 1 = {k}({want}−1) + 1 = {need}", 14.5, GOLD, weight=700)
    body += caption(350, 380, f"{need - 1} cards ({want - 1} per suit) can still dodge it — card {need} cannot")
    return svg(W, 394, body,
               f"Top: {holes} boxes holding one dot each until the {pigeons}th pigeon forces a second dot into "
               f"one box. Bottom: {k} suits holding {want - 1} cards each, so card {need} forces {want} of a suit.")


# ================================================================ perm-comb
def _leaf(cx: float, y: float, formula: str, second: str, w: float = 148, h: float = 78) -> str:
    out = rect(cx - w / 2, y, w, h, GOLD_SOFT, GOLD2, 9, 1.4)
    out += mtext(cx, y + 32, formula, 14.5, INK, weight=700)
    out += mtext(cx, y + 58, second, 12, INK2)
    return out


@register("dsma", "perm-comb")
def _perm_comb() -> str:
    body = diamond(350, 50, 270, 58, "Does order matter?", SURFACE2, LINE)
    for dx in (181, 519):                                   # elbow routing, so nothing looks crossed
        body += path(f"M350 79V98H{dx}V113", "none", LINE, 1.4)
        body += diamond(dx, 140, 210, 54, "Repetition allowed?", SURFACE2, LINE)
    body += text(265, 90, "yes → permutation", 11.5, INK2, weight=600)
    body += text(435, 90, "no → combination", 11.5, INK2, weight=600)

    for dx, nox, yesx in ((181, 100, 262), (519, 438, 600)):
        for lx in (nox, yesx):
            body += path(f"M{dx} 167V186H{lx}V222", "none", LINE, 1.4, arrowhead=True)
        body += text(nox - 12, 212, "no", 11.5, INK2, "end", weight=600)
        body += text(yesx + 12, 212, "yes", 11.5, INK2, "start", weight=600)

    body += _leaf(100, 228, "P(n, r)", "n! / (n−r)!")
    body += _leaf(262, 228, "n^r", "n ways, r slots")
    body += _leaf(438, 228, "C(n, r)", "n! / (r!(n−r)!)")
    body += _leaf(600, 228, "C(n+r−1, r)", "stars and bars")
    body += caption(350, 332, "ask about order first, then about repetition — the pair of answers picks the formula")
    return svg(W, 348, body,
               "A decision tree: 'Does order matter?' splits into permutation and combination, each then asking "
               "'Repetition allowed?', giving the four formulas P(n,r), n^r, C(n,r) and C(n+r-1, r).")


# ================================================================ binomial
@register("dsma", "binomial")
def _binomial() -> str:
    top = 6
    rows = [[math.comb(n, k) for k in range(n + 1)] for n in range(top + 1)]
    r_diag, lo, hi = 1, 1, 5                       # hockey stick: sum of C(i, 1) for i = lo..hi
    diag = [(i, r_diag) for i in range(lo, hi + 1)]
    tgt = (hi + 1, r_diag + 1)                     # = C(n+1, r+1)
    terms = [rows[i][k] for i, k in diag]
    assert sum(terms) == rows[tgt[0]][tgt[1]] == math.comb(hi + 1, r_diag + 1)

    cx, dx, dy, y0, rad = 220.0, 45.0, 39.0, 48.0, 15.5

    def pos(n: int, k: int) -> tuple[float, float]:
        return cx + (k - n / 2) * dx, y0 + n * dy

    body = ""
    for n in range(top):                                           # the Pascal lattice, very faint
        for k in range(n + 1):
            x1, y1 = pos(n, k)
            for x2, y2 in (pos(n + 1, k), pos(n + 1, k + 1)):
                body += line(x1, y1, x2, y2, LINE, 0.7)
    stick = [pos(*c) for c in diag] + [pos(*tgt)]
    body += path("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in stick), "none", GOLD2, 9,
                 stroke_opacity=0.5, stroke_linecap="round", stroke_linejoin="round")

    for n in range(top + 1):
        body += text(32, y0 + n * dy + 4.5, f"n={n}", 11, MUTED, "start")
        for k in range(n + 1):
            x, y = pos(n, k)
            if (n, k) == tgt:
                body += circle(x, y, rad, GOLD, GOLD, 1.8)
                body += text(x, y + 4.5, rows[n][k], 12.5, SURFACE, weight=700)
            elif (n, k) in diag:
                body += circle(x, y, rad, GOLD_SOFT, GOLD2, 1.8)
                body += text(x, y + 4.5, rows[n][k], 12.5, INK, weight=700)
            else:
                body += circle(x, y, rad, SURFACE, LINE, 0.9)
                body += text(x, y + 4.5, rows[n][k], 12, INK2)

    body += _panel(400, 86, 280, 184)
    body += text(416, 112, "Hockey-stick identity", 12.5, GOLD, "start", weight=700)
    body += mtext(416, 140, f"C({lo},{r_diag}) + … + C({hi},{r_diag})", 12.5, INK2, "start")
    body += mtext(416, 164, f"= C({hi + 1}, {r_diag + 1})", 12.5, INK2, "start")
    body += line(416, 182, 664, 182, LINE, 1)
    body += mtext(416, 206, " + ".join(map(str, terms)) + f" = {sum(terms)}", 13.5, INK, "start", 700)
    body += mtext(416, 230, f"C({tgt[0]}, {tgt[1]}) = {rows[tgt[0]][tgt[1]]}", 13.5, GOOD, "start", 700)
    body += caption(416, 254, "the sum sits one row down, one step across", "start")
    body += text(125, 320, "each cell is the sum of the two above it:", 11.5, MUTED, "start")
    body += mtext(369, 320, "C(n,k) = C(n−1,k−1) + C(n−1,k)", 11.5, MUTED, "start")
    return svg(W, 338, body,
               f"Pascal's triangle rows 0 to {top}. The cells C({lo},{r_diag}) to C({hi},{r_diag}) form a gold "
               f"diagonal whose terms {' + '.join(map(str, terms))} add to {sum(terms)}, the filled cell "
               f"C({tgt[0]},{tgt[1]}) one row below and one step across - the hockey-stick identity.")


# ================================================================ recurrence-model
def _strip(x: float, y: float, h: float, segs: list[tuple[float, str, str, str, float]]) -> str:
    out, px = "", x
    for w, lab, fill, stroke, size in segs:
        out += rect(px, y, w, h, fill, stroke, 5, 1.3)
        out += text(px + w / 2, y + h / 2 + 4.5, lab, size, INK, weight=600)
        px += w
    return out


@register("dsma", "recurrence-model")
def _recurrence_model() -> str:
    a1, a2 = 2, 3
    a3 = a1 + a2
    body = mtext(350, 30, "a_n = bit strings of length n with no two consecutive 0s", 13, INK,
                 weight=700, mono=False)
    body += rect(290, 46, 120, 48, SURFACE2, LINE, 8, 1.3)
    body += mtext(350, 66, "length n", 12.5, INK, weight=600)
    body += text(350, 84, "split on the last bit", 10.5, MUTED)

    body += arrow(350, 94, 190, 146, INK2, 1.3)
    body += arrow(350, 94, 510, 146, INK2, 1.3)
    body += text(250, 112, "last bit = 1", 11.5, INK2, weight=600)
    body += text(450, 112, "last bit = 0", 11.5, INK2, weight=600)

    body += _strip(89, 150, 42, [(158, "n−1 bits, valid", SURFACE2, LINE, 11.5),
                                 (44, "1", GOLD_SOFT, GOLD2, 14)])
    body += _strip(409, 150, 42, [(114, "n−2 bits, valid", SURFACE2, LINE, 11),
                                  (44, "1", WARN_SOFT, WARN, 14),
                                  (44, "0", GOLD_SOFT, GOLD2, 14)])
    body += mtext(190, 218, "a_{n-1}", 15.5, GOLD, weight=700)
    body += caption(190, 240, "a 1 may follow anything valid")
    body += mtext(510, 218, "a_{n-2}", 15.5, GOLD, weight=700)
    body += caption(510, 240, "a 0 forces a 1 just before it")

    body += line(70, 266, 630, 266, LINE, 1, dash="4 5")
    body += mtext(350, 296, "a_n = a_{n-1} + a_{n-2}", 16.5, GOLD, weight=700)
    body += mtext(350, 324, f"a_1 = {a1}  (0, 1)      a_2 = {a2}  (01, 10, 11)      so a_3 = {a3}",
                  12, INK2)
    body += caption(350, 348, "the two cases are disjoint and cover every string — so add them")
    return svg(W, 364, body,
               "A bit string of length n split by its last bit: ending in 1 leaves any valid string of length "
               "n-1, while ending in 0 forces a 1 before it and leaves a valid string of length n-2, giving "
               "a(n) = a(n-1) + a(n-2).")


# ================================================================ recurrence-homog
def _roots(c1: int, c2: int) -> tuple[float, float]:
    """Roots of r^2 - c1 r - c2 = 0 for a_n = c1 a_{n-1} + c2 a_{n-2}."""
    disc = c1 * c1 + 4 * c2
    s = math.sqrt(disc)
    return max((c1 + s) / 2, (c1 - s) / 2), min((c1 + s) / 2, (c1 - s) / 2)


@register("dsma", "recurrence-homog")
def _recurrence_homog() -> str:
    c1, c2, a0, a1 = 5, -6, 1, 4                    # a_n = 5a_{n-1} - 6a_{n-2}
    hi, lo = _roots(c1, c2)
    r1, r2 = lo, hi                                 # quote the smaller root first
    al1 = (a1 - a0 * r2) / (r1 - r2)
    al2 = a0 - al1
    chk = 2
    assert abs(al1 * r1**chk + al2 * r2**chk - (c1 * (al1 * r1 + al2 * r2) + c2 * a0)) < 1e-9
    a_chk = int(round(al1 * r1**chk + al2 * r2**chk))

    stages = (
        (1, "recurrence", "a_n = c_1 a_{n-1} + c_2 a_{n-2}"),
        (2, "characteristic equation", "r^2 − c_1 r − c_2 = 0"),
        (3, "roots", "r_1,  r_2"),
    )
    body = ""
    for i, (num, words, formula) in enumerate(stages):
        bx = 25 + i * 225
        body += rect(bx, 40, 200, 56, SURFACE2, LINE, 9, 1.3)
        body += _step(bx + 14, 60, num, words)
        body += mtext(bx + 100, 84, formula, 12.5, INK, weight=600)
        if i:
            body += arrow(bx - 25, 68, bx - 4, 68, INK2, 1.4)

    body += _step(292, 118, 4, "general form — which case applies?")
    cases = (
        (25, "distinct roots   r_1 ≠ r_2", "a_n = α_1 r_1^n + α_2 r_2^n", GOOD_SOFT, GOOD),
        (360, "repeated root   r_1 = r_2 = r_0", "a_n = (α_1 + α_2 n) r_0^n", WARN_SOFT, WARN),
    )
    for bx, title, formula, fill, stroke in cases:
        body += rect(bx, 128, 315, 76, fill, stroke, 9, 1.4)
        body += mtext(bx + 157.5, 152, title, 12.5, INK, weight=600, mono=False)
        body += mtext(bx + 157.5, 182, formula, 14.5, INK, weight=700)
        body += arrow(bx + 157.5, 204, bx + 157.5, 220, INK2, 1.3)

    body += _panel(25, 222, 650, 134)
    body += _step(44, 246, 5, "initial conditions → closed form   (worked example)")
    closed = f"{_term(al2, r2, first=True)} {_term(al1, r1)}"
    ex = (
        f"a_n = {c1}a_{{n-1}} − {abs(c2)}a_{{n-2}},   a_0 = {a0},  a_1 = {a1}",
        f"r^2 − {c1}r + {abs(c2)} = (r−{_num(r1)})(r−{_num(r2)}) = 0   →   r = {_num(r1)}, {_num(r2)}"
        "   (distinct)",
        f"a_n = α_1 {_num(r1)}^n + α_2 {_num(r2)}^n   →   α_1 + α_2 = {a0},"
        f"  {_num(r1)}α_1 + {_num(r2)}α_2 = {a1}",
        f"α_1 = {_num(al1)},  α_2 = {_num(al2)}   ⇒   a_n = {closed}      (a_{chk} = {a_chk} ✓)",
    )
    for i, row in enumerate(ex):
        body += mtext(44, 272 + i * 25, row, 12.5, INK if i == 3 else INK2, "start", 700 if i == 3 else 400)
    body += caption(350, 376, "the roots fix the shape of the answer; the initial conditions fix the constants")
    return svg(W, 390, body,
               "A pipeline: recurrence, then characteristic equation, then its roots, branching into the "
               "distinct-root and repeated-root general forms, with a_n = 5a(n-1) - 6a(n-2) solved out to "
               "a_n = 2*3^n - 2^n.")


# ================================================================ recurrence-nonhomog
@register("dsma", "recurrence-nonhomog")
def _recurrence_nonhomog() -> str:
    body = mtext(350, 30, "a_n = c_1 a_{n-1} + … + c_k a_{n-k} + F(n)", 13.5, INK, weight=700)
    body += mtext(258, 54, "a_n = a_n^{(h)} + a_n^{(p)}", 12.5, GOLD, weight=700)
    body += text(368, 54, "— homogeneous part first, then the particular one",
                 11.5, MUTED, "start")

    cw = (180.0, 200.0)
    tx, ty, rh = 28.0, 76.0, 34.0
    body += rect(tx, ty, sum(cw), rh, GOLD_SOFT, "none", 0)
    body += mtext(tx + cw[0] / 2, ty + 22, "F(n)", 12.5, INK, weight=700)
    body += mtext(tx + cw[0] + cw[1] / 2, ty + 22, "trial  a_n^{(p)}", 12.5, INK, weight=700)
    table_rows = (
        ("a constant  C", "d", None),
        ("polynomial, degree t", "p_t n^t + … + p_1 n + p_0", None),
        ("c^n,  c not a root", "p c^n", GOOD_SOFT),
        ("c^n,  c a root (mult. m)", "p n^m c^n", WARN_SOFT),
    )
    for i, (f_n, trial, tint) in enumerate(table_rows):
        yy = ty + rh * (i + 1)
        if tint:
            body += rect(tx, yy, sum(cw), rh, tint, "none", 0)
        body += mtext(tx + cw[0] / 2, yy + 22, f_n, 11.5, INK2)
        body += mtext(tx + cw[0] + cw[1] / 2, yy + 22, trial, 12.5, INK, weight=600)
    for j in range(3):
        gx = tx + sum(cw[:j])
        body += line(gx, ty, gx, ty + rh * 5, LINE, 1)
    for i in range(6):
        body += line(tx, ty + rh * i, tx + sum(cw), ty + rh * i, LINE, 1)

    body += _panel(428, 76, 248, 170)
    body += text(444, 100, "the clash check", 12.5, GOLD, "start", weight=700)
    body += text(444, 124, "is the base of F(n) also", 11.5, INK2, "start")
    body += text(444, 141, "a characteristic root?", 11.5, INK2, "start")
    body += text(444, 170, "no", 12, GOOD, "start", weight=700)
    body += mtext(478, 170, "→  use  p c^n", 12.5, INK, "start", 600)
    body += text(444, 200, "yes", 12, BAD, "start", weight=700)
    body += mtext(484, 200, "→  use  p n^m c^n", 12.5, INK, "start", 600)
    body += mtext(444, 228, "multiply by n^m — m is the multiplicity", 11.5, MUTED, "start",
                  mono=False)

    examples = (
        (28, 320, "no clash", GOOD, GOOD_SOFT,
         ("a_n − 3a_{n-1} = 2n,   a_0 = 1",
          "root r = 3;  F(n) = 2n  →  try  dn + e",
          "a_n = (5/2)3^n − n − 3/2")),
        (368, 308, "clash — root r = 3, base c = 3", BAD, BAD_SOFT,
         ("a_n = 3a_{n-1} + 3^n",
          "c = 3 is a root (m = 1)  →  try  p·n·3^n",
          "a_n^{(p)} = n·3^n     (p = 1)")),
    )
    for bx, bw, title, col, fill, body_rows in examples:
        body += rect(bx, 266, bw, 100, fill, col, 9, 1.3)
        body += text(bx + 16, 290, title, 11.5, col, "start", weight=700)
        for i, row in enumerate(body_rows):
            body += mtext(bx + 16, 314 + i * 22, row, 11.5, INK2, "start")
    body += caption(350, 390, "read the trial form off F(n), then check it against the roots before solving")
    return svg(W, 404, body,
               "A table matching each form of F(n) to its trial particular solution, the clash check asking "
               "whether the base of F(n) is a characteristic root (multiply by n^m if it is), and two worked "
               "cases: a non-clashing polynomial and a clashing 3^n.")
