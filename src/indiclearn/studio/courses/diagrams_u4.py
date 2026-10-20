"""Figures for DSMA Unit 4 - Relations, Posets and Lattices.

Seven builders: relations, rel-repr, closures, equiv-rel, posets, extremal, lattices.

Everything that could be wrong is computed instead of typed: the covering pairs of a Hasse diagram
come from removing transitive edges, the zero-one matrix comes from the edge list, Warshall's new
edges come from running Warshall, and the upper/lower bounds come from the order itself. Pure
stdlib, deterministic, no network.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

from indiclearn.studio.courses.diagrams import (
    GOLD,
    GOLD_SOFT,
    GOLD_WASH,
    GOOD,
    GOOD_SOFT,
    INFO,
    INK,
    INK2,
    LINE,
    MUTED,
    SURFACE,
    SURFACE2,
    WARN,
    caption,
    circle,
    legend,
    line,
    lines,
    node,
    path,
    rect,
    register,
    svg,
    table,
    text,
)

Pt = tuple[float, float]

# ---------------------------------------------------------------- order theory (computed)


def _strict(elems: Sequence[Any], leq: Callable[[Any, Any], bool]) -> set[tuple[Any, Any]]:
    return {(a, b) for a in elems for b in elems if a != b and leq(a, b)}


def _hasse(elems: Sequence[Any], leq: Callable[[Any, Any], bool]):
    """Covering pairs and levels of a poset.

    (a, b) is a cover when a < b and no c sits strictly between them - that is exactly the test that
    throws away the transitive edges, so a Hasse diagram built from this can never keep one. The
    level of x is the length of the longest chain of covers below it, which is what puts an element
    on the lowest row that still sits above everything under it.
    """
    lt = _strict(elems, leq)
    cover = {(a, b) for (a, b) in lt if not any((a, c) in lt and (c, b) in lt for c in elems)}
    level: dict[Any, int] = {}

    def lv(x: Any) -> int:
        if x not in level:
            level[x] = 1 + max((lv(a) for a in elems if (a, x) in cover), default=-1)
        return level[x]

    for e in elems:
        lv(e)
    return cover, level


def _layout(elems: Sequence[Any], leq: Callable[[Any, Any], bool], cx: float, span: float,
            ybot: float, dy: float, key: Callable[[Any], Any] | None = None):
    """Lay the levels out bottom-to-top, elements of a level spread evenly across `span`."""
    cover, level = _hasse(elems, leq)
    rows: dict[int, list[Any]] = {}
    for e in elems:
        rows.setdefault(level[e], []).append(e)
    pos: dict[Any, Pt] = {}
    for lvl, row in sorted(rows.items()):
        row.sort(key=key or (lambda e: e))
        y = ybot - lvl * dy
        n = len(row)
        for i, e in enumerate(row):
            x = cx if n == 1 else cx - span / 2 + i * span / (n - 1)
            pos[e] = (x, y)
    return cover, level, pos


def _bounds(elems: Sequence[Any], leq: Callable[[Any, Any], bool], sub: Iterable[Any]):
    """Upper bounds, lub, lower bounds, glb of a subset - straight from the definitions."""
    sub = list(sub)
    ub = [x for x in elems if all(leq(a, x) for a in sub)]
    lb = [x for x in elems if all(leq(x, a) for a in sub)]
    least = [u for u in ub if all(leq(u, v) for v in ub)]
    great = [g for g in lb if all(leq(v, g) for v in lb)]
    return ub, (least[0] if len(least) == 1 else None), lb, (great[0] if len(great) == 1 else None)


def _divides(a: int, b: int) -> bool:
    return b % a == 0


# ---------------------------------------------------------------- drawing helpers


def _toward(c: Pt, t: Pt, r: float) -> Pt:
    """The point on the circle of radius r around c, in the direction of t."""
    dx, dy = t[0] - c[0], t[1] - c[1]
    d = math.hypot(dx, dy) or 1.0
    return c[0] + dx / d * r, c[1] + dy / d * r


def _edge(p: Pt, q: Pt, r: float = 19, stroke: str = LINE, sw: float = 1.4,
          dash: str | None = None) -> str:
    """A plain Hasse edge: stops at both node rims so it never runs through a circle."""
    a, b = _toward(p, q, r + 1.5), _toward(q, p, r + 1.5)
    return line(a[0], a[1], b[0], b[1], stroke, sw, dash)


def _curve(p: Pt, q: Pt, r: float, bulge: float, stroke: str = GOLD, sw: float = 1.5,
           dash: str | None = None) -> str:
    """An arrow from circle p to circle q, bowed `bulge` px to the left of its direction."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    d = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / d, dx / d
    ctrl = ((p[0] + q[0]) / 2 + nx * bulge, (p[1] + q[1]) / 2 + ny * bulge)
    a, b = _toward(p, ctrl, r + 1), _toward(q, ctrl, r + 2)
    return path(f"M{a[0]:.1f} {a[1]:.1f} Q{ctrl[0]:.1f} {ctrl[1]:.1f} {b[0]:.1f} {b[1]:.1f}",
                "none", stroke, sw, arrowhead=True, dash=dash)


def _loop(cx: float, cy: float, r: float, ang: float, stroke: str = GOLD, sw: float = 1.5,
          out: float = 23) -> str:
    """A self-loop leaving the node at `ang` degrees (SVG angles: 90 points down)."""
    a = math.radians(ang)
    p1 = (cx + r * math.cos(a - 0.44), cy + r * math.sin(a - 0.44))
    p2 = (cx + r * math.cos(a + 0.44), cy + r * math.sin(a + 0.44))
    far = r + out
    c1 = (cx + far * math.cos(a - 0.90), cy + far * math.sin(a - 0.90))
    c2 = (cx + far * math.cos(a + 0.90), cy + far * math.sin(a + 0.90))
    return path(f"M{p1[0]:.1f} {p1[1]:.1f} C{c1[0]:.1f} {c1[1]:.1f} {c2[0]:.1f} {c2[1]:.1f} "
                f"{p2[0]:.1f} {p2[1]:.1f}", "none", stroke, sw, arrowhead=True)


def _ring(cx: float, cy: float, r: float, stroke: str, dash: str | None = None) -> str:
    return circle(cx, cy, r + 5.5, "none", stroke, 2, stroke_dasharray=dash)


def _hasse_body(cover, pos, r: float = 19, tint: dict[Any, tuple[str, str]] | None = None,
                hot: Iterable[tuple[Any, Any]] = (), size: float = 13) -> str:
    """Edges then nodes, so no edge is drawn over a circle. `hot` edges are gold and thicker."""
    hot = {tuple(e) for e in hot}
    out = ""
    for a, b in sorted(cover, key=lambda e: (str(e[0]), str(e[1]))):
        if (a, b) in hot:
            continue
        out += _edge(pos[a], pos[b], r, LINE, 1.4)
    for a, b in sorted(hot, key=lambda e: (str(e[0]), str(e[1]))):
        out += _edge(pos[a], pos[b], r, GOLD, 2.4)
    for e, (x, y) in sorted(pos.items(), key=lambda kv: str(kv[0])):
        fill, stroke = (tint or {}).get(e, (SURFACE, LINE))
        out += node(x, y, e, r, fill, stroke, size, INK, 1.6)
    return out


# ---------------------------------------------------------------- relations


@register("dsma", "relations")
def _relations() -> str:
    """Three panels on {a, b, c}: self-loops, a two-way pair, a forced shortcut."""
    r, top, pw, ph = 18, 20, 210, 214
    body = ""
    panels = [
        ("Reflexive", "every element loops to itself", "(x, x) in R for all x"),
        ("Symmetric", "arrows always come in pairs", "(x, y) in R gives (y, x) in R"),
        ("Transitive", "a to b to c forces a to c", "(x, y), (y, z) give (x, z)"),
    ]
    for i, (title, cap1, cap2) in enumerate(panels):
        x = 20 + i * 235
        cx = x + pw / 2
        a, b, c = (cx, top + 92), (cx - 50, top + 166), (cx + 50, top + 166)
        body += rect(x, top, pw, ph, SURFACE2, LINE, 12, 1)
        body += text(cx, top + 24, title, 13.5, GOLD, weight=700)
        if i == 0:
            for p in (a, b, c):
                body += _loop(p[0], p[1], r, -90)
        elif i == 1:
            body += _curve(a, b, r, 21) + _curve(b, a, r, 21)
            body += _curve(b, c, r, 21) + _curve(c, b, r, 21)
        else:
            body += _curve(a, b, r, 0, INK2, 1.5) + _curve(b, c, r, 0, INK2, 1.5)
            body += _curve(a, c, r, -24, GOLD, 2, dash="5 4")
            body += text(cx + 74, top + 118, "forced", 11.5, GOLD, "middle", weight=600, italic=True)
        for p, lab in ((a, "a"), (b, "b"), (c, "c")):
            body += node(p[0], p[1], lab, r, SURFACE, LINE, 13, INK, 1.6)
        body += caption(cx, top + ph + 22, cap1, size=12)
        body += text(cx, top + ph + 40, cap2, 11.5, MUTED, mono=True)
    return svg(700, 302, body, "Reflexive, symmetric and transitive relations on three elements")


# ---------------------------------------------------------------- rel-repr


@register("dsma", "rel-repr")
def _rel_repr() -> str:
    """One relation, three representations; the matrix is computed from the edge list."""
    elems = [1, 2, 3]
    edges = [(1, 2), (2, 3)]
    pairs = sorted(set(edges))
    mat = [[1 if (i, j) in set(edges) else 0 for j in elems] for i in elems]

    body = ""
    for x in (236, 466):
        body += line(x, 34, x, 232, LINE, 1)

    # 1. digraph
    r = 18
    cx = 128
    p = {1: (cx - 50, 92), 2: (cx + 50, 92), 3: (cx, 174)}
    body += text(cx, 32, "directed graph", 12.5, GOLD, weight=700)
    for a, b in pairs:
        body += _curve(p[a], p[b], r, 0, INK2, 1.6)
    for e, (x, y) in p.items():
        body += node(x, y, e, r, SURFACE, LINE, 13, INK, 1.6)
    body += caption(cx, 212, "an arrow i to j for each pair", size=11.5)

    # 2. zero-one matrix
    cw = 42
    mx = 351 - cw * 2
    body += text(351, 32, "zero-one matrix", 12.5, GOLD, weight=700)
    rows = [[str(i)] + [mat[i - 1][j - 1] for j in elems] for i in elems]
    body += table(mx, 62, [""] + [str(j) for j in elems], rows, cw, 30,
                  cell_fill=lambda i, j, v: GOLD_SOFT if j and v == 1 else "")
    body += caption(351, 212, "m[i][j] = 1 exactly when (i, j) is in R", size=11.5)

    # 3. set of ordered pairs
    body += text(578, 32, "set of ordered pairs", 12.5, GOLD, weight=700)
    body += rect(486, 62, 184, 120, GOLD_WASH, LINE, 10, 1)
    body += lines(578, 88, ["A = {1, 2, 3}",
                            "R = {(1, 2), (2, 3)}"], 12.5, INK, leading=22, mono=True)
    body += lines(578, 136, [f"|R| = {len(pairs)} of the",
                             f"{len(elems) ** 2} pairs in A x A"], 12, MUTED, leading=17)
    body += caption(578, 212, "R is just a subset of A x A", size=11.5)

    body += caption(350, 254, "the same relation three times - the arrows, the ones in the matrix "
                              "and the listed pairs all say (1,2) and (2,3)")
    return svg(700, 272, body, "A relation shown as a digraph, a zero-one matrix and a set of pairs")


# ---------------------------------------------------------------- closures


def _warshall(n: int, edges: Iterable[tuple[int, int]]):
    """Run Warshall and record, for each k, the pairs that step added."""
    m = set(edges)
    steps = []
    for k in range(1, n + 1):
        new = []
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if (i, j) not in m and (i, k) in m and (k, j) in m:
                    m.add((i, j))
                    new.append((i, j))
        steps.append((k, sorted(new), set(m)))
    return steps


@register("dsma", "closures")
def _closures() -> str:
    """The transitive closure built up a step at a time, new edges in gold."""
    chain = [2, 1, 3, 4]               # the nodes laid out left to right
    base = [(2, 1), (1, 3), (3, 4)]
    steps = _warshall(4, base)
    live = [s for s in steps if s[1]]
    quiet = [k for k, new, _ in steps if not new]

    r = 18
    xs = {e: 140 + i * 125 for i, e in enumerate(chain)}
    idx = {e: i for i, e in enumerate(chain)}
    ys = [70, 195, 320]
    rows = [("R", "given", set(base), [])]
    for n, (k, new, cur) in enumerate(live[:2]):
        rows.append((f"after k = {k}", "= R+" if n == len(live[:2]) - 1 else "", cur, new))

    body = ""
    for y, (tag, sub, cur, new) in zip(ys, rows):
        body += text(28, y + (0 if sub else 4.5), tag, 12.5, INK, "start", weight=700)
        if sub:
            body += text(28, y + 17, sub, 11, MUTED, "start", mono=True)
        for a, b in sorted(cur):
            d = abs(idx[b] - idx[a])
            col, sw = (GOLD, 1.9) if (a, b) in new else (INK2, 1.4)
            if d == 1:
                body += _curve((xs[a], y), (xs[b], y), r, 0, col, sw)
            else:
                # long hops go under the chain, short ones over it, so no two arrowheads collide
                sgn = (1 if xs[b] > xs[a] else -1) * (-1 if d > 2 else 1)
                body += _curve((xs[a], y), (xs[b], y), r, -sgn * (30 + 46 * (d - 1)), col, sw)
        if new:
            body += text(566, y + 4.5, "new: " + ", ".join(f"({i},{j})" for i, j in new),
                         11.5, GOLD, "start", weight=600)
        for e in chain:
            body += node(xs[e], y, e, r, SURFACE, LINE, 13, INK, 1.6)

    body += legend(104, 422, [(INK2, "edges already in R"), (GOLD, "added by this step")], 230)
    body += caption(350, 450, f"Warshall adds (i, j) when i to k and k to j already hold; "
                              f"k = {' and '.join(map(str, quiet))} add nothing here.")
    return svg(700, 466, body, "Transitive closure of a relation built up by Warshall's algorithm")


# ---------------------------------------------------------------- equiv-rel


@register("dsma", "equiv-rel")
def _equiv_rel() -> str:
    """Congruence mod 3 cutting {0..8} into three disjoint classes."""
    universe = list(range(9))
    classes: dict[int, list[int]] = {}
    for n in universe:
        classes.setdefault(n % 3, []).append(n)

    body = rect(20, 40, 660, 284, SURFACE, LINE, 14, 1)
    body += text(36, 62, "S = {0, 1, ..., 8}", 12.5, MUTED, "start", mono=True)
    r, br = 17, 98
    for k in sorted(classes):
        cx = 130 + k * 220
        cy = 182
        members = classes[k]
        body += circle(cx, cy, br, GOLD_WASH, GOLD, 1.4, stroke_dasharray="6 5")
        spots = [(cx, cy - 44), (cx - 48, cy + 26), (cx + 48, cy + 26)]
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                a, b = _toward(spots[i], spots[j], r + 2), _toward(spots[j], spots[i], r + 2)
                body += line(a[0], a[1], b[0], b[1], GOLD, 1.4,
                             marker_end="url(#dg-ag)", marker_start="url(#dg-ag)")
        for p, ang in zip(spots, (-90, 135, 45)):
            body += _loop(p[0], p[1], r, ang, GOLD, 1.3, 18)
        for p, m in zip(spots, members):
            body += node(p[0], p[1], m, r, SURFACE, GOLD, 12.5, INK, 1.6)
        body += text(cx, cy + 78, f"[{k}] = {{{', '.join(map(str, members))}}}", 12.5, INK,
                     weight=600)
        body += text(cx, cy + 120, f"n ≡ {k} (mod 3)", 11.5, MUTED, mono=True)

    body += caption(350, 348, "self-loops are reflexivity, the two-way arrows are symmetry, and "
                              "every two-step hop stays in the class - transitivity")
    body += caption(350, 368, "the classes are disjoint, cover S, and no arrow ever crosses "
                              "between them")
    return svg(700, 384, body, "The set 0 to 8 partitioned into three equivalence classes mod 3")


# ---------------------------------------------------------------- posets


@register("dsma", "posets")
def _posets() -> str:
    """Divisibility on the divisors of 12, drawn from computed covers."""
    elems = [1, 2, 3, 4, 6, 12]
    cover, level, pos = _layout(elems, _divides, cx=200, span=130, ybot=276, dy=66)
    body = _hasse_body(cover, pos, 19)

    # one of the transitive edges a Hasse diagram deliberately leaves out, drawn clear of the nodes
    a, b = _toward(pos[1], (52, 266), 21), _toward(pos[12], (52, 88), 21)
    body += path(f"M{a[0]:.1f} {a[1]:.1f} C52 266 52 88 {b[0]:.1f} {b[1]:.1f}", "none", MUTED,
                 1.2, dash="4 4")
    body += lines(24, 68, ["1 | 12 is true", "but not drawn -", "it follows"], 11, MUTED, "start",
                  leading=14)

    body += text(pos[12][0] + 28, pos[12][1] + 4.5, "greatest", 11.5, MUTED, "start", italic=True)
    body += text(pos[1][0] + 28, pos[1][1] + 4.5, "least", 11.5, MUTED, "start", italic=True)
    body += line(380, 40, 380, 300, LINE, 1)

    cov = [f"{a} | {b}" for a, b in sorted(cover)]
    implied = sorted(_strict(elems, _divides) - cover)
    body += text(400, 62, "edges = covering pairs", 12.5, GOLD, "start", weight=700)
    body += lines(400, 86, [",  ".join(cov[i:i + 3]) for i in range(0, len(cov), 3)], 12, INK2,
                  "start", leading=19, mono=True)
    body += text(400, 158, "true but never drawn", 12.5, GOLD, "start", weight=700)
    body += text(400, 182, ", ".join(f"{a}|{b}" for a, b in implied), 12, INK2, "start", mono=True)
    body += text(400, 216, "partial, not total", 12.5, GOLD, "start", weight=700)
    body += lines(400, 240, ["2 and 3 are incomparable:",
                             "neither 2 | 3 nor 3 | 2 holds,",
                             "so neither sits below the other"], 12, INK2, "start", leading=18)
    body += caption(350, 324, "a Hasse diagram draws x below y with no arrowheads; the reflexive "
                              "and transitive edges are understood")
    return svg(700, 340, body, "Hasse diagram for divisibility on 1, 2, 3, 4, 6, 12")


# ---------------------------------------------------------------- extremal


@register("dsma", "extremal")
def _extremal() -> str:
    """Divisors of 36 with the bounds of a subset marked - all of them computed."""
    elems = [1, 2, 3, 4, 6, 9, 12, 18, 36]
    sub = [4, 6]
    ub, lub, lb, glb = _bounds(elems, _divides, sub)
    cover, level, pos = _layout(elems, _divides, cx=352, span=240, ybot=330, dy=62)
    r = 19

    tint = {e: (GOLD_SOFT, GOLD) for e in sub}
    if lub is not None:
        tint[lub] = (GOOD_SOFT, GOOD)
    if glb is not None:
        tint[glb] = (GOOD_SOFT, GOOD)
    body = ""
    for e in ub:
        body += _ring(pos[e][0], pos[e][1], r, GOOD if e == lub else INFO,
                      None if e == lub else "4 4")
    for e in lb:
        body += _ring(pos[e][0], pos[e][1], r, GOOD if e == glb else WARN,
                      None if e == glb else "4 4")
    body += _hasse_body(cover, pos, r, tint)

    # the highlighted subset
    xs = sorted(pos[e][0] for e in sub)
    ysub = pos[sub[0]][1]
    body += rect(xs[0] - 32, ysub - 30, xs[-1] - xs[0] + 64, 60, "none", GOLD, 30, 1.6,
                 stroke_dasharray="6 5")
    body += text(xs[0] - 52, ysub + 4.5, f"A = {{{', '.join(map(str, sub))}}}", 12.5, GOLD, "end",
                 weight=700)

    body += text(pos[36][0], pos[36][1] - 34, "upper bound (not least)", 11.5, INFO, weight=600)
    body += text(pos[lub][0] - 36, pos[lub][1] - 1, f"lub A = {lub}", 12.5, GOOD, "end",
                 weight=700)
    body += text(pos[lub][0] - 36, pos[lub][1] + 15, "= lcm(4, 6)", 11, MUTED, "end")
    body += text(pos[glb][0] - 36, pos[glb][1] - 1, f"glb A = {glb}", 12.5, GOOD, "end",
                 weight=700)
    body += text(pos[glb][0] - 36, pos[glb][1] + 15, "= gcd(4, 6)", 11, MUTED, "end")
    body += text(pos[1][0], pos[1][1] + 38, "lower bound (not greatest)", 11.5, WARN, weight=600)

    body += text(556, 108, "upper bounds", 12, INFO, "start", weight=700)
    body += text(556, 128, "{" + ", ".join(map(str, ub)) + "}", 12, INK2, "start", mono=True)
    body += text(556, 162, "lower bounds", 12, WARN, "start", weight=700)
    body += text(556, 182, "{" + ", ".join(map(str, lb)) + "}", 12, INK2, "start", mono=True)
    body += lines(556, 222, ["the least of the", "upper bounds is", "the lub; the",
                             "greatest of the", "lower bounds is", "the glb"], 11.5, MUTED,
                  "start", leading=16)

    body += legend(74, 400, [(GOLD_SOFT, "subset A"), (INFO, "upper bound"),
                             (WARN, "lower bound"), (GOOD, "lub / glb")], 140)
    body += caption(350, 428, "divisors of 36 under divisibility; bounds are taken over the whole "
                              "poset, not just over A")
    return svg(700, 444, body, "Hasse diagram of the divisors of 36 with the bounds of the subset "
                               "{4, 6} marked")


# ---------------------------------------------------------------- lattices


@register("dsma", "lattices")
def _lattices() -> str:
    """Divisors of 12: a join and a meet worked out with gcd and lcm."""
    elems = [1, 2, 3, 4, 6, 12]
    a, b = 4, 6
    join = a * b // math.gcd(a, b)
    meet = math.gcd(a, b)
    cover, level, pos = _layout(elems, _divides, cx=176, span=116, ybot=286, dy=70)
    r = 19

    tint = {a: (GOLD_SOFT, GOLD), b: (GOLD_SOFT, GOLD),
            join: (GOOD_SOFT, GOOD), meet: (GOOD_SOFT, GOOD)}
    hot = [e for e in cover if set(e) <= {a, b, join, meet}]
    body = _hasse_body(cover, pos, r, tint, hot)
    body += text(pos[a][0] - 30, pos[a][1] + 4.5, "a = 4", 11.5, GOLD, "end", weight=700)
    body += text(pos[b][0] + 30, pos[b][1] + 4.5, "b = 6", 11.5, GOLD, "start", weight=700)
    body += text(pos[join][0], pos[join][1] - 32, f"a ∨ b = {join}   (join)", 12, GOOD,
                 weight=700)
    body += text(pos[meet][0] - 30, pos[meet][1] + 4.5, f"a ∧ b = {meet}", 12, GOOD, "end",
                 weight=700)
    body += text(pos[meet][0] - 30, pos[meet][1] + 20, "(meet)", 11, MUTED, "end")
    body += line(336, 40, 336, 306, LINE, 1)

    body += rect(360, 52, 320, 76, GOLD_WASH, LINE, 10, 1)
    body += lines(520, 78, [f"a ∨ b = lub{{4, 6}} = lcm(4, 6) = {join}",
                            f"a ∧ b = glb{{4, 6}} = gcd(4, 6) = {meet}"], 12.5, INK,
                  leading=24)
    pairs = [(2, 3), (3, 4), (4, 6), (6, 12)]
    rows = [[x, y, x * y // math.gcd(x, y), math.gcd(x, y)] for x, y in pairs]
    body += text(520, 154, "every pair has both, so it is a lattice", 12, GOLD, weight=700)
    body += table(380, 168, ["x", "y", "x ∨ y", "x ∧ y"], rows, 70, 26, mono=False)
    body += caption(350, 324, "divisors of 12: the join is the lcm, the meet is the gcd, and both "
                              "always land back inside the set")
    return svg(700, 340, body, "Divisors of 12 as a lattice, with the join and meet of 4 and 6")
