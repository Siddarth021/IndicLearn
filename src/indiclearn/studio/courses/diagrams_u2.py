"""Figures for DSMA Unit 2 - Sets, Functions, Sequences, Induction, Recursion.

One builder per lesson topic, registered on the shared registry in `diagrams.py`:

    sets         three-circle Venn diagram inside the universal set, A n B n C shaded
    functions    a bijection A -> B, beside the two ways a function can fail to be one
    cardinality  the bijection N -> Z that makes Z countably infinite
    sums         the nested-loop grid of a double summation, the j <= i triangle counted
    induction    the domino chain: base case tipped, P(k) => P(k+1), the n-th still standing
    recursion    the call tree of F(4) unfolding down to its base cases

Everything is computed (circle intersections, region label positions, the mapping, the
inner sums, the tree layout), so no number or coordinate in a figure is typed by hand.
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
    MUTED,
    SURFACE,
    SURFACE2,
    arrow,
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

# ---------------------------------------------------------------- small shared helpers


def _oval(cx: float, cy: float, rx: float, ry: float, fill: str = SURFACE,
          stroke: str = MUTED, sw: float = 1.4) -> str:
    """An ellipse - the set bubble of a mapping diagram (no primitive covers ellipses)."""
    return (f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def _rotated(cx: float, cy: float, deg: float, body: str) -> str:
    """Rotate a piece of the drawing about a point (used for the tipped domino)."""
    return f'<g transform="rotate({deg:.2f} {cx:.1f} {cy:.1f})">{body}</g>'


def _pill(cx: float, cy: float, s: str, fill: str, stroke: str, text_fill: str,
          w: float = 0, size: float = 12) -> str:
    """A small rounded tag, e.g. 'bijection'."""
    w = w or (len(s) * size * 0.60 + 20)
    return rect(cx - w / 2, cy - 11, w, 22, fill, stroke, 11, 1) + \
        text(cx, cy + 4, s, size, text_fill, weight=600)


def _map_arrows(ax: float, bx: float, ys_a: list[float], ys_b: list[float],
                pairs: list[tuple[int, int]], color: str = GOLD, sw: float = 1.6) -> str:
    """Arrows from element i of the left bubble to element j of the right bubble."""
    return "".join(arrow(ax, ys_a[i], bx, ys_b[j], color, sw) for i, j in pairs)


# ---------------------------------------------------------------- sets

_SET_A = (1, 2, 3, 5)
_SET_B = (2, 3, 4, 6)
_SET_C = (3, 5, 6, 7)
_UNIVERSE = tuple(range(1, 9))


def _circle_pair(p: tuple[float, float], q: tuple[float, float], r: float,
                 near: tuple[float, float]) -> tuple[float, float]:
    """The intersection point of two equal circles that lies on the `near` side."""
    mx, my = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2
    dx, dy = q[0] - p[0], q[1] - p[1]
    d = math.hypot(dx, dy)
    h = math.sqrt(max(r * r - (d / 2) ** 2, 0.0))
    ux, uy = -dy / d, dx / d
    c1 = (mx + ux * h, my + uy * h)
    c2 = (mx - ux * h, my - uy * h)
    return c1 if math.dist(c1, near) < math.dist(c2, near) else c2


def _arc_points(c: tuple[float, float], r: float, start: tuple[float, float],
                end: tuple[float, float], steps: int = 26) -> list[tuple[float, float]]:
    """The short arc of circle (c, r) from `start` to `end`, as sampled points."""
    a0 = math.atan2(start[1] - c[1], start[0] - c[0])
    a1 = math.atan2(end[1] - c[1], end[0] - c[0])
    delta = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi  # shortest signed sweep
    return [(c[0] + r * math.cos(a0 + delta * k / steps),
             c[1] + r * math.sin(a0 + delta * k / steps)) for k in range(steps + 1)]


def _triple_region(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float],
                   r: float) -> str:
    """The exact boundary of A n B n C: one arc of each circle between the pairwise crossings."""
    p_ab = _circle_pair(a, b, r, c)   # on circles A and B, inside C
    p_bc = _circle_pair(b, c, r, a)   # on circles B and C, inside A
    p_ca = _circle_pair(c, a, r, b)   # on circles C and A, inside B
    pts = (_arc_points(a, r, p_ab, p_ca) + _arc_points(c, r, p_ca, p_bc)
           + _arc_points(b, r, p_bc, p_ab))
    d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z"
    return d


def _best_spot(centres: dict[str, tuple[float, float]], r: float, inside: str,
               box_xy: tuple[float, float, float, float]) -> tuple[float, float]:
    """The point of a Venn region deepest inside it - the roomiest place for its label.

    `inside` names the circles the point must be inside; the others it must be outside.
    We maximise the smallest distance to any boundary, so the label never sits on a curve.
    """
    x0, y0, x1, y1 = box_xy
    best, best_m = (x0, y0), -1e9
    steps = 90
    for i in range(steps + 1):
        px = x0 + (x1 - x0) * i / steps
        for j in range(steps + 1):
            py = y0 + (y1 - y0) * j / steps
            margin = 1e9
            for name, ctr in centres.items():
                d = math.dist((px, py), ctr)
                margin = min(margin, (r - d) if name in inside else (d - r))
            if margin > best_m:
                best, best_m = (px, py), margin
    return best


@register("dsma", "sets")
def _sets() -> str:
    r = 88.0
    d = 103.0                      # distance between circle centres
    h = d / math.sqrt(3.0)         # centroid -> centre distance
    cx, cy = 212.0, 188.0
    A = (cx - d / 2, cy - h / 2)
    B = (cx + d / 2, cy - h / 2)
    C = (cx, cy + h)
    centres = {"A": A, "B": B, "C": C}

    left, right = cx - d / 2 - r, cx + d / 2 + r
    top, bottom = cy - h / 2 - r, cy + h + r
    pad = 26.0
    ux, uy = left - pad, top - pad
    uw, uh = (right - left) + 2 * pad, (bottom - top) + 2 * pad

    body = rect(ux, uy, uw, uh, SURFACE2, LINE, 10)
    body += text(ux + 14, uy + 22, "U", 14, MUTED, "start", weight=700, italic=True)

    # the genuine triple intersection, drawn from its own three arcs
    body += path(_triple_region(A, B, C, r), GOLD_SOFT, "none", 0)

    for name in ("A", "B", "C"):
        body += circle(*centres[name], r, "none", MUTED, 1.5)

    # circle names, pushed outward from the centroid so they sit in the lens-free part
    for name, ctr in centres.items():
        vx, vy = ctr[0] - cx, ctr[1] - cy
        n = math.hypot(vx, vy)
        body += text(ctr[0] + vx / n * r * 0.80, ctr[1] + vy / n * r * 0.80 + 5,
                     name, 15, INK, weight=700, italic=True)

    # one element per Venn region, each placed where its region is widest
    sets = {"A": set(_SET_A), "B": set(_SET_B), "C": set(_SET_C)}
    bbox = (left, top, right, bottom)
    for elem in _UNIVERSE:
        inside = "".join(k for k in "ABC" if elem in sets[k])
        if not inside:
            continue
        px, py = _best_spot(centres, r, inside, bbox)
        gold = len(inside) == 3
        body += text(px, py + 5, elem, 15 if gold else 13.5,
                     GOLD if gold else INK2, weight=700 if gold else 500, mono=True)

    outside = sorted(set(_UNIVERSE) - set(_SET_A) - set(_SET_B) - set(_SET_C))
    body += text(ux + 26, uy + uh - 20, ", ".join(str(v) for v in outside), 13.5,
                 MUTED, "start", weight=500, mono=True)
    body += caption(ux + 26, uy + uh - 38, "in U only", "start")

    # the key column
    kx, ky, kw = 418.0, 128.0, 258.0
    body += rect(kx, ky, kw, 196, SURFACE, LINE, 10)
    rows = [(f"A = {{{', '.join(map(str, _SET_A))}}}", INK2),
            (f"B = {{{', '.join(map(str, _SET_B))}}}", INK2),
            (f"C = {{{', '.join(map(str, _SET_C))}}}", INK2)]
    for i, (s, col) in enumerate(rows):
        body += text(kx + 18, ky + 32 + i * 26, s, 13, col, "start", mono=True)
    body += line(kx + 14, ky + 118, kx + kw - 14, ky + 118, LINE, 1)
    triple = sorted(set(_SET_A) & set(_SET_B) & set(_SET_C))
    body += rect(kx + 18, ky + 139, 12, 12, GOLD_SOFT, GOLD2, 3, 1)
    body += text(kx + 38, ky + 149,
                 f"A ∩ B ∩ C = {{{', '.join(map(str, triple))}}}",
                 13, GOLD, "start", weight=600, mono=True)
    body += text(kx + 18, ky + 177, f"U = {{1, …, {max(_UNIVERSE)}}}", 12.5,
                 MUTED, "start", mono=True)

    cap = ("The shaded centre is A ∩ B ∩ C — the elements that lie in all "
           "three sets at once.")
    body += caption(350, uy + uh + 26, cap)
    return svg(700, int(uy + uh + 44), body,
               "Three overlapping circles A, B and C inside a universal set rectangle U, "
               "with the central region A intersect B intersect C shaded")


# ---------------------------------------------------------------- functions


def _bubble(cx: float, cy: float, rx: float, ry: float, title: str, labels: list[str],
            gap: float, size: float = 13) -> tuple[str, list[float]]:
    """A set bubble with its elements stacked down the middle; returns the element y's."""
    out = _oval(cx, cy, rx, ry)
    out += text(cx, cy - ry - 11, title, 12.5, MUTED, weight=600)
    ys = [cy + (i - (len(labels) - 1) / 2) * gap for i in range(len(labels))]
    for y, lab in zip(ys, labels):
        out += text(cx, y + size / 3, lab, size, INK, weight=600, mono=True)
    return out, ys


@register("dsma", "functions")
def _functions() -> str:
    body = ""

    # --- the bijection, the figure the lesson asks for
    ax, bx, cy = 118.0, 322.0, 150.0
    rx, ry = 52.0, 84.0
    left, ys_a = _bubble(ax, cy, rx, ry, "A  (domain)", ["a", "b", "c"], 50)
    rightb, ys_b = _bubble(bx, cy, rx, ry, "B  (codomain)", ["1", "2", "3"], 50)
    body += rect(26, 40, 390, 290, SURFACE, LINE, 12)
    body += left + rightb
    body += _map_arrows(ax + 26, bx - 26, ys_a, ys_b, [(0, 0), (1, 1), (2, 2)])
    body += text((ax + bx) / 2, cy - 62, "f", 15, GOLD, weight=700, italic=True)
    body += _pill(220, cy + 122, "bijection: one-to-one and onto",
                  GOOD_SOFT, GOOD, GOOD, size=12)
    body += caption(220, cy + 152, "every element of B receives exactly one arrow")

    # --- the two ways it can fail, for contrast
    def mini(y0: float, title: str, a_lab: list[str], b_lab: list[str],
             pairs: list[tuple[int, int]], note: str, flag: int | None) -> str:
        out = rect(432, y0, 244, 136, SURFACE, LINE, 10)
        out += text(554, y0 + 21, title, 12, INK2, weight=600)
        mx, nx, mcy = 492.0, 616.0, y0 + 74
        la, ysa = _bubble(mx, mcy, 28.0, 40.0, "", a_lab, 28, 12)
        lb, ysb = _bubble(nx, mcy, 28.0, 40.0, "", b_lab, 28, 12)
        out += la + lb
        out += _map_arrows(mx + 15, nx - 15, ysa, ysb, pairs, GOLD, 1.3)
        if flag is not None:
            out += circle(nx, ysb[flag], 12.5, BAD_SOFT, BAD, 1.2)
            out += text(nx, ysb[flag] + 4, b_lab[flag], 12, BAD, weight=600, mono=True)
        out += caption(554, y0 + 126, note)
        return out

    body += mini(40, "one-to-one, not onto", ["a", "b"], ["1", "2", "3"],
                 [(0, 0), (1, 1)], "3 has no pre-image", 2)
    body += mini(194, "onto, not one-to-one", ["a", "b", "c"], ["1", "2"],
                 [(0, 0), (1, 0), (2, 1)], "a and b share the value 1", None)

    body += caption(350, 356, "f ⁻¹ exists only for the bijection on the left.")
    return svg(700, 374, body,
               "A bijection from domain A to codomain B drawn with arrows, beside two "
               "smaller diagrams showing a function that is one-to-one but not onto and "
               "one that is onto but not one-to-one")


# ---------------------------------------------------------------- cardinality


def _nat_to_int(n: int) -> int:
    """The standard bijection N -> Z: evens go to the positives, odds to the negatives."""
    return n // 2 if n % 2 == 0 else -(n + 1) // 2


def _signed(v: int) -> str:
    return f"−{-v}" if v < 0 else str(v)


@register("dsma", "cardinality")
def _cardinality() -> str:
    ns = list(range(5))
    xs = [128.0 + i * 96.0 for i in range(len(ns))]
    y_top, y_bot = 74.0, 188.0
    r = 21.0
    body = ""

    body += text(58, y_top + 6, "ℕ", 17, MUTED, weight=700)
    body += caption(58, y_top + 26, "naturals")
    body += text(58, y_bot + 6, "ℤ", 17, MUTED, weight=700)
    body += caption(58, y_bot + 26, "integers")

    for x, n in zip(xs, ns):
        img = _nat_to_int(n)
        body += node(x, y_top, n, r, SURFACE, MUTED, 13.5, INK)
        body += node(x, y_bot, _signed(img), r, GOLD_SOFT, GOLD, 13.5, INK)
        body += arrow(x, y_top + r, x, y_bot - r, GOLD, 1.7)
        body += caption(x + 26, (y_top + y_bot) / 2 + 4, f"f({n})", "start")

    xe = xs[-1] + 96.0
    body += circle(xe, y_top, r, SURFACE, MUTED, 1.2, stroke_dasharray="4 3")
    body += text(xe, y_top + 5, "…", 15, MUTED, weight=700)
    body += circle(xe, y_bot, r, GOLD_WASH, GOLD2, 1.2, stroke_dasharray="4 3")
    body += text(xe, y_bot + 5, "…", 15, GOLD, weight=700)
    body += arrow(xe, y_top + r, xe, y_bot - r, GOLD, 1.4, dash="4 3")

    # the rule, and the first few values it produces, both computed from _nat_to_int
    seq = ", ".join(_signed(_nat_to_int(n)) for n in range(8))
    body += rect(96, 240, 508, 70, SURFACE, LINE, 10)
    body += text(118, 264, "f(n) = n/2 for even n,   −(n+1)/2 for odd n", 13,
                 INK2, "start", weight=600)
    body += text(118, 290, f"f(0), f(1), f(2), … = {seq}, …", 12.5,
                 GOLD, "start", mono=True)

    body += caption(350, 334, "Every integer is hit exactly once, so f is a bijection: "
                              "|ℤ| = |ℕ| = ℵ₀ (countably infinite).")
    return svg(700, 352, body,
               "Natural numbers 0 to 4 on the top row mapped by arrows to the integers "
               "0, -1, 1, -2, 2 on the bottom row")


# ---------------------------------------------------------------- sums

_SUM_N = 5


def _inner_sum(i: int) -> int:
    """Sum over j = 1..i of (i + j) - evaluated term by term, not from a formula."""
    return sum(i + j for j in range(1, i + 1))


@register("dsma", "sums")
def _sums() -> str:
    n = _SUM_N
    cell = 44.0
    gx, gy = 104.0, 92.0
    body = ""

    body += text(gx + n * cell / 2, gy - 44, "j  →  (inner loop, j = 1 … i)",
                 12.5, MUTED, weight=600)
    for j in range(1, n + 1):
        body += text(gx + (j - 1) * cell + cell / 2, gy - 12, j, 12.5, INK2,
                     weight=600, mono=True)
    body += text(gx - 58, gy + n * cell / 2 - 12, "i", 12.5, MUTED, weight=600)
    body += text(gx - 58, gy + n * cell / 2 + 6, "↓", 12.5, MUTED, weight=600)
    body += caption(gx - 58, gy + n * cell / 2 + 26, "outer")

    for i in range(1, n + 1):
        yy = gy + (i - 1) * cell
        body += text(gx - 18, yy + cell / 2 + 4.5, i, 12.5, INK2, "middle",
                     weight=600, mono=True)
        for j in range(1, n + 1):
            xx = gx + (j - 1) * cell
            counted = j <= i
            body += rect(xx, yy, cell, cell,
                         GOLD_SOFT if counted else SURFACE2,
                         GOLD2 if counted else LINE, 0, 1)
            if counted:
                body += text(xx + cell / 2, yy + cell / 2 + 4.5, i + j, 12.5, GOLD,
                             weight=600, mono=True)
            else:
                body += text(xx + cell / 2, yy + cell / 2 + 4, "·", 12.5, LINE)

    # the staircase boundary j = i, so the triangle is unmistakable
    steps = ["M" + f"{gx:.1f} {gy:.1f}"]
    for i in range(1, n + 1):
        steps.append(f"L{gx + i * cell:.1f} {gy + (i - 1) * cell:.1f}")
        steps.append(f"L{gx + i * cell:.1f} {gy + i * cell:.1f}")
    body += path(" ".join(steps), "none", GOLD, 2)

    gw = n * cell
    shaded = sum(range(1, n + 1))
    body += caption(gx + gw / 2, gy + gw + 28,
                    f"{shaded} shaded cells — the ones with j ≤ i")

    # right panel: the inner sum per row and the grand total
    px = 350.0
    body += rect(px, 58, 326, 272, SURFACE, LINE, 10)
    body += lines(px + 163, 86, ["for a fixed row i, the inner sum is",
                                 "Σ (i + j) over j = 1 … i  =  i² + i(i+1)/2"],
                  12.5, INK2, leading=20)
    rows = [[i, _inner_sum(i), i * i + i * (i + 1) // 2] for i in range(1, n + 1)]
    assert all(a == b for _, a, b in rows)      # the closed form matches the term-by-term sum
    body += table(px + 36, 130, ["i", "inner sum", "i²+i(i+1)/2"],
                  [[r[0], r[1], r[2]] for r in rows], [52, 92, 110], 26,
                  highlight=1)
    total = sum(_inner_sum(i) for i in range(1, n + 1))
    body += text(px + 163, 312,
                 f"outer sum over i = 1 … {n}  =  {total}", 13, GOLD, weight=700)

    body += caption(350, 366, f"The double sum Σᵢ Σⱼ (i + j) adds one "
                              f"shaded cell at a time, row by row (here n = {n}).")
    return svg(700, 384, body,
               "A five by five grid of a nested loop, the triangular region where j is at "
               "most i shaded and each shaded cell labelled with i plus j")


# ---------------------------------------------------------------- induction

_DOM_W, _DOM_H = 34.0, 86.0


def _domino(x: float, ground: float, fill: str, stroke: str, dash: str | None = None) -> str:
    """One upright domino standing on the ground line, x is its left edge."""
    d = {"stroke_dasharray": dash} if dash else {}
    out = rect(x, ground - _DOM_H, _DOM_W, _DOM_H, fill, stroke, 4, 1.6, **d)
    out += line(x + 7, ground - _DOM_H / 2, x + _DOM_W - 7, ground - _DOM_H / 2, stroke, 1)
    return out


def _knock_arc(x1: float, x2: float, y: float, rise: float, dash: str | None = None) -> str:
    """The gold arc from the top of one domino to the top of the next."""
    return path(f"M{x1:.1f} {y:.1f} Q{(x1 + x2) / 2:.1f} {y - rise:.1f} {x2:.1f} {y:.1f}",
                "none", GOLD, 1.7, arrowhead=True, dash=dash)


@register("dsma", "induction")
def _induction() -> str:
    ground = 196.0
    top = ground - _DOM_H
    x1 = 46.0                               # the tipped one; it pivots on its right edge
    xs = [160.0, 250.0, 340.0]              # P(2), P(3), P(4)
    xn = 500.0                              # P(n), still standing
    mid = _DOM_W / 2
    body = line(36, ground, 676, ground, MUTED, 1.4)

    # P(1): pushed over, and proven - the base case
    body += _rotated(x1 + _DOM_W, ground, 62, _domino(x1, ground, GOOD_SOFT, GOOD))
    body += text(x1 + 66, ground + 22, "P(1)", 13, GOOD, weight=700)
    body += caption(x1 + 66, ground + 40, "base case ✓")

    for i, x in enumerate(xs, start=2):
        body += _domino(x, ground, SURFACE, GOLD)
        body += text(x + mid, ground + 22, f"P({i})", 13, INK2, weight=600)

    body += _domino(xn, ground, GOLD_SOFT, GOLD, dash="6 4")
    body += text(xn + mid, ground + 22, "P(n)", 13, GOLD, weight=700)
    body += caption(xn + mid, ground + 40, "still standing")
    for k in range(3):                      # the dashed continuation beyond P(n)
        body += _domino(572.0 + k * 34, ground, "none", LINE, dash="4 5")

    # the implication chain, one arc per knock
    arc_y = top - 8
    body += _knock_arc(x1 + 78, xs[0] + mid, arc_y, 30)
    body += _knock_arc(xs[0] + mid, xs[1] + mid, arc_y, 34)
    body += _knock_arc(xs[1] + mid, xs[2] + mid, arc_y, 34)
    body += _knock_arc(xs[2] + mid, xn + mid, arc_y, 54, dash="6 4")
    body += text(235, top - 62, "inductive step:  P(k) ⇒ P(k+1)", 13, GOLD, weight=700)
    body += text(500, top - 62, "… and so on, for every n", 12, MUTED)

    body += caption(350, 272, "Tip the first domino and make sure each one knocks over "
                              "the next: then P(n) holds for every n ≥ 1.")
    return svg(700, 290, body,
               "A row of dominoes: the first has been tipped over as the base case, gold "
               "arrows show each one knocking over the next, and the n-th domino is still "
               "standing with a dashed continuation behind it")


# ---------------------------------------------------------------- recursion


class _Call:
    """One node of a recursion tree."""

    __slots__ = ("n", "kids", "x", "y", "value")

    def __init__(self, n: int, kids: list["_Call"]):
        self.n, self.kids = n, kids
        self.x = self.y = 0.0
        self.value = 0


def _fib_calls(n: int) -> _Call:
    """The call tree of the naive Fibonacci, with each node's returned value."""
    if n <= 1:
        leaf = _Call(n, [])
        leaf.value = n
        return leaf
    kids = [_fib_calls(n - 1), _fib_calls(n - 2)]
    me = _Call(n, kids)
    me.value = kids[0].value + kids[1].value
    return me


def _layout(root: _Call, x0: float, gap: float, y0: float, dy: float) -> None:
    """Leaves get evenly spaced columns; a parent sits above the middle of its children."""
    counter = [0]

    def walk(c: _Call, depth: int) -> None:
        c.y = y0 + depth * dy
        if not c.kids:
            c.x = x0 + counter[0] * gap
            counter[0] += 1
            return
        for k in c.kids:
            walk(k, depth + 1)
        c.x = sum(k.x for k in c.kids) / len(c.kids)

    walk(root, 0)


def _walk(c: _Call) -> list[_Call]:
    out = [c]
    for k in c.kids:
        out += _walk(k)
    return out


@register("dsma", "recursion")
def _recursion() -> str:
    root = _fib_calls(4)
    _layout(root, 90.0, 120.0, 74.0, 84.0)
    nodes = _walk(root)
    leaves = [c for c in nodes if not c.kids]
    r = 24.0
    body = ""

    for c in nodes:
        for k in c.kids:
            body += line(c.x, c.y + r, k.x, k.y - r, LINE, 1.4)

    for c in nodes:
        base = not c.kids
        if c is root:
            fill, stroke = GOLD_SOFT, GOLD
        elif base:
            fill, stroke = GOOD_SOFT, GOOD
        else:
            fill, stroke = SURFACE, GOLD
        body += node(c.x, c.y, f"F({c.n})", r, fill, stroke, 13, INK)
        body += text(c.x, c.y + r + 16, f"= {c.value}", 12,
                     GOOD if base else GOLD, weight=600, mono=True)

    body += rect(496, 30, 182, 92, SURFACE, LINE, 10)
    body += lines(587, 52, ["F(0) = 0,  F(1) = 1", "F(n) = F(n−1) + F(n−2)"],
                  12.5, INK2, leading=22, mono=True)
    body += caption(587, 110, "base cases, then the rule")

    body += legend(296, 372, [(GOOD_SOFT, "base case — recursion stops"),
                              (GOLD_SOFT, "the call being evaluated")], 212)
    body += caption(350, 404,
                    f"F(4) unfolds into {len(nodes)} calls and {len(leaves)} base cases; "
                    f"each value returns up the tree until F(4) = {root.value}.")
    return svg(700, 422, body,
               "A recursion tree for Fibonacci of 4 branching downward into smaller calls "
               "until it reaches the base cases F(1) and F(0)")
