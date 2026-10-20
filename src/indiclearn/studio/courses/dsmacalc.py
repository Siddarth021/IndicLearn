"""Exact problem generators for Discrete Structures and Matrix Algebra (DSMA).

Same contract as oscalc.py: every numerical, NAT, numeric/choice MCQ and MSQ is generated together with its answer, and
the answer is COMPUTED (truth tables, counting formulas checked by brute force where small, exact rational matrix
arithmetic with sympy) - never written by a language model.

Blocks: {"p": text with $inline LaTeX$}, {"math": display LaTeX}, {"table": rows}.  Conventions are the ones stated in
courses/lessons.py DSMA_RULES (truth tables start from all-true rows, PDNF/PCNF row numbering with T = 1, ...).
"""

from __future__ import annotations

import itertools
import math
import random
from fractions import Fraction
from typing import Any, Callable

import sympy as sp

# =========================================================================== formatting


def fx(x: Any) -> str:
    """Exact number as LaTeX: 3, -\\frac{3}{2}, 2\\sqrt{5}."""
    x = sp.nsimplify(x) if not isinstance(x, (int, Fraction)) else sp.Rational(x.numerator, x.denominator) if isinstance(x, Fraction) else sp.Integer(x)
    if x.is_Rational and x.q != 1:
        return ("-" if x < 0 else "") + f"\\frac{{{abs(x.p)}}}{{{x.q}}}"
    return sp.latex(x)


def ft(x: Any) -> str:
    """Exact number as plain text: 3, -3/2, 2*sqrt(5)."""
    x = sp.nsimplify(x) if not isinstance(x, int) else sp.Integer(x)
    return str(x).replace("**", "^")


def plus(x: Any, sym: str = "") -> str:
    """'+ 3k' / '- 3k' for the next term of a sum (never '+ -3k')."""
    v = sp.nsimplify(x)
    mag = "" if sym and abs(v) == 1 else fx(abs(v))
    return f"{'-' if v < 0 else '+'} {mag}{sym}"


def mtex(M: Any) -> str:
    M = sp.Matrix(M)
    rows = [" & ".join(fx(M[i, j]) for j in range(M.cols)) for i in range(M.rows)]
    return "\\begin{bmatrix} " + " \\\\ ".join(rows) + " \\end{bmatrix}"


def vtex(v: Any) -> str:
    return mtex(sp.Matrix(v).reshape(len(v), 1))


def num(x: Any) -> float | int:
    """Value for a NAT answer box."""
    x = sp.nsimplify(x)
    if x.is_Integer:
        return int(x)
    return round(float(x), 2)


def _out(kind: str, topic: str, marks: int, blocks: list, answer: list, **data: Any) -> dict[str, Any]:
    return {"kind": "numerical", "topic": topic, "marks": marks, "blocks": blocks, "answer": answer, "data": data}


def _n(difficulty: str, easy: int, medium: int, hard: int) -> int:
    return {"easy": easy, "medium": medium, "hard": hard}.get(difficulty, medium)


# =========================================================================== propositional logic
# formula = ("var", "p") | ("not", f) | (op, f, g) with op in and, or, imp, iff

PREC = {"var": 5, "not": 4, "and": 3, "or": 2, "imp": 1, "iff": 0}
SYM = {"and": "\\wedge", "or": "\\vee", "imp": "\\rightarrow", "iff": "\\leftrightarrow"}
SYM_T = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}


def V(name: str) -> tuple:
    return ("var", name)


def NOT(f: tuple) -> tuple:
    return ("not", f)


def BIN(op: str, a: tuple, b: tuple) -> tuple:
    return (op, a, b)


def ev(f: tuple, env: dict[str, bool]) -> bool:
    op = f[0]
    if op == "var":
        return env[f[1]]
    if op == "not":
        return not ev(f[1], env)
    a, b = ev(f[1], env), ev(f[2], env)
    return {"and": a and b, "or": a or b, "imp": (not a) or b, "iff": a == b}[op]


def tex(f: tuple, parent: int = -1, right: bool = False) -> str:
    op = f[0]
    if op == "var":
        return f[1]
    if op == "not":
        return "\\neg " + tex(f[1], PREC["not"])
    p = PREC[op]
    s = f"{tex(f[1], p)} {SYM[op]} {tex(f[2], p, True)}"
    return f"({s})" if parent >= 0 else s  # every nested connective is bracketed: no precedence puzzles for students


def txt(f: tuple, parent: int = -1, right: bool = False) -> str:
    op = f[0]
    if op == "var":
        return f[1]
    if op == "not":
        return "¬" + txt(f[1], PREC["not"])
    p = PREC[op]
    s = f"{txt(f[1], p)} {SYM_T[op]} {txt(f[2], p, True)}"
    return f"({s})" if parent >= 0 else s


def vars_of(f: tuple) -> list[str]:
    out: set[str] = set()

    def walk(g: tuple) -> None:
        if g[0] == "var":
            out.add(g[1])
        else:
            for x in g[1:]:
                walk(x)
    walk(f)
    return sorted(out)


def rows_of(vs: list[str]) -> list[dict[str, bool]]:
    return [dict(zip(vs, vals)) for vals in itertools.product([True, False], repeat=len(vs))]


def classify(f: tuple, vs: list[str] | None = None) -> str:
    vals = [ev(f, r) for r in rows_of(vs or vars_of(f))]
    return "tautology" if all(vals) else "contradiction" if not any(vals) else "contingency"


def subformulas(f: tuple) -> list[tuple]:
    out: list[tuple] = []

    def walk(g: tuple) -> None:
        if g[0] == "var":
            return
        for x in g[1:]:
            walk(x)
        if g not in out:
            out.append(g)
    walk(f)
    return out


def truth_table(f: tuple, vs: list[str], extra: list[tuple] | None = None) -> list[list[str]]:
    cols = subformulas(f) if extra is None else extra
    head = [*vs, *("$" + tex(c) + "$" for c in cols)]
    body = [[*("T" if r[v] else "F" for v in vs), *("T" if ev(c, r) else "F" for c in cols)] for r in rows_of(vs)]
    return [head, *body]


def rand_formula(rng: random.Random, vs: list[str], depth: int) -> tuple:
    if depth == 0 or rng.random() < 0.15:
        v = V(rng.choice(vs))
        return NOT(v) if rng.random() < 0.3 else v
    if rng.random() < 0.15:
        return NOT(rand_formula(rng, vs, depth - 1))
    op = rng.choice(["and", "or", "imp", "imp", "iff"])
    return BIN(op, rand_formula(rng, vs, depth - 1), rand_formula(rng, vs, depth - 1))


def _trivial(f: tuple) -> bool:
    """A piece like q ∨ ¬q, p ∧ p or ¬¬p: it makes a formula look like a trick and teaches nothing."""
    for g in subformulas(f):
        if g[0] == "not" and g[1][0] == "not":
            return True
        if g[0] in SYM:
            a, b = g[1], g[2]
            if a == b or a == ("not", b) or b == ("not", a):
                return True
    return False


def good_formula(rng: random.Random, vs: list[str], depth: int, want: str | None = None) -> tuple:
    for _ in range(600):
        f = rand_formula(rng, vs, depth)
        if (vars_of(f) == sorted(vs) and len(subformulas(f)) >= len(vs) and not _trivial(f) and len(subformulas(f)) <= 6
                and (want is None or classify(f, vs) == want)):
            return f
    return KNOWN[want or "contingency"][0]


p_, q_, r_ = V("p"), V("q"), V("r")
KNOWN = {
    "tautology": [BIN("or", p_, NOT(p_)), BIN("imp", BIN("and", p_, BIN("imp", p_, q_)), q_),
                  BIN("imp", BIN("and", BIN("imp", p_, q_), BIN("imp", q_, r_)), BIN("imp", p_, r_)),
                  BIN("iff", BIN("imp", p_, q_), BIN("imp", NOT(q_), NOT(p_))), BIN("imp", BIN("and", NOT(q_), BIN("imp", p_, q_)), NOT(p_)),
                  BIN("imp", p_, BIN("or", p_, q_)), BIN("iff", NOT(BIN("and", p_, q_)), BIN("or", NOT(p_), NOT(q_))),
                  BIN("imp", BIN("and", BIN("or", p_, q_), NOT(p_)), q_)],
    "contradiction": [BIN("and", p_, NOT(p_)), BIN("and", BIN("and", p_, q_), NOT(q_)), NOT(BIN("imp", p_, BIN("or", p_, q_))),
                      BIN("and", BIN("imp", p_, q_), BIN("and", p_, NOT(q_))), BIN("iff", p_, NOT(p_))],
    "contingency": [BIN("imp", BIN("or", p_, q_), p_), BIN("imp", q_, BIN("and", p_, q_)), BIN("imp", BIN("imp", p_, q_), p_),
                    BIN("iff", BIN("imp", p_, q_), BIN("imp", q_, p_)), BIN("imp", BIN("and", BIN("or", p_, q_), q_), p_)],
}


def gen_truth(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    vs = ["p", "q"] if difficulty == "easy" else ["p", "q", "r"] if difficulty == "hard" or rng.random() < 0.5 else ["p", "q"]
    f = good_formula(rng, vs, 2 if difficulty == "easy" else 3)
    tt = truth_table(f, vs)
    c = classify(f, vs)
    n_true = sum(1 for r in rows_of(vs) if ev(f, r))
    blocks = [{"p": f"Construct the truth table of $F = {tex(f)}$. In how many rows is $F$ true? Is $F$ a tautology, a contradiction or a contingency?"}]
    answer = [{"table": tt}, {"p": f"$F$ is true in {n_true} of {2 ** len(vs)} rows, so it is a **{c}**."}]
    return _out("truth", topic, marks, blocks, answer, formula=txt(f), true_rows=n_true, cls=c)


def _taut_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    vs = ["p", "q", "r"]
    f = good_formula(rng, vs, 3)
    n_true = sum(1 for r in rows_of(vs) if ev(f, r))
    blocks = [{"p": f"The number of rows in the truth table of $F = {tex(f)}$ in which $F$ is TRUE is ______."}]
    traps = [8 - n_true, n_true + 1, n_true - 1]
    return blocks, n_true, [{"table": truth_table(f, vs)}, {"p": f"True in {n_true} rows."}], traps


def gen_taut(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    want = rng.choice(["tautology", "contradiction", "contingency"])
    f = rng.choice(KNOWN[want]) if rng.random() < 0.5 else good_formula(rng, ["p", "q"] if difficulty == "easy" else ["p", "q", "r"], 3, want)
    vs = vars_of(f)
    blocks = [{"p": f"Using a truth table, decide whether $F = {tex(f)}$ is a tautology, a contradiction or a contingency."}]
    answer = [{"table": truth_table(f, vs)}, {"p": f"The last column is {'all T' if want == 'tautology' else 'all F' if want == 'contradiction' else 'a mix of T and F'}, so $F$ is a **{classify(f, vs)}**."}]
    return _out("taut", topic, marks, blocks, answer, cls=classify(f, vs))


def _equiv_pool() -> list[tuple[tuple, list[tuple]]]:
    pq = BIN("imp", p_, q_)
    return [
        (pq, [BIN("or", NOT(p_), q_), BIN("imp", NOT(q_), NOT(p_)), NOT(BIN("and", p_, NOT(q_))), BIN("imp", q_, p_), BIN("imp", NOT(p_), NOT(q_)),
              BIN("or", p_, NOT(q_)), BIN("and", NOT(p_), q_)]),
        (NOT(BIN("imp", p_, q_)), [BIN("and", p_, NOT(q_)), BIN("imp", NOT(p_), q_), BIN("and", NOT(p_), q_), NOT(BIN("or", NOT(p_), q_)), BIN("imp", p_, NOT(q_))]),
        (BIN("iff", p_, q_), [BIN("and", BIN("imp", p_, q_), BIN("imp", q_, p_)), BIN("or", BIN("and", p_, q_), BIN("and", NOT(p_), NOT(q_))),
                              BIN("iff", NOT(p_), NOT(q_)), BIN("imp", p_, q_), BIN("iff", p_, NOT(q_)), BIN("and", p_, q_)]),
        (NOT(BIN("and", p_, q_)), [BIN("or", NOT(p_), NOT(q_)), BIN("imp", p_, NOT(q_)), BIN("and", NOT(p_), NOT(q_)), BIN("or", NOT(p_), q_)]),
        (BIN("imp", BIN("and", p_, q_), r_), [BIN("imp", p_, BIN("imp", q_, r_)), BIN("or", BIN("or", NOT(p_), NOT(q_)), r_),
                                              BIN("and", BIN("imp", p_, r_), BIN("imp", q_, r_)), BIN("or", BIN("imp", p_, r_), BIN("imp", q_, r_)),
                                              BIN("imp", BIN("imp", p_, q_), r_)]),
        (BIN("imp", BIN("or", p_, q_), r_), [BIN("and", BIN("imp", p_, r_), BIN("imp", q_, r_)), BIN("or", BIN("imp", p_, r_), BIN("imp", q_, r_)),
                                             BIN("or", BIN("and", NOT(p_), NOT(q_)), r_), BIN("imp", p_, BIN("imp", q_, r_))]),
    ]


def same(f: tuple, g: tuple) -> bool:
    vs = sorted(set(vars_of(f)) | set(vars_of(g)))
    return all(ev(f, r) == ev(g, r) for r in rows_of(vs))


def gen_equiv(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    base, cands = rng.choice(_equiv_pool())
    g = rng.choice(cands)
    vs = sorted(set(vars_of(base)) | set(vars_of(g)))
    eq = same(base, g)
    blocks = [{"p": f"Using a truth table, check whether $P = {tex(base)}$ and $Q = {tex(g)}$ are logically equivalent."}]
    tt = truth_table(base, vs, [base, g])
    diff = next((r for r in rows_of(vs) if ev(base, r) != ev(g, r)), None)
    concl = ("The two columns agree in every row, so $P \\equiv Q$." if eq else
             "They differ when " + ", ".join(f"${v}$ = {'T' if diff[v] else 'F'}" for v in vs) + ", so $P \\not\\equiv Q$.")
    return _out("equiv", topic, marks, blocks, [{"table": tt}, {"p": concl}], equivalent=eq)


def gen_conditional(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    p, q = rng.choice(COND_PAIRS)
    blocks = [{"p": f"Write the converse, inverse and contrapositive of: \"If {p}, then {q}.\" Which of them is logically equivalent to the original statement?"}]
    answer = [{"p": f"Converse ($q \\rightarrow p$): If {q}, then {p}."}, {"p": f"Inverse ($\\neg p \\rightarrow \\neg q$): If {neg(p)}, then {neg(q)}."},
              {"p": f"Contrapositive ($\\neg q \\rightarrow \\neg p$): If {neg(q)}, then {neg(p)}."},
              {"p": "Only the contrapositive is equivalent to the original (and the converse is equivalent to the inverse)."}]
    return _out("conditional", topic, marks, blocks, answer)


COND_PAIRS = [("it rains", "the ground is wet"), ("a number is divisible by 6", "it is divisible by 3"), ("the program compiles", "it has no syntax errors"),
              ("you score 90", "you get an A grade"), ("a triangle is equilateral", "it is isosceles"), ("the switch is on", "the bulb glows"),
              ("n is prime and greater than 2", "n is odd"), ("the matrix is invertible", "its determinant is nonzero")]


def neg(s: str) -> str:
    for a, b in ((" is ", " is not "), (" has ", " does not have "), (" are ", " are not "), (" compiles", " does not compile"), (" rains", " does not rain"),
                 (" score ", " do not score "), (" get ", " do not get "), (" glows", " does not glow")):
        if a in " " + s + " " or a in s:
            return s.replace(a.strip(), b.strip(), 1) if a.strip() and a.strip() in s else s
    return "it is not true that " + s


def _cond_choice(rng: random.Random, difficulty: str) -> tuple[list, list[str], int, list]:
    p, q = rng.choice(COND_PAIRS)
    which = rng.choice(["contrapositive", "converse", "inverse"])
    opts = {"converse": f"If {q}, then {p}.", "inverse": f"If {neg(p)}, then {neg(q)}.", "contrapositive": f"If {neg(q)}, then {neg(p)}.",
            "wrong": f"If {neg(p)}, then {q}."}
    order = ["converse", "inverse", "contrapositive", "wrong"]
    rng.shuffle(order)
    blocks = [{"p": f"Statement: \"If {p}, then {q}.\" Which of the following is its {which}?"}]
    return blocks, [opts[k] for k in order], order.index(which), [{"p": f"The {which} is: {opts[which]}"}]


def _equiv_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    base, cands = rng.choice(_equiv_pool())
    picks = rng.sample(cands, min(4, len(cands)))
    good = [i for i, g in enumerate(picks) if same(base, g)]
    if not 1 <= len(good) <= 3 or len(picks) < 4:
        return None
    blocks = [{"p": f"Which of the following are logically equivalent to $ {tex(base)} $?"}]
    ans = [{"p": "; ".join(f"${tex(g)}$: {'equivalent' if same(base, g) else 'not equivalent'}" for g in picks) + " (checked row by row on the truth table)."}]
    return blocks, [f"${tex(g)}$" for g in picks], good, ans


def _taut_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    k = rng.choice([1, 2, 3])
    picks = rng.sample(KNOWN["tautology"], k) + rng.sample(KNOWN["contingency"] + KNOWN["contradiction"], 4 - k)
    rng.shuffle(picks)
    good = [i for i, f in enumerate(picks) if classify(f) == "tautology"]
    blocks = [{"p": "Which of the following formulas are tautologies?"}]
    return blocks, [f"${tex(f)}$" for f in picks], good, [{"p": "; ".join(f"${tex(f)}$: {classify(f)}" for f in picks) + "."}]


# ---- predicates and quantifiers over a finite domain
PREDS: list[tuple[str, str, Callable[[int, int], bool]]] = [
    ("x + y \\le {k}", "x + y ≤ {k}", lambda x, y, k: x + y <= k), ("x < y", "x < y", lambda x, y, k: x < y),
    ("x \\le y", "x ≤ y", lambda x, y, k: x <= y), ("x \\cdot y \\text{ is even}", "x·y is even", lambda x, y, k: (x * y) % 2 == 0),
    ("x \\mid y", "x divides y", lambda x, y, k: y % x == 0), ("x + y = {k}", "x + y = {k}", lambda x, y, k: x + y == k),
    ("x \\ne y", "x ≠ y", lambda x, y, k: x != y),
]
QFORMS = [("\\forall x\\, \\forall y", "AA"), ("\\forall x\\, \\exists y", "AE"), ("\\exists x\\, \\forall y", "EA"), ("\\exists x\\, \\exists y", "EE"),
          ("\\exists y\\, \\forall x", "E'A")]


def _quant_truth(form: str, dom: list[int], P: Callable[[int, int], bool]) -> tuple[bool, str]:
    if form == "AA":
        bad = next(((x, y) for x in dom for y in dom if not P(x, y)), None)
        return bad is None, (f"fails at $x={bad[0]}, y={bad[1]}$" if bad else "holds for every pair")
    if form == "EE":
        good = next(((x, y) for x in dom for y in dom if P(x, y)), None)
        return good is not None, (f"witness $x={good[0]}, y={good[1]}$" if good else "no pair works")
    if form == "AE":
        bad = next((x for x in dom if not any(P(x, y) for y in dom)), None)
        return bad is None, (f"no $y$ works for $x={bad}$" if bad is not None else "each $x$ has a $y$: " + ", ".join(f"$x={x} \\to y={next(y for y in dom if P(x, y))}$" for x in dom))
    if form == "EA":
        good = next((x for x in dom if all(P(x, y) for y in dom)), None)
        return good is not None, (f"$x={good}$ works for every $y$" if good is not None else "no single $x$ works for all $y$")
    good = next((y for y in dom if all(P(x, y) for x in dom)), None)  # E'A: exists y for all x
    return good is not None, (f"$y={good}$ works for every $x$" if good is not None else "no single $y$ works for all $x$")


def _quant_setup(rng: random.Random, difficulty: str) -> tuple[list[int], str, str, Callable[[int, int], bool], int]:
    lo = rng.choice([1, 1, 0])
    dom = list(range(lo, lo + _n(difficulty, 3, 4, 4)))
    pt, pp, f = rng.choice(PREDS)
    k = rng.randint(min(dom) * 2 + 1, max(dom) * 2 - 1)
    if pt.startswith("x \\mid") and 0 in dom:
        dom = [d for d in dom if d] + [max(dom) + 1]
    return dom, pt.replace("{k}", str(k)), pp.replace("{k}", str(k)), (lambda x, y, _f=f, _k=k: _f(x, y, _k)), k


def gen_quant(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    dom, pt, _pp, P, _k = _quant_setup(rng, difficulty)
    forms = rng.sample(QFORMS[:4], 3) if marks < 4 else QFORMS[:4]
    blocks = [{"p": f"The domain of $x$ and $y$ is $\\{{{', '.join(map(str, dom))}\\}}$ and $P(x, y)$ is \"${pt}$\". Find the truth value of each statement, with a reason: "
                    + "; ".join(f"$ {qt}\\; P(x,y)$" for qt, _c in forms) + "."}]
    ans = []
    for qt, code in forms:
        val, why = _quant_truth(code, dom, P)
        ans.append({"p": f"$ {qt}\\; P(x,y)$ is **{'TRUE' if val else 'FALSE'}** - {why}."})
    return _out("quant", topic, marks, blocks, ans)


def _quant_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    dom, pt, _pp, P, _k = _quant_setup(rng, difficulty)
    forms = rng.sample(QFORMS, 4)
    vals = [_quant_truth(c, dom, P) for _q, c in forms]
    good = [i for i, (v, _w) in enumerate(vals) if v]
    if not 1 <= len(good) <= 3:
        return None
    blocks = [{"p": f"Domain $\\{{{', '.join(map(str, dom))}\\}}$ for both $x$ and $y$; $P(x,y)$: ${pt}$. Which of the following statements are TRUE?"}]
    return blocks, [f"$ {q}\\; P(x,y)$" for q, _c in forms], good, [{"p": "; ".join(f"$ {q}\\, P$: {'T' if v else 'F'} ({w})" for (q, _c), (v, w) in zip(forms, vals)) + "."}]


# ---- arguments
ARGS = [  # (name, premises, conclusion)
    ("modus ponens", [p_, BIN("imp", p_, q_)], q_), ("modus tollens", [NOT(q_), BIN("imp", p_, q_)], NOT(p_)),
    ("hypothetical syllogism", [BIN("imp", p_, q_), BIN("imp", q_, r_)], BIN("imp", p_, r_)), ("disjunctive syllogism", [BIN("or", p_, q_), NOT(p_)], q_),
    ("resolution", [BIN("or", p_, q_), BIN("or", NOT(p_), r_)], BIN("or", q_, r_)),
    ("affirming the conclusion (fallacy)", [q_, BIN("imp", p_, q_)], p_), ("denying the hypothesis (fallacy)", [NOT(p_), BIN("imp", p_, q_)], NOT(q_)),
    ("an invalid chain", [BIN("imp", p_, q_), BIN("imp", r_, q_)], BIN("imp", p_, r_)), ("an invalid disjunction step", [BIN("or", p_, q_), p_], NOT(q_)),
    ("constructive dilemma", [BIN("imp", p_, q_), BIN("imp", r_, q_), BIN("or", p_, r_)], q_),
]


def valid(prem: list[tuple], concl: tuple) -> tuple[bool, dict[str, bool] | None]:
    vs = sorted(set().union(*(set(vars_of(x)) for x in prem + [concl])))
    for r in rows_of(vs):
        if all(ev(x, r) for x in prem) and not ev(concl, r):
            return False, r
    return True, None


def gen_argument(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    name, prem, concl = rng.choice(ARGS)
    ok, bad = valid(prem, concl)
    vs = sorted(set().union(*(set(vars_of(x)) for x in prem + [concl])))
    blocks = [{"p": "Premises: " + ", ".join(f"${tex(x)}$" for x in prem) + f". Conclusion: ${tex(concl)}$. Is the argument valid? Justify with a truth table."}]
    tt = truth_table(concl, vs, prem + [concl])
    concl_t = ("In every row where all premises are T the conclusion is T, so the argument is **valid**" + (f" ({name})." if "fallacy" not in name else ".") if ok else
               "Invalid: when " + ", ".join(f"${v}$ = {'T' if bad[v] else 'F'}" for v in vs) + f" all premises are T but the conclusion is F ({name}).")
    return _out("argument", topic, marks, blocks, [{"table": tt}, {"p": concl_t}], valid=ok)


def _arg_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    picks = rng.sample(ARGS, 4)
    good = [i for i, (_n, pr, c) in enumerate(picks) if valid(pr, c)[0]]
    if not 1 <= len(good) <= 3:
        return None
    opts = [", ".join(f"${tex(x)}$" for x in pr) + f" $\\;\\therefore\\; {tex(c)}$" for _n, pr, c in picks]
    return [{"p": "Which of the following arguments are valid?"}], opts, good, [{"p": "; ".join(f"{n}: {'valid' if valid(pr, c)[0] else 'invalid'}" for n, pr, c in picks) + "."}]


# ---- normal forms
def gen_nf(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    vs = ["p", "q", "r"] if difficulty != "easy" else ["p", "q"]
    f = good_formula(rng, vs, 2 if difficulty == "easy" else 3, "contingency")
    rows = rows_of(vs)
    n = len(vs)
    idx = lambda r: int("".join("1" if r[v] else "0" for v in vs), 2)  # noqa: E731
    mins = sorted(idx(r) for r in rows if ev(f, r))
    maxs = sorted(idx(r) for r in rows if not ev(f, r))
    minterm = lambda r: " \\wedge ".join(v if r[v] else "\\neg " + v for v in vs)  # noqa: E731
    maxterm = lambda r: " \\vee ".join("\\neg " + v if r[v] else v for v in vs)  # noqa: E731
    pdnf = " \\vee ".join(f"({minterm(r)})" for r in sorted(rows, key=idx) if ev(f, r))
    pcnf = " \\wedge ".join(f"({maxterm(r)})" for r in sorted(rows, key=idx) if not ev(f, r))
    blocks = [{"p": f"Find the principal disjunctive normal form (PDNF) and the principal conjunctive normal form (PCNF) of $F = {tex(f)}$."}]
    tt = truth_table(f, vs, [f])
    tt[0].insert(0, "row")
    for i, r in enumerate(rows, 1):
        tt[i].insert(0, str(idx(r)))
    answer = [{"table": tt}, {"p": f"$F$ is T in rows {', '.join(map(str, mins))}: PDNF $= \\Sigma m({', '.join(map(str, mins))})$"}, {"math": pdnf},
              {"p": f"$F$ is F in rows {', '.join(map(str, maxs))}: PCNF $= \\Pi M({', '.join(map(str, maxs))})$"}, {"math": pcnf}]
    return _out("nf", topic, marks, blocks, answer, minterms=mins, maxterms=maxs, nvars=n)


def _nf_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    vs = ["p", "q", "r"]
    f = good_formula(rng, vs, 3, "contingency")
    mins = [r for r in rows_of(vs) if ev(f, r)]
    blocks = [{"p": f"The number of minterms in the principal disjunctive normal form of $ {tex(f)} $ is ______."}]
    return blocks, len(mins), [{"table": truth_table(f, vs, [f])}, {"p": f"One minterm per T row: {len(mins)}."}], [8 - len(mins), len(mins) + 1, 3]


# =========================================================================== sets and functions


def sfmt(s: set | list) -> str:
    items = sorted(s, key=lambda x: (str(type(x)), x))
    return "\\{" + ", ".join(map(str, items)) + "\\}" if items else "\\emptyset"


def gen_sets(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    U = set(range(1, 11))
    A = set(rng.sample(sorted(U), rng.randint(4, 6)))
    B = set(rng.sample(sorted(U), rng.randint(3, 5)))
    blocks = [{"p": f"Let $U = {sfmt(U)}$, $A = {sfmt(A)}$ and $B = {sfmt(B)}$. Find $A \\cup B$, $A \\cap B$, $A - B$, $A \\oplus B$, $\\overline{{A \\cup B}}$, $|P(A \\cap B)|$ and $|A \\times B|$."}]
    ans = [{"p": f"$A \\cup B = {sfmt(A | B)}$, $A \\cap B = {sfmt(A & B)}$, $A - B = {sfmt(A - B)}$, $A \\oplus B = {sfmt(A ^ B)}$."},
           {"p": f"$\\overline{{A \\cup B}} = {sfmt(U - (A | B))}$; $|P(A \\cap B)| = 2^{{{len(A & B)}}} = {2 ** len(A & B)}$; $|A \\times B| = {len(A)} \\cdot {len(B)} = {len(A) * len(B)}$."},
           {"p": f"Check: $|A \\cup B| = |A| + |B| - |A \\cap B| = {len(A)} + {len(B)} - {len(A & B)} = {len(A | B)}$."}]
    return _out("sets", topic, marks, blocks, ans)


def _sets_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    n = rng.choice([50, 60, 80, 100, 120])
    a, b = rng.randint(n // 4, n // 2), rng.randint(n // 4, n // 2)
    both = rng.randint(max(1, a + b - n + 5), min(a, b) - 2)
    langs = rng.choice([("Python", "Java"), ("cricket", "football"), ("Telugu", "Hindi"), ("OS", "DBMS")])
    neither = n - (a + b - both)
    blocks = [{"p": f"In a class of {n} students, {a} like {langs[0]}, {b} like {langs[1]} and {both} like both. The number of students who like neither is ______."}]
    return blocks, neither, [{"p": f"$|A \\cup B| = {a} + {b} - {both} = {a + b - both}$; neither $= {n} - {a + b - both} = {neither}$."}], [n - a - b, n - (a + b), a + b - both, n - both]


def surjections(m: int, n: int) -> int:
    return sum((-1) ** i * math.comb(n, i) * (n - i) ** m for i in range(n + 1))


def gen_func(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    m, n = rng.choice([(3, 4), (4, 3), (4, 4), (5, 3), (3, 5), (4, 2), (5, 4)])
    inj = math.perm(n, m) if m <= n else 0
    sur = surjections(m, n)
    blocks = [{"p": f"Let $|A| = {m}$ and $|B| = {n}$. How many functions $f: A \\to B$ are there? How many are one-to-one? How many are onto? How many are bijections?"}]
    ans = [{"p": f"All functions: each of the {m} elements has {n} choices: ${n}^{{{m}}} = {n ** m}$."},
           {"p": (f"One-to-one: $P({n},{m}) = {n}!/{n - m}! = {inj}$." if m <= n else f"One-to-one: none, since $|A| > |B|$ (pigeonhole).")},
           {"p": "Onto: $\\sum_{i=0}^{" + str(n) + "} (-1)^i \\binom{" + str(n) + "}{i} (" + str(n) + "-i)^{" + str(m) + "} = "
                 + " ".join(f"{'-' if i % 2 else '+'} {math.comb(n, i)} \\cdot {n - i}^{{{m}}}" for i in range(n)).lstrip("+ ") + f" = {sur}$."},
           {"p": f"Bijections: {math.factorial(n) if m == n else 0}" + (" ($n!$, since $|A| = |B|$)." if m == n else " (sizes differ).")}]
    return _out("func", topic, marks, blocks, ans, functions=n ** m, injective=inj, onto=sur)


def _func_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    m, n = rng.choice([(4, 3), (5, 3), (5, 2), (4, 2), (6, 3), (5, 4)])
    sur = surjections(m, n)
    blocks = [{"p": f"The number of onto functions from a set with {m} elements to a set with {n} elements is ______."}]
    return blocks, sur, [{"p": f"$\\sum_{{i=0}}^{{{n}}} (-1)^i \\binom{{{n}}}{{i}} ({n}-i)^{{{m}}} = {sur}$."}], [n ** m, m ** n, math.perm(max(m, n), min(m, n)), n ** m - n]


def gen_sum(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    kind = rng.choice(["lin", "sq", "geo"])
    lo, hi = rng.choice([1, 1, 2, 5]), rng.choice([10, 12, 15, 20, 25])
    if kind == "lin":
        a, b = rng.randint(2, 5), rng.randint(-3, 4)
        val = sum(a * k + b for k in range(lo, hi + 1))
        term = f"({a}k {'+' if b >= 0 else '-'} {abs(b)})"
        steps = f"$= {a}\\sum k {plus(b)}\\cdot({hi}-{lo}+1) = {a}\\cdot{sum(range(lo, hi + 1))} {plus(b * (hi - lo + 1))} = {val}$"
    elif kind == "sq":
        val = sum(k * k for k in range(lo, hi + 1))
        term = "k^2"
        s = lambda n: n * (n + 1) * (2 * n + 1) // 6  # noqa: E731
        steps = f"$= \\frac{{{hi}\\cdot{hi + 1}\\cdot{2 * hi + 1}}}{{6}} - \\frac{{{lo - 1}\\cdot{lo}\\cdot{2 * lo - 1}}}{{6}} = {s(hi)} - {s(lo - 1)} = {val}$"
    else:
        a, r = rng.randint(1, 5), rng.choice([2, 3])
        lo, hi = 0, rng.randint(5, 9)
        val = sum(a * r ** k for k in range(lo, hi + 1))
        term = f"{a}\\cdot {r}^k"
        steps = f"$= {a}\\cdot\\frac{{{r}^{{{hi + 1}}} - 1}}{{{r} - 1}} = {val}$"
    blocks = [{"p": f"Evaluate $\\displaystyle\\sum_{{k={lo}}}^{{{hi}}} {term}$."}]
    return _out("sum", topic, marks, blocks, [{"p": steps}], value=val)


def _sum_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_sum(rng, 2, difficulty, "sums")
    v = q["data"]["value"]
    return [{"p": q["blocks"][0]["p"].replace("Evaluate", "The value of").rstrip(".") + " is ______."}], v, q["answer"], [v - 1, v + 1, int(v * 0.9)]


RECS = [  # name, first values, rule text, step function
    ("f(0) = {a}, f(n) = 2f(n-1) + {b}", lambda a, b: [a], lambda seq, n, a, b: 2 * seq[-1] + b),
    ("f(0) = {a}, f(n) = 3f(n-1) - {b}", lambda a, b: [a], lambda seq, n, a, b: 3 * seq[-1] - b),
    ("f(0) = {a}, f(1) = {b}, f(n) = f(n-1) + f(n-2)", lambda a, b: [a, b], lambda seq, n, a, b: seq[-1] + seq[-2]),
    ("f(0) = {a}, f(1) = {b}, f(n) = f(n-1) + 2f(n-2)", lambda a, b: [a, b], lambda seq, n, a, b: seq[-1] + 2 * seq[-2]),
    ("f(1) = {a}, f(n) = f(n-1) + n^2", lambda a, b: [None, a], lambda seq, n, a, b: seq[-1] + n * n),
    ("H(1) = 1, H(n) = 2H(n-1) + 1 (Tower of Hanoi moves)", lambda a, b: [None, 1], lambda seq, n, a, b: 2 * seq[-1] + 1),
]


def _rec_run(rng: random.Random, difficulty: str) -> tuple[str, list, int, int]:
    name, init, step = rng.choice(RECS)
    a, b = rng.randint(1, 4), rng.randint(1, 5)
    seq = list(init(a, b))
    n_target = _n(difficulty, 5, 6, 8)
    while len(seq) <= n_target:
        seq.append(step(seq, len(seq), a, b))
    return name.format(a=a, b=b), seq, n_target, seq[n_target]


def gen_receval(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    rule, seq, n, val = _rec_run(rng, difficulty)
    fn = "H" if rule.startswith("H") else "f"
    blocks = [{"p": f"A function is defined recursively by ${rule.split(' (')[0]}$. Find ${fn}({n})$, showing every term."}]
    terms = ", ".join(f"${fn}({i}) = {v}$" for i, v in enumerate(seq) if v is not None)
    return _out("receval", topic, marks, blocks, [{"p": terms}, {"p": f"So ${fn}({n}) = {val}$."}], value=val)


def _receval_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    rule, seq, n, val = _rec_run(rng, difficulty)
    fn = "H" if rule.startswith("H") else "f"
    blocks = [{"p": f"For ${rule.split(' (')[0]}$, the value of ${fn}({n})$ is ______."}]
    return blocks, val, [{"p": ", ".join(f"${fn}({i}) = {v}$" for i, v in enumerate(seq) if v is not None)}], [seq[n - 1], seq[n + 1] if n + 1 < len(seq) else val * 2, val + 1]


# =========================================================================== counting


def _count_q(rng: random.Random, difficulty: str) -> tuple[str, int, str, list]:
    t = rng.randrange(6)
    if t == 0:
        n = rng.choice([6, 7, 8, 10])
        v = 2 ** (n - 1) + 2 ** (n - 2) - 2 ** (n - 3)
        return (f"How many bit strings of length {n} start with 1 or end with 00?", v,
                f"Start with 1: $2^{{{n - 1}}}$; end with 00: $2^{{{n - 2}}}$; both: $2^{{{n - 3}}}$. Total $= {2 ** (n - 1)} + {2 ** (n - 2)} - {2 ** (n - 3)} = {v}$.",
                [2 ** (n - 1) + 2 ** (n - 2), 2 ** n, 2 ** (n - 3)])
    if t == 1:
        L, D = rng.choice([2, 3]), rng.choice([3, 4])
        v = 26 ** L * 10 ** D
        return (f"A vehicle code has {L} capital letters followed by {D} digits (repetition allowed). How many codes are possible?", v,
                f"Product rule: $26^{{{L}}} \\cdot 10^{{{D}}} = {v}$.", [26 * L * 10 * D, 26 ** L + 10 ** D, math.perm(26, L) * math.perm(10, D)])
    if t == 2:
        n = rng.choice([4, 5, 6])
        v = sum(36 ** k - 26 ** k for k in range(n, n + 2))
        return (f"A password has {n} or {n + 1} characters, each an uppercase letter or a digit, and must contain at least one digit. How many passwords are possible?", v,
                f"For length $k$: $36^k - 26^k$ (all minus those with no digit). Total $= (36^{{{n}}} - 26^{{{n}}}) + (36^{{{n + 1}}} - 26^{{{n + 1}}}) = {v}$.",
                [36 ** n + 36 ** (n + 1), 10 * 36 ** (n - 1) + 10 * 36 ** n])
    if t == 3:
        n = rng.choice([5, 6, 7])
        v = 2 ** n - 2 ** (n - 2)
        return (f"How many bit strings of length {n} do NOT start with 11?", v,
                f"All: $2^{{{n}}}$; starting with 11: $2^{{{n - 2}}}$; so $2^{{{n}}} - 2^{{{n - 2}}} = {v}$.", [2 ** (n - 2), 2 ** n - 2, 2 ** (n - 1)])
    if t == 4:
        lo, hi, d = rng.choice([(100, 999, 7), (100, 999, 9), (1000, 9999, 11), (50, 500, 6)])
        v = hi // d - (lo - 1) // d
        return (f"How many integers from {lo} to {hi} (inclusive) are divisible by {d}?", v,
                f"$\\lfloor {hi}/{d} \\rfloor - \\lfloor {lo - 1}/{d} \\rfloor = {hi // d} - {(lo - 1) // d} = {v}$.", [(hi - lo) // d, hi // d, v + 1])
    k = rng.choice([3, 4])
    fac = [9, 9, 8, 7][:k]
    v = math.prod(fac)
    prod = " \\cdot ".join(map(str, fac))
    return (f"How many {k}-digit numbers have all digits distinct?", v,
            f"First digit: 9 choices (not 0); then {', '.join(map(str, fac[1:]))} choices: ${prod} = {v}$.", [math.perm(10, k), 9 * 10 ** (k - 1), v + 9])


def gen_count(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    q, v, sol, _t = _count_q(rng, difficulty)
    return _out("count", topic, marks, [{"p": q}], [{"p": sol}], value=v)


def _count_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q, v, sol, traps = _count_q(rng, difficulty)
    return [{"p": q.replace("How many", "The number of").rstrip("?") + " is ______."}], v, [{"p": sol}], traps


def _ie_q(rng: random.Random, difficulty: str) -> tuple[str, int, str, list]:
    t = rng.randrange(3)
    if t == 0:
        N = rng.choice([100, 200, 250, 300, 500, 1000])
        a, b, c = rng.choice([(2, 3, 5), (3, 5, 7), (2, 5, 7), (4, 6, 9), (2, 3, 7)])
        L = math.lcm
        u = N // a + N // b + N // c - N // L(a, b) - N // L(a, c) - N // L(b, c) + N // L(a, b, c)
        notdiv = rng.random() < 0.5
        v = N - u if notdiv else u
        sol = (f"$\\lfloor {N}/{a} \\rfloor + \\lfloor {N}/{b} \\rfloor + \\lfloor {N}/{c} \\rfloor - \\lfloor {N}/{L(a, b)} \\rfloor - \\lfloor {N}/{L(a, c)} \\rfloor - \\lfloor {N}/{L(b, c)} \\rfloor + \\lfloor {N}/{L(a, b, c)} \\rfloor$"
               f" $= {N // a} + {N // b} + {N // c} - {N // L(a, b)} - {N // L(a, c)} - {N // L(b, c)} + {N // L(a, b, c)} = {u}$" + (f"; not divisible by any: ${N} - {u} = {v}$." if notdiv else "."))
        return (f"How many integers from 1 to {N} are {'divisible by none of' if notdiv else 'divisible by at least one of'} {a}, {b} or {c}?", v, sol,
                [N - v, N // a + N // b + N // c, v + N // L(a, b, c)])
    if t == 1:
        n = rng.choice([4, 5, 6])
        d = round(math.factorial(n) * sum((-1) ** i / math.factorial(i) for i in range(n + 1)))
        return (f"In how many ways can {n} letters be put into their {n} addressed envelopes so that NO letter goes into its own envelope?", d,
                f"Derangements: $D_{{{n}}} = {n}!\\left(1 - \\frac{{1}}{{1!}} + \\frac{{1}}{{2!}} - \\cdots\\right) = {d}$.", [math.factorial(n) - 1, math.factorial(n - 1), d + 1])
    m, n = rng.choice([(5, 3), (6, 3), (4, 3), (5, 2)])
    s = surjections(m, n)
    return (f"In how many ways can {m} different jobs be given to {n} employees so that every employee gets at least one job?", s,
            f"Onto functions from {m} jobs to {n} employees: $\\sum_{{i=0}}^{{{n}}}(-1)^i\\binom{{{n}}}{{i}}({n}-i)^{{{m}}} = {s}$.", [n ** m, math.comb(m - 1, n - 1), s // math.factorial(n)])


def gen_inclexcl(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    q, v, sol, _t = _ie_q(rng, difficulty)
    return _out("inclexcl", topic, marks, [{"p": q}], [{"p": sol}], value=v)


def _inclexcl_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q, v, sol, traps = _ie_q(rng, difficulty)
    return [{"p": q.replace("How many", "The number of").replace("In how many ways", "The number of ways").rstrip("?") + " is ______."}], v, [{"p": sol}], traps


def _pigeon_q(rng: random.Random, difficulty: str) -> tuple[str, int, str, list]:
    t = rng.randrange(3)
    if t == 0:
        r = rng.choice([3, 4, 5, 6])
        v = 12 * (r - 1) + 1
        return (f"What is the minimum number of students needed to guarantee that at least {r} of them were born in the same month?", v,
                f"12 months are the boxes; $12({r}-1) + 1 = {v}$.", [12 * r, 12 * (r - 1), v + 1])
    if t == 1:
        r = rng.choice([3, 4, 5])
        v = 4 * (r - 1) + 1
        return (f"How many cards must be drawn from a standard deck to guarantee that at least {r} cards of the same suit are chosen?", v,
                f"4 suits are the boxes; $4({r}-1) + 1 = {v}$.", [4 * r, 4 * (r - 1), 13 * (r - 1) + 1])
    N, k = rng.choice([(100, 7), (50, 6), (200, 9), (85, 4), (30, 7)])
    v = -(-N // k)
    return (f"{N} letters are put into {k} mailboxes. What is the least number of letters that some mailbox must contain?", v,
            f"$\\lceil {N}/{k} \\rceil = {v}$.", [N // k, v + 1, N // k - 1])


def gen_pigeon(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    q, v, sol, _t = _pigeon_q(rng, difficulty)
    return _out("pigeon", topic, marks, [{"p": q}], [{"p": sol}], value=v)


def _pigeon_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q, v, sol, traps = _pigeon_q(rng, difficulty)
    return [{"p": q.replace("What is the", "The").replace("How many cards must", "The number of cards that must").rstrip("?") + " is ______."}], v, [{"p": sol}], traps


WORDS = ["MISSISSIPPI", "BANANA", "ENGINEERING", "COMMITTEE", "STATISTICS", "SUCCESS", "BALLOON", "LETTER", "PEPPER", "ASSESS"]


def _pc_q(rng: random.Random, difficulty: str) -> tuple[str, int, str, list]:
    t = rng.randrange(4)
    if t == 0:
        w = rng.choice(WORDS)
        cnt = {c: w.count(c) for c in sorted(set(w))}
        v = math.factorial(len(w))
        for c in cnt.values():
            v //= math.factorial(c)
        rep = " \\cdot ".join(f"{c}!" for c in cnt.values() if c > 1)
        return (f"How many different arrangements of the letters of the word {w} are there?", v,
                f"${len(w)}! / ({rep}) = {v}$ (" + ", ".join(f"{k}: {c}" for k, c in cnt.items() if c > 1) + ").", [math.factorial(len(w)), v * 2, math.perm(len(w), len(cnt))])
    if t == 1:
        men, women, k = rng.choice([(6, 4, 4), (7, 5, 5), (5, 5, 4), (8, 4, 5)])
        tot = math.comb(men + women, k)
        none = math.comb(men, k)
        v = tot - none
        return (f"A committee of {k} is chosen from {men} men and {women} women. In how many ways can this be done if it must include at least one woman?", v,
                f"All committees $\\binom{{{men + women}}}{{{k}}} = {tot}$ minus all-men $\\binom{{{men}}}{{{k}}} = {none}$: ${v}$.",
                [tot, women * math.comb(men + women - 1, k - 1), math.comb(women, 1) * math.comb(men, k - 1)])
    if t == 2:
        n, k = rng.choice([(10, 3), (12, 4), (15, 3), (9, 3), (8, 4)])
        lo = rng.choice([0, 1])
        nn = n - lo * k
        v = math.comb(nn + k - 1, k - 1)
        cond = "nonnegative" if lo == 0 else "positive"
        sol = (f"Stars and bars: $\\binom{{{n} + {k} - 1}}{{{k} - 1}} = \\binom{{{n + k - 1}}}{{{k - 1}}} = {v}$." if lo == 0 else
               f"Put $x_i = y_i + 1$: $y_1 + \\cdots + y_{k} = {nn}$ with $y_i \\ge 0$, so $\\binom{{{nn + k - 1}}}{{{k - 1}}} = {v}$.")
        xs = " + ".join(f"x_{i}" for i in range(1, k + 1))
        return (f"How many {cond} integer solutions does $ {xs} = {n} $ have?", v, sol, [math.comb(n + k - 1, k), math.comb(n - 1, k - 1) if lo == 0 else math.comb(n + k - 1, k - 1), math.comb(n, k)])
    n, r = rng.choice([(8, 3), (10, 4), (7, 3), (9, 2), (6, 4)])
    v = math.perm(n, r)
    return (f"In how many ways can a president, a secretary{' and a treasurer' if r == 3 else ''}{' ' if r == 3 else ''}be chosen from {n} people (one person, one post)?" if r <= 3 else
            f"In how many ways can {r} of {n} runners finish first to {r}th (no ties)?", v, f"Order matters: $P({n},{r}) = {n}!/{n - r}! = {v}$.", [math.comb(n, r), n ** r, v // 2])


def gen_permcomb(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    q, v, sol, _t = _pc_q(rng, difficulty)
    return _out("permcomb", topic, marks, [{"p": q}], [{"p": sol}], value=v)


def _permcomb_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q, v, sol, traps = _pc_q(rng, difficulty)
    stem = q.replace("How many different arrangements", "The number of different arrangements").replace("In how many ways can", "The number of ways in which").replace("How many", "The number of")
    return [{"p": stem.rstrip("?") + " is ______."}], v, [{"p": sol}], traps


def _binom_q(rng: random.Random, difficulty: str) -> tuple[str, int, str, list]:
    n = rng.choice([5, 6, 7, 8, 9, 10])
    a, b = rng.choice([1, 2, 3]), rng.choice([-1, 1, 2, -2, 3, -3])
    k = rng.randint(2, n - 2)
    v = math.comb(n, k) * a ** (n - k) * b ** k
    ys = f"{'+' if b > 0 else '-'} {abs(b) if abs(b) != 1 else ''}y"
    xs = f"{a if a != 1 else ''}x"
    return (f"What is the coefficient of $x^{{{n - k}}} y^{{{k}}}$ in the expansion of $({xs} {ys})^{{{n}}}$?", v,
            f"$\\binom{{{n}}}{{{k}}} ({a})^{{{n - k}}} ({b})^{{{k}}} = {math.comb(n, k)} \\cdot {a ** (n - k)} \\cdot {b ** k} = {v}$.",
            [math.comb(n, k), math.comb(n, k) * a ** k * b ** (n - k), abs(v) if v < 0 else -v])


def gen_binom(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    q, v, sol, _t = _binom_q(rng, difficulty)
    return _out("binom", topic, marks, [{"p": q}], [{"p": sol}], value=v)


def _binom_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q, v, sol, traps = _binom_q(rng, difficulty)
    return [{"p": q.replace("What is the", "The").rstrip("?") + " is ______."}], v, [{"p": sol}], traps


# ---- recurrences
def _recur_setup(rng: random.Random, difficulty: str) -> dict[str, Any]:
    while True:
        r1 = rng.choice([-3, -2, -1, 2, 3, 4, 5])
        r2 = r1 if rng.random() < 0.3 else rng.choice([-3, -2, -1, 1, 2, 3, 4])
        c1, c2 = r1 + r2, -r1 * r2
        if c2 == 0 or (r1 == r2 and r1 in (1, -1)):
            continue
        a0, a1 = rng.randint(0, 4), rng.randint(1, 9)
        return {"r1": r1, "r2": r2, "c1": c1, "c2": c2, "a0": a0, "a1": a1}


def _recur_solve(d: dict[str, Any]) -> tuple[sp.Expr, str, list[str]]:
    n = sp.Symbol("n", integer=True, nonnegative=True)
    r1, r2, a0, a1 = d["r1"], d["r2"], d["a0"], d["a1"]
    A, B = sp.symbols("alpha_1 alpha_2")
    if r1 != r2:
        gen = A * sp.Integer(r1) ** n + B * sp.Integer(r2) ** n
        eqs = [sp.Eq(A + B, a0), sp.Eq(A * r1 + B * r2, a1)]
        gen_t = f"a_n = \\alpha_1 ({r1})^n + \\alpha_2 ({r2})^n"
    else:
        gen = (A + B * n) * sp.Integer(r1) ** n
        eqs = [sp.Eq(A, a0), sp.Eq((A + B) * r1, a1)]
        gen_t = f"a_n = (\\alpha_1 + \\alpha_2 n)({r1})^n"
    sol = sp.solve(eqs, [A, B])
    closed = sp.simplify(gen.subs(sol))
    steps = [gen_t, f"\\alpha_1 = {fx(sol[A])},\\ \\alpha_2 = {fx(sol[B])}"]
    return closed, gen_t, steps


def gen_recur(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    d = _recur_setup(rng, difficulty)
    c1, c2 = d["c1"], d["c2"]
    closed, _g, steps = _recur_solve(d)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    k = rng.choice([5, 6, 7])
    seq = [d["a0"], d["a1"]]
    while len(seq) <= k:
        seq.append(c1 * seq[-1] + c2 * seq[-2])
    assert all(int(closed.subs(n, i)) == seq[i] for i in range(k + 1)), "closed form disagrees with the recurrence"
    rel = f"a_n = {c1} a_{{n-1}} {'+' if c2 >= 0 else '-'} {abs(c2)} a_{{n-2}}"
    blocks = [{"p": f"Solve the recurrence ${rel}$ for $n \\ge 2$, with $a_0 = {d['a0']}$ and $a_1 = {d['a1']}$. Then find $a_{{{k}}}$."}]
    roots = f"r = {d['r1']}" + (f" (repeated)" if d["r1"] == d["r2"] else f",\\ r = {d['r2']}")
    ans = [{"p": f"Characteristic equation: $r^2 - ({c1})r - ({c2}) = 0$, roots ${roots}$."}, {"p": f"General solution: ${steps[0]}$."},
           {"p": f"From $a_0, a_1$: ${steps[1]}$, so $a_n = {sp.latex(closed)}$."},
           {"p": f"Check by the recurrence: " + ", ".join(f"$a_{{{i}}} = {v}$" for i, v in enumerate(seq)) + f"; $a_{{{k}}} = {seq[k]}$."}]
    return _out("recur", topic, marks, blocks, ans, value=seq[k], closed=str(closed))


def _recur_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_recur(rng, 2, difficulty, "recurrence-homog")
    v = q["data"]["value"]
    stem = q["blocks"][0]["p"].replace("Solve the recurrence", "For the recurrence").replace(". Then find", ", the value of").rstrip(".") + " is ______."
    return [{"p": stem}], v, q["answer"], [v + 1, v - 1, -v if v else 1]


def gen_recurnh(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = sp.Symbol("n", integer=True, nonnegative=True)
    c = rng.choice([2, 3, -2])
    kind = rng.choice(["const", "lin", "exp", "exp_res"])
    d = rng.randint(1, 5)
    if kind == "const":
        F, Ft, guess = sp.Integer(d), f"{d}", "a_n^{(p)} = A"
    elif kind == "lin":
        F, Ft, guess = d * n, f"{d}n", "a_n^{(p)} = An + B"
    elif kind == "exp":
        s = rng.choice([x for x in (2, 3, 4, 5) if x != c])
        F, Ft, guess = d * sp.Integer(s) ** n, f"{d}\\cdot {s}^n", f"a_n^{{(p)}} = A\\cdot {s}^n"
    else:
        F, Ft, guess = d * sp.Integer(c) ** n, f"{d}\\cdot ({c})^n", f"a_n^{{(p)}} = An\\,({c})^n \\text{{ (}}{c}\\text{{ is a characteristic root, so multiply by }} n)"
    a0 = rng.randint(0, 5)
    a = sp.Function("a")
    sol = sp.rsolve(a(n) - c * a(n - 1) - F, a(n), {a(0): a0})
    closed = sp.simplify(sol)
    k = rng.choice([4, 5, 6])
    seq = [a0]
    for i in range(1, k + 1):
        seq.append(c * seq[-1] + int(F.subs(n, i)))
    assert all(sp.simplify(closed.subs(n, i) - seq[i]) == 0 for i in range(k + 1)), "rsolve disagrees with iteration"
    blocks = [{"p": f"Solve $a_n = {c} a_{{n-1}} + {Ft}$ for $n \\ge 1$ with $a_0 = {a0}$, and find $a_{{{k}}}$."}]
    ans = [{"p": f"Homogeneous part: $a_n^{{(h)}} = \\alpha ({c})^n$."}, {"p": f"Particular solution: try ${guess}$ and substitute."},
           {"p": f"Adding and using $a_0 = {a0}$: $a_n = {sp.latex(closed)}$."},
           {"p": "Check: " + ", ".join(f"$a_{{{i}}} = {v}$" for i, v in enumerate(seq)) + "."}]
    return _out("recurnh", topic, marks, blocks, ans, value=seq[k], closed=str(closed))


def _recurnh_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_recurnh(rng, 2, difficulty, "recurrence-nonhomog")
    v = q["data"]["value"]
    stem = q["blocks"][0]["p"].replace("Solve", "For").replace(", and find", ", the value of").rstrip(".") + " is ______."
    return [{"p": stem}], v, q["answer"], [v + 1, v - 1, 2 * v]


# =========================================================================== relations


def pairs_tex(R: set[tuple[int, int]]) -> str:
    return "\\{" + ", ".join(f"({a},{b})" for a, b in sorted(R)) + "\\}" if R else "\\emptyset"


def props(R: set[tuple[int, int]], A: list[int]) -> dict[str, tuple[bool, str]]:
    miss = next((a for a in A if (a, a) not in R), None)
    asym = next(((a, b) for a, b in sorted(R) if (b, a) not in R), None)
    anti = next(((a, b) for a, b in sorted(R) if a != b and (b, a) in R), None)
    trans = next(((a, b, c) for a, b in sorted(R) for b2, c in sorted(R) if b == b2 and (a, c) not in R), None)
    return {
        "reflexive": (miss is None, f"$({miss},{miss}) \\notin R$" if miss is not None else "every $(a,a)$ is in $R$"),
        "symmetric": (asym is None, f"$({asym[0]},{asym[1]}) \\in R$ but $({asym[1]},{asym[0]}) \\notin R$" if asym else "every pair has its mirror"),
        "antisymmetric": (anti is None, f"$({anti[0]},{anti[1]})$ and $({anti[1]},{anti[0]})$ are both in $R$" if anti else "no two distinct elements are related both ways"),
        "transitive": (trans is None, f"$({trans[0]},{trans[1]}), ({trans[1]},{trans[2]}) \\in R$ but $({trans[0]},{trans[2]}) \\notin R$" if trans else "every chain $a \\to b \\to c$ has $(a,c)$"),
    }


def _rand_relation(rng: random.Random, A: list[int]) -> set[tuple[int, int]]:
    style = rng.randrange(4)
    if style == 0:  # rule-based
        rule = rng.choice([lambda a, b: a <= b, lambda a, b: (a + b) % 2 == 0, lambda a, b: b % a == 0, lambda a, b: a < b, lambda a, b: abs(a - b) <= 1])
        return {(a, b) for a in A for b in A if rule(a, b)}
    R = {(a, b) for a in A for b in A if rng.random() < 0.3}
    if style == 1:
        R |= {(a, a) for a in A}
    if style == 2:
        R |= {(b, a) for a, b in R}
    return R


def gen_relprop(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A = [1, 2, 3, 4]
    R = _rand_relation(rng, A)
    pr = props(R, A)
    blocks = [{"p": f"On $A = \\{{1, 2, 3, 4\\}}$ let $R = {pairs_tex(R)}$. Is $R$ reflexive, symmetric, antisymmetric, transitive? Give a reason for each."}]
    ans = [{"p": f"**{k.capitalize()}**: {'yes' if v else 'no'} - {why}."} for k, (v, why) in pr.items()]
    eq = pr["reflexive"][0] and pr["symmetric"][0] and pr["transitive"][0]
    po = pr["reflexive"][0] and pr["antisymmetric"][0] and pr["transitive"][0]
    ans.append({"p": "So $R$ is " + ("an equivalence relation" if eq else "a partial order" if po else "neither an equivalence relation nor a partial order") + "."})
    return _out("relprop", topic, marks, blocks, ans)


def _relprop_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    A = [1, 2, 3, 4]
    R = _rand_relation(rng, A)
    pr = props(R, A)
    names = ["reflexive", "symmetric", "antisymmetric", "transitive"]
    good = [i for i, k in enumerate(names) if pr[k][0]]
    if not 1 <= len(good) <= 3:
        return None
    return ([{"p": f"Let $R = {pairs_tex(R)}$ on $A = \\{{1,2,3,4\\}}$. Which of the following properties does $R$ have?"}], [k.capitalize() for k in names], good,
            [{"p": "; ".join(f"{k}: {'yes' if v else 'no'} ({w})" for k, (v, w) in pr.items()) + "."}])


def _relcount_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    n = rng.choice([3, 4, 5])
    kind = rng.choice(["reflexive", "symmetric", "antisymmetric", "reflexive and symmetric", "irreflexive"])
    f = {"reflexive": (2 ** (n * (n - 1)), f"2^{{n(n-1)}} = 2^{{{n * (n - 1)}}}"), "symmetric": (2 ** (n * (n + 1) // 2), f"2^{{n(n+1)/2}} = 2^{{{n * (n + 1) // 2}}}"),
         "antisymmetric": (2 ** n * 3 ** (n * (n - 1) // 2), f"2^n 3^{{n(n-1)/2}} = 2^{{{n}}} \\cdot 3^{{{n * (n - 1) // 2}}}"),
         "reflexive and symmetric": (2 ** (n * (n - 1) // 2), f"2^{{n(n-1)/2}} = 2^{{{n * (n - 1) // 2}}}"), "irreflexive": (2 ** (n * (n - 1)), f"2^{{n(n-1)}} = 2^{{{n * (n - 1)}}}")}[kind]
    blocks = [{"p": f"The number of {kind} relations on a set with {n} elements is ______."}]
    traps = [2 ** (n * n), 2 ** (n * (n + 1) // 2), 2 ** (n * (n - 1) // 2), 2 ** n * 3 ** (n * (n - 1) // 2), 2 ** (n * (n - 1))]
    return blocks, f[0], [{"p": f"${f[1]} = {f[0]}$."}], [t for t in traps if t != f[0]]


def bool_mat(R: set[tuple[int, int]], A: list[int]) -> list[list[int]]:
    return [[1 if (a, b) in R else 0 for b in A] for a in A]


def gen_relcomp(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A = [1, 2, 3, 4]
    R = {(a, b) for a in A for b in A if rng.random() < 0.3} or {(1, 2)}
    S = {(a, b) for a in A for b in A if rng.random() < 0.3} or {(2, 3)}
    SoR = {(a, c) for a, b in R for b2, c in S if b == b2}
    blocks = [{"p": f"On $A = \\{{1,2,3,4\\}}$, $R = {pairs_tex(R)}$ and $S = {pairs_tex(S)}$. Find $S \\circ R$, $R^{{-1}}$ and the matrix of $S \\circ R$."}]
    ans = [{"p": "$S \\circ R$: follow $R$ then $S$: " + (", ".join(f"$({a},{b}) \\in R, ({b},{c}) \\in S \\Rightarrow ({a},{c})$" for a, b in sorted(R) for b2, c in sorted(S) if b == b2) or "no chain") + "."},
           {"p": f"$S \\circ R = {pairs_tex(SoR)}$, $R^{{-1}} = {pairs_tex({(b, a) for a, b in R})}$."},
           {"p": f"$M_{{S \\circ R}} = M_R \\odot M_S$:"}, {"math": f"M_R = {mtex(bool_mat(R, A))},\\quad M_S = {mtex(bool_mat(S, A))},\\quad M_{{S\\circ R}} = {mtex(bool_mat(SoR, A))}"}]
    return _out("relcomp", topic, marks, blocks, ans, size=len(SoR))


def warshall(R: set[tuple[int, int]], A: list[int]) -> list[list[list[int]]]:
    W = bool_mat(R, A)
    out = [[row[:] for row in W]]
    n = len(A)
    for k in range(n):
        W = [[W[i][j] or (W[i][k] and W[k][j]) for j in range(n)] for i in range(n)]
        out.append([row[:] for row in W])
    return out


def gen_closure(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A = list(range(1, _n(difficulty, 4, 4, 5) + 1))
    R = set()
    while len(R) < len(A):
        a, b = rng.choice(A), rng.choice(A)
        if a != b:
            R.add((a, b))
    Ws = warshall(R, A)
    closure = {(A[i], A[j]) for i in range(len(A)) for j in range(len(A)) if Ws[-1][i][j]}
    blocks = [{"p": f"Use Warshall's algorithm to find the transitive closure of $R = {pairs_tex(R)}$ on $A = \\{{{', '.join(map(str, A))}\\}}$. Also write the reflexive and symmetric closures."}]
    ans = [{"math": ",\\quad ".join(f"W_{{{k}}} = {mtex(W)}" for k, W in enumerate(Ws))},
           {"p": f"Transitive closure $R^* = {pairs_tex(closure)}$ ({len(closure)} pairs)."},
           {"p": f"Reflexive closure: $R \\cup \\{{(a,a)\\}} = {pairs_tex(R | {(a, a) for a in A})}$. Symmetric closure: $R \\cup R^{{-1}} = {pairs_tex(R | {(b, a) for a, b in R})}$."}]
    return _out("closure", topic, marks, blocks, ans, size=len(closure))


def _closure_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_closure(rng, 2, difficulty, "closures")
    v = q["data"]["size"]
    stem = q["blocks"][0]["p"].split(" Also")[0].replace("Use Warshall's algorithm to find the transitive closure of", "The number of ordered pairs in the transitive closure of").rstrip(".") + " is ______."
    return [{"p": stem}], v, q["answer"][:2], [v - 1, v + 1, v + 2]


def bell(n: int) -> int:
    row = [1]
    for _ in range(n - 1):
        nxt = [row[-1]]
        for x in row:
            nxt.append(nxt[-1] + x)
        row = nxt
    return row[-1]


def gen_equivrel(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    m = rng.choice([3, 4, 5])
    lo, hi = rng.choice([(1, 12), (0, 10), (1, 15), (-3, 9)])
    S = list(range(lo, hi + 1))
    classes: dict[int, list[int]] = {}
    for x in S:
        classes.setdefault(x % m, []).append(x)
    blocks = [{"p": f"On $S = \\{{{lo}, {lo + 1}, \\dots, {hi}\\}}$ define $a \\, R \\, b$ iff $a \\equiv b \\pmod{{{m}}}$. Show that $R$ is an equivalence relation and list its equivalence classes."}]
    ans = [{"p": f"Reflexive: $a - a = 0$ is divisible by {m}. Symmetric: ${m} \\mid a-b \\Rightarrow {m} \\mid b-a$. Transitive: ${m} \\mid a-b$ and ${m} \\mid b-c \\Rightarrow {m} \\mid a-c$."},
           {"p": "Classes: " + "; ".join(f"$[{r}] = {sfmt(v)}$" for r, v in sorted(classes.items())) + f". They form a partition of $S$ into {len(classes)} blocks."}]
    return _out("equivrel", topic, marks, blocks, ans, classes=len(classes))


def _equivrel_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    n = rng.choice([3, 4, 5, 6])
    v = bell(n)
    return ([{"p": f"The number of equivalence relations on a set with {n} elements is ______."}], v,
            [{"p": f"Equivalence relations correspond to partitions: the Bell number $B_{{{n}}} = {v}$."}], [2 ** (n * (n - 1) // 2), math.factorial(n), bell(n - 1), 2 ** n])


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def hasse(elems: list[int]) -> list[tuple[int, int]]:
    le = lambda a, b: b % a == 0  # noqa: E731
    return [(a, b) for a in elems for b in elems if a != b and le(a, b) and not any(c not in (a, b) and le(a, c) and le(c, b) for c in elems)]


def gen_poset(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    if rng.random() < 0.5:
        n = rng.choice([12, 18, 20, 24, 30, 36, 45, 60, 72])
        E = divisors(n)
        what = f"the set of positive divisors of {n}, $D_{{{n}}}$"
    else:
        E = sorted(rng.sample([2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 20, 24, 30, 36], rng.choice([6, 7])))
        what = f"$S = {sfmt(E)}$"
    H = hasse(E)
    le = lambda a, b: b % a == 0  # noqa: E731
    maximal = [x for x in E if not any(y != x and le(x, y) for y in E)]
    minimal = [x for x in E if not any(y != x and le(y, x) for y in E)]
    greatest = next((x for x in E if all(le(y, x) for y in E)), None)
    least = next((x for x in E if all(le(x, y) for y in E)), None)
    a, b = rng.sample(E, 2)
    ub = [x for x in E if le(a, x) and le(b, x)]
    lb = [x for x in E if le(x, a) and le(x, b)]
    lub = next((x for x in ub if all(le(x, y) for y in ub)), None)
    glb = next((x for x in lb if all(le(y, x) for y in lb)), None)
    blocks = [{"p": f"Consider {what} ordered by divisibility. Draw the Hasse diagram, find the maximal, minimal, greatest and least elements, and find the upper bounds, lower bounds, LUB and GLB of $\\{{{a}, {b}\\}}$."}]
    ans = [{"p": "Hasse diagram edges (covering pairs, upward): " + ", ".join(f"{x}–{y}" for x, y in H) + "."},
           {"p": f"Maximal: {sfmt(maximal).replace(chr(92), '')}; minimal: {sfmt(minimal).replace(chr(92), '')}; greatest: {greatest if greatest is not None else 'none'}; least: {least if least is not None else 'none'}."},
           {"p": f"Upper bounds of $\\{{{a},{b}\\}}$: {', '.join(map(str, ub)) or 'none'}; LUB: {lub if lub is not None else 'does not exist'}. Lower bounds: {', '.join(map(str, lb)) or 'none'}; GLB: {glb if glb is not None else 'does not exist'}."}]
    return _out("poset", topic, marks, blocks, ans, hasse_edges=len(H))


def _poset_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    n = rng.choice([12, 18, 20, 24, 30, 36, 60, 72])
    E = divisors(n)
    H = hasse(E)
    comp = sum(1 for a in E for b in E if a != b and b % a == 0)
    return ([{"p": f"The number of edges in the Hasse diagram of the poset $(D_{{{n}}}, \\mid)$ of positive divisors of {n} under divisibility is ______."}], len(H),
            [{"p": "Covering pairs: " + ", ".join(f"{x}–{y}" for x, y in H) + f" ({len(H)} edges)."}], [comp, len(E), len(E) - 1])


def _poset_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    n = rng.choice([12, 18, 20, 24, 30, 36, 60, 72])
    E = divisors(n)
    a, b = rng.sample(E[1:-1], 2) if len(E) > 3 else (E[0], E[-1])
    stmts = [(f"The LUB of {a} and {b} is {math.lcm(a, b)}", True), (f"The GLB of {a} and {b} is {math.gcd(a, b)}", True),
             (f"The LUB of {a} and {b} is {a * b}", math.lcm(a, b) == a * b), (f"$(D_{{{n}}}, \\mid)$ is a totally ordered set", len(hasse(E)) == len(E) - 1 and all(sum(1 for x, y in hasse(E) if x == e) <= 1 for e in E)),
             (f"$(D_{{{n}}}, \\mid)$ has a greatest element", True), (f"Every element of $D_{{{n}}}$ has a complement", all(math.gcd(d, n // d) == 1 for d in E))]
    picks = rng.sample(stmts, 4)
    good = [i for i, (_s, v) in enumerate(picks) if v]
    if not 1 <= len(good) <= 3:
        return None
    return ([{"p": f"For the poset of positive divisors of {n} under divisibility, which of the following are TRUE?"}], [s for s, _v in picks], good,
            [{"p": f"In $D_{{{n}}}$: LUB = lcm, GLB = gcd; greatest element {n}, least 1. " + "; ".join(f"{s}: {'true' if v else 'false'}" for s, v in picks) + "."}])


def gen_lattice(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = rng.choice([30, 42, 12, 18, 20, 36, 70, 66])
    E = divisors(n)
    comps = {d: [e for e in E if math.lcm(d, e) == n and math.gcd(d, e) == 1] for d in E}
    blocks = [{"p": f"Show that $(D_{{{n}}}, \\mid)$ is a lattice. Give its join and meet, find the complement of every element (if it exists), and decide whether the lattice is complemented."}]
    ans = [{"p": f"Every pair $a, b$ has $a \\vee b = \\mathrm{{lcm}}(a,b)$ and $a \\wedge b = \\gcd(a,b)$ in $D_{{{n}}}$, so it is a lattice with $0 = 1$ and $1 = {n}$."},
           {"p": "Complements ($\\mathrm{lcm} = " + str(n) + "$ and $\\gcd = 1$): " + "; ".join(f"{d}: {', '.join(map(str, c)) if c else 'none'}" for d, c in comps.items()) + "."},
           {"p": f"{'Every element has a complement, so it is a complemented lattice' if all(comps.values()) else 'Some elements have no complement, so it is NOT complemented'} ({n} is {'' if all(comps.values()) else 'not '}square-free). Divisor lattices are always distributive."}]
    return _out("lattice", topic, marks, blocks, ans)


# =========================================================================== matrices (exact, sympy)


def _unit_lower(rng: random.Random, n: int, lo: int = -3, hi: int = 3) -> sp.Matrix:
    L = sp.eye(n)
    for i in range(n):
        for j in range(i):
            L[i, j] = rng.randint(lo, hi)
    return L


def _upper(rng: random.Random, n: int, piv: tuple = (1, 2, 3, -1, -2, 4)) -> sp.Matrix:
    U = sp.zeros(n)
    for i in range(n):
        U[i, i] = rng.choice(piv)
        for j in range(i + 1, n):
            U[i, j] = rng.randint(-3, 4)
    return U


def _system_tex(A: sp.Matrix, b: sp.Matrix, names: str = "xyzw") -> list[str]:
    out = []
    for i in range(A.rows):
        terms = []
        for j in range(A.cols):
            c = A[i, j]
            if c == 0:
                continue
            s = ("-" if c < 0 else "+") + " " + ("" if abs(c) == 1 else fx(abs(c))) + names[j]
            terms.append(s)
        eq = " ".join(terms).lstrip("+ ").replace("- ", "-", 1) if terms else "0"
        out.append(f"{eq} = {fx(b[i])}")
    return out


def eliminate(A: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix, list[str]]:
    """Forward elimination without row exchanges: returns L, U and the step lines."""
    n = A.rows
    U = A.copy()
    L = sp.eye(n)
    steps = []
    for k in range(n):
        if U[k, k] == 0:
            raise ValueError("zero pivot")
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            if m != 0:
                U[i, :] = U[i, :] - m * U[k, :]
                steps.append(f"\\ell_{{{i + 1}{k + 1}}} = {fx(m)}:\\ R_{i + 1} \\leftarrow R_{i + 1} - ({fx(m)})R_{k + 1}")
    return L, U, steps


def gen_elim(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = 3
    L, U = _unit_lower(rng, n, -2, 2), _upper(rng, n, (1, 2, -1, 3))
    A = L * U
    x = sp.Matrix([rng.randint(-3, 4) for _ in range(n)])
    b = A * x
    L2, U2, steps = eliminate(A)
    c = L2.inv() * b
    blocks = [{"p": "Solve by Gaussian elimination (no row exchanges needed), naming the multipliers and pivots:"}, *({"math": e} for e in _system_tex(A, b))]
    ans = [{"math": f"[A \\mid b] = {mtex(A.row_join(b))}"}, {"math": "\\\\ ".join(steps) if steps else "\\text{already upper triangular}"},
           {"math": f"[U \\mid c] = {mtex(U2.row_join(c))}"}, {"p": "Pivots: " + ", ".join(f"${fx(U2[i, i])}$" for i in range(n)) + "."},
           {"p": f"Back substitution: $z = {fx(x[2])}$, $y = {fx(x[1])}$, $x = {fx(x[0])}$."}]
    return _out("elim", topic, marks, blocks, ans, x=[int(v) for v in x])


def _elim_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_elim(rng, 2, difficulty, "elimination")
    xs = q["data"]["x"]
    which = rng.randrange(3)
    stem = [{"p": "For the system below, the value of $" + "xyz"[which] + "$ is ______."}] + q["blocks"][1:]
    return stem, xs[which], q["answer"], [xs[(which + 1) % 3], -xs[which], xs[which] + 1]


def gen_inverse(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = 3 if difficulty != "easy" else 2
    while True:
        A = _unit_lower(rng, n, -2, 2) * _upper(rng, n, (1, -1, 2) if n == 3 else (1, 2, -1, 3))
        if A.det() != 0 and max(abs(v) for v in A) <= 9:
            break
    inv = A.inv()
    aug = A.row_join(sp.eye(n))
    R = aug.rref()[0]
    blocks = [{"p": f"Find $A^{{-1}}$ by Gauss-Jordan elimination, where"}, {"math": f"A = {mtex(A)}"}]
    ans = [{"math": f"[A \\mid I] = {mtex(aug)}"}, {"p": "Eliminate below the pivots, then above them, then divide each row by its pivot:"},
           {"math": f"[I \\mid A^{{-1}}] = {mtex(R)}"}, {"math": f"A^{{-1}} = {mtex(inv)}"}, {"p": f"Check: $\\det A = {fx(A.det())} \\ne 0$, and $AA^{{-1}} = I$."}]
    return _out("inverse", topic, marks, blocks, ans, det=int(A.det()))


def _inverse_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    while True:
        A = _unit_lower(rng, 3, -2, 2) * _upper(rng, 3, (1, -1, 2))
        if A.det() != 0:
            break
    inv = A.inv()
    i, j = rng.randrange(3), rng.randrange(3)
    v = inv[i, j]
    return ([{"p": f"For $A = {mtex(A)}$, the entry in row {i + 1}, column {j + 1} of $A^{{-1}}$ is ______."}], num(v),
            [{"math": f"A^{{-1}} = {mtex(inv)}"}], [num(A[i, j]), num(inv[j, i]) if inv[j, i] != v else num(-v), num(1 / A[i, j]) if A[i, j] else num(v + 1)])


def gen_lu(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = 3
    L, U = _unit_lower(rng, n, -3, 3), _upper(rng, n, (1, 2, 3, -1, -2))
    A = L * U
    L2, U2, steps = eliminate(A)
    blocks = [{"p": "Factor $A = LU$ (no row exchanges) and find $\\det A$:"}, {"math": f"A = {mtex(A)}"}]
    ans = [{"math": "\\\\ ".join(steps) if steps else "\\text{no elimination needed}"}, {"math": f"L = {mtex(L2)},\\quad U = {mtex(U2)}"},
           {"p": "$\\det A = $ product of pivots $= " + " \\cdot ".join(fx(U2[i, i]) for i in range(n)) + f" = {fx(A.det())}$."}]
    return _out("lu", topic, marks, blocks, ans, det=int(A.det()))


def _lu_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    L, U = _unit_lower(rng, 3, -3, 3), _upper(rng, 3, (1, 2, 3, -1, -2))
    A = L * U
    which = rng.choice(["l21", "l31", "l32", "u33", "det"])
    v = {"l21": L[1, 0], "l31": L[2, 0], "l32": L[2, 1], "u33": U[2, 2], "det": A.det()}[which]
    what = {"l21": "$\\ell_{21}$ of $L$", "l31": "$\\ell_{31}$ of $L$", "l32": "$\\ell_{32}$ of $L$", "u33": "the third pivot $u_{33}$", "det": "$\\det A$"}[which]
    return ([{"p": f"In the factorization $A = LU$ ($L$ unit lower triangular) of $A = {mtex(A)}$, the value of {what} is ______."}], num(v),
            [{"math": f"L = {mtex(L)},\\ U = {mtex(U)}"}], [num(-v), num(v + 1), num(A[2, 2])])


def gen_transpose(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    swap = rng.choice([(0, 1), (0, 2)])
    L, U = _unit_lower(rng, 3, -2, 2), _upper(rng, 3, (1, 2, -1, 3))
    L[swap[1], 0] = 0  # the row that moves to the top must start with 0, so A has a zero first pivot
    M = L * U  # = PA
    P = sp.eye(3)
    P.row_swap(*swap)
    A = P * M  # P is its own inverse for one swap, so PA = M = LU
    L2, U2, steps = eliminate(P * A)
    blocks = [{"p": "The first pivot of $A$ is zero. Find a permutation matrix $P$ and the factors with $PA = LU$; also write $A^T$ and check that $P^{-1} = P^T$."}, {"math": f"A = {mtex(A)}"}]
    elim = "\\\\ ".join(steps) if steps else "\\text{no elimination needed}"
    ans = [{"math": f"P = {mtex(P)},\\quad PA = {mtex(P * A)}"}, {"math": elim},
           {"math": f"L = {mtex(L2)},\\quad U = {mtex(U2)}"}, {"math": f"A^T = {mtex(A.T)},\\quad P^TP = {mtex(P.T * P)} = I"}]
    return _out("transpose", topic, marks, blocks, ans)


def _rank_matrix(rng: random.Random, m: int, n: int, r: int) -> sp.Matrix:
    """An m x n integer matrix of rank exactly r with small entries."""
    while True:
        C = sp.Matrix(m, r, lambda i, j: rng.randint(-2, 3))
        Rm = sp.Matrix(r, n, lambda i, j: rng.randint(-2, 3))
        A = C * Rm
        if A.rank() == r and max(abs(v) for v in A) <= 12 and all(any(A[i, j] for j in range(n)) for i in range(m)):
            return A


def gen_nullspace(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    m, n, r = rng.choice([(3, 4, 2), (3, 4, 2), (3, 5, 2), (2, 4, 2), (3, 4, 1)])
    A = _rank_matrix(rng, m, n, r)
    R, piv = A.rref()
    N = A.nullspace()
    free = [j for j in range(n) if j not in piv]
    blocks = [{"p": "Reduce $A$ to its row reduced echelon form $R$, name the pivot and free variables, and find the special solutions (a basis of $N(A)$):"}, {"math": f"A = {mtex(A)}"}]
    ans = [{"math": f"R = \\operatorname{{rref}}(A) = {mtex(R)}"},
           {"p": f"Pivot columns {', '.join(str(j + 1) for j in piv)} (rank $r = {len(piv)}$); free variables " + ", ".join(f"$x_{j + 1}$" for j in free) + "."},
           {"p": "Special solutions (one free variable = 1, the others 0):"}, {"math": ",\\quad ".join(f"s_{k + 1} = {vtex(list(v))}" for k, v in enumerate(N))},
           {"p": f"$N(A)$ = all combinations of these; $\\dim N(A) = n - r = {n} - {len(piv)} = {len(N)}$."}]
    return _out("nullspace", topic, marks, blocks, ans, nullity=len(N))


def _nullspace_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    A = _rank_matrix(rng, 3, 4, 2)
    N = A.nullspace()
    good_vecs = [N[0] * rng.choice([1, 2, -1]), N[0] + N[1] if len(N) > 1 else 2 * N[0]]
    bad = [sp.Matrix([rng.randint(-2, 2) for _ in range(4)]) for _ in range(6)]
    bad = [v for v in bad if (A * v).norm() != 0][:2]
    if len(bad) < 2:
        return None
    vecs = good_vecs[: rng.choice([1, 2])] + bad
    while len(vecs) < 4:
        vecs.append(N[-1] * 3)
    rng.shuffle(vecs)
    good = [i for i, v in enumerate(vecs) if (A * v).norm() == 0]
    if not 1 <= len(good) <= 3:
        return None
    return ([{"p": f"Which of the following vectors lie in the null space of $A = {mtex(A)}$?"}], [f"${vtex(list(v))}$" for v in vecs], good,
            [{"p": "Check $Av = 0$: " + "; ".join(f"$A{vtex(list(v))} = {vtex(list(A * v))}$" for v in vecs) + "."}])


def gen_rank(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    m, n, r = rng.choice([(3, 3, 2), (3, 4, 2), (3, 3, 2), (3, 4, 3)])
    A = _rank_matrix(rng, m, n, r)
    solvable = rng.random() < 0.6
    if solvable:
        xp = sp.Matrix([rng.randint(-2, 3) for _ in range(n)])
        b = A * xp
    else:
        b = sp.Matrix([rng.randint(-3, 5) for _ in range(m)])
        if A.row_join(b).rank() == A.rank():
            b[m - 1] += 1
    aug = A.row_join(b)
    R = aug.rref()[0]
    ra, rb = A.rank(), aug.rank()
    blocks = [{"p": "Find the rank of $A$ and decide whether $Ax = b$ has no solution, exactly one, or infinitely many. If it is solvable, write the complete solution $x = x_p + x_n$."},
              {"math": f"A = {mtex(A)},\\quad b = {vtex(list(b))}"}]
    ans = [{"math": f"\\operatorname{{rref}}[A \\mid b] = {mtex(R)}"}, {"p": f"rank$(A) = {ra}$, rank$[A \\mid b] = {rb}$."}]
    if ra != rb:
        ans.append({"p": "A row reads $0 = $ (nonzero), so the system is **inconsistent: no solution**."})
        kind = "none"
    else:
        sol, params = A.gauss_jordan_solve(b)
        x_p = sol.subs({t: 0 for t in params})
        N = A.nullspace()
        kind = "one" if not N else "infinite"
        ans.append({"p": f"Solvable. Particular solution (free variables 0): $x_p = {vtex(list(x_p))}$."})
        if N:
            ans.append({"math": "x = " + vtex(list(x_p)) + " + " + " + ".join(f"c_{i + 1}{vtex(list(v))}" for i, v in enumerate(N))})
            ans.append({"p": f"$r = {ra} < n = {n}$: **infinitely many** solutions."})
        else:
            ans.append({"p": f"$r = n = {n}$: **exactly one** solution."})
    return _out("rank", topic, marks, blocks, ans, rank=ra, solutions=kind)


def _rank_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    m, n = rng.choice([(3, 3), (3, 4), (4, 4), (4, 3)])
    r = rng.randint(1, min(m, n) - (1 if rng.random() < 0.7 else 0))
    A = _rank_matrix(rng, m, n, r)
    return ([{"p": f"The rank of $ {mtex(A)} $ is ______."}], r, [{"math": f"\\operatorname{{rref}} = {mtex(A.rref()[0])}"}, {"p": f"{r} pivots, so rank = {r}."}],
            [min(m, n), r + 1, r - 1 if r > 1 else 3])


def gen_basis(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    dim = rng.choice([3, 4])
    k = rng.choice([3, 4]) if dim == 4 else 3
    r = rng.choice([k, k - 1, k - 1])
    M = _rank_matrix(rng, dim, k, r)
    vs = [list(M[:, j]) for j in range(k)]
    R, piv = M.rref()
    blocks = [{"p": f"Are the vectors " + ", ".join(f"$v_{i + 1} = {vtex(v)}$" for i, v in enumerate(vs)) + f" in $\\mathbb{{R}}^{dim}$ linearly independent? Find a basis and the dimension of their span."}]
    ans = [{"math": f"[v_1 \\cdots v_{k}] \\to \\operatorname{{rref}} = {mtex(R)}"}, {"p": f"Rank $= {len(piv)}$ " + ("= number of vectors, so they are **independent**." if len(piv) == k else f"< {k}, so they are **dependent**.")},
           {"p": "Basis of the span (the pivot columns of the ORIGINAL vectors): " + ", ".join(f"$v_{j + 1}$" for j in piv) + f"; dimension $= {len(piv)}$."}]
    if len(piv) < k:
        N = M.nullspace()[0]
        ans.append({"p": "A dependency: $" + " + ".join(f"({fx(N[j])})v_{j + 1}" for j in range(k) if N[j] != 0) + " = 0$."})
    return _out("basis", topic, marks, blocks, ans, dim=len(piv))


def _basis_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_basis(rng, 2, difficulty, "basis")
    v = q["data"]["dim"]
    stem = q["blocks"][0]["p"].split(" Are the vectors ")[-1].split(" linearly independent")[0]
    return [{"p": f"The dimension of the subspace spanned by {stem} is ______."}], v, q["answer"], [v + 1, v - 1, 4]


def _basis_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    sets = []
    for _ in range(4):
        r = rng.choice([2, 3, 3])
        M = _rank_matrix(rng, 3, 3, r)
        sets.append(([list(M[:, j]) for j in range(3)], r == 3))
    good = [i for i, (_v, ok) in enumerate(sets) if ok]
    if not 1 <= len(good) <= 3:
        return None
    return ([{"p": "Which of the following sets of vectors form a basis of $\\mathbb{R}^3$?"}], ["$\\{" + ", ".join(vtex(v) for v in vs) + "\\}$" for vs, _ok in sets], good,
            [{"p": "Three vectors form a basis of $\\mathbb{R}^3$ iff the matrix with them as columns has rank 3 (nonzero determinant): "
                   + "; ".join(f"det = {fx(sp.Matrix(vs).T.det())}" for vs, _ok in sets) + "."}])


def gen_four(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    m, n, r = rng.choice([(3, 4, 2), (3, 5, 2), (4, 3, 2), (3, 3, 2)])
    A = _rank_matrix(rng, m, n, r)
    R, piv = A.rref()
    N, LN = A.nullspace(), A.T.nullspace()
    rows = [list(R[i, :]) for i in range(len(piv))]
    blocks = [{"p": "Find the dimension and a basis of each of the four fundamental subspaces of"}, {"math": f"A = {mtex(A)}"}]
    ans = [{"math": f"\\operatorname{{rref}}(A) = {mtex(R)},\\quad r = {r}"},
           {"p": f"$C(A)$ (in $\\mathbb{{R}}^{m}$), dim {r}: pivot columns of $A$: " + ", ".join(f"${vtex(list(A[:, j]))}$" for j in piv) + "."},
           {"p": f"$C(A^T)$ (row space, in $\\mathbb{{R}}^{n}$), dim {r}: nonzero rows of $R$: " + ", ".join(f"$({', '.join(fx(x) for x in row)})$" for row in rows) + "."},
           {"p": f"$N(A)$ (in $\\mathbb{{R}}^{n}$), dim $n - r = {n - r}$: " + (", ".join(f"${vtex(list(v))}$" for v in N) or "only $0$") + "."},
           {"p": f"$N(A^T)$ (in $\\mathbb{{R}}^{m}$), dim $m - r = {m - r}$: " + (", ".join(f"${vtex(list(v))}$" for v in LN) or "only $0$") + "."}]
    return _out("four", topic, marks, blocks, ans)


def _four_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    m, n = rng.choice([(5, 7), (4, 6), (6, 4), (7, 5), (3, 8), (8, 3)])
    r = rng.randint(1, min(m, n) - 1)
    which = rng.choice(["N(A)", "N(A^T)", "C(A^T)", "C(A)"])
    v = {"N(A)": n - r, "N(A^T)": m - r, "C(A^T)": r, "C(A)": r}[which]
    return ([{"p": f"$A$ is a ${m} \\times {n}$ matrix of rank {r}. The dimension of ${which}$ is ______."}], v,
            [{"p": f"$\\dim C(A) = \\dim C(A^T) = r = {r}$, $\\dim N(A) = n - r = {n - r}$, $\\dim N(A^T) = m - r = {m - r}$."}], [n - r, m - r, r, n, m])


def gen_ortho(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A = _rank_matrix(rng, 2, 3, 2)
    N = A.nullspace()[0]
    N = N * sp.lcm([sp.fraction(x)[1] for x in N])
    u = sp.Matrix([rng.randint(-3, 3) for _ in range(3)])
    w = sp.Matrix([rng.randint(-3, 3), rng.randint(-3, 3), 0])
    k = sp.Symbol("k")
    w[2] = k
    if u[2] == 0:
        u[2] = 1
    ksol = sp.solve(u.dot(w), k)[0]
    blocks = [{"p": f"(a) Find $k$ so that $u = {vtex(list(u))}$ and $w = {vtex([w[0], w[1], 'k'])}$ are orthogonal. (b) Find a basis of the orthogonal complement of the row space of $A = {mtex(A)}$."}]
    ans = [{"p": f"(a) $u^T w = {fx(u[0] * w[0] + u[1] * w[1])} {plus(u[2], 'k')} = 0 \\Rightarrow k = {fx(ksol)}$."},
           {"p": f"(b) $C(A^T)^\\perp = N(A)$. Solving $Ax = 0$ gives $N(A) = \\text{{span}}\\{{{vtex(list(N))}\\}}$; check: each row of $A$ dotted with it is 0."}]
    return _out("ortho", topic, marks, blocks, ans, k=num(ksol))


def _ortho_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_ortho(rng, 2, difficulty, "orthogonal")
    v = q["data"]["k"]
    stem = q["blocks"][0]["p"].split(" (b)")[0].replace("(a) Find $k$ so that", "If").replace("are orthogonal.", "are orthogonal, then $k$ = ______.")
    return [{"p": stem}], v, q["answer"][:1], [-v if v else 1, v + 1, 0]


def gen_proj(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    if rng.random() < 0.5:
        a = sp.Matrix([rng.randint(-2, 3) or 1 for _ in range(3)])
        b = sp.Matrix([rng.randint(-3, 4) for _ in range(3)])
        x = a.dot(b) / a.dot(a)
        p = x * a
        e = b - p
        blocks = [{"p": f"Project $b = {vtex(list(b))}$ onto the line through $a = {vtex(list(a))}$. Find $p$, the error $e = b - p$, and the projection matrix $P$."}]
        P = a * a.T / a.dot(a)
        ans = [{"p": f"$\\hat{{x}} = \\dfrac{{a^Tb}}{{a^Ta}} = \\dfrac{{{fx(a.dot(b))}}}{{{fx(a.dot(a))}}}$, so $p = \\hat{{x}}a = {vtex(list(p))}$."},
               {"p": f"$e = b - p = {vtex(list(e))}$; check $a^Te = {fx(a.dot(e))}$."}, {"math": f"P = \\frac{{aa^T}}{{a^Ta}} = {mtex(P)}"}]
        return _out("proj", topic, marks, blocks, ans, xhat=num(x))
    npts = rng.choice([3, 4])
    ts = sorted(rng.sample(range(-2, 5), npts))
    bs = [rng.randint(-2, 8) for _ in ts]
    A = sp.Matrix([[1, t] for t in ts])
    b = sp.Matrix(bs)
    AtA, Atb = A.T * A, A.T * b
    xh = AtA.solve(Atb)
    blocks = [{"p": "Find the least-squares line $b = C + Dt$ through the points " + ", ".join(f"$({t}, {v})$" for t, v in zip(ts, bs)) + "."}]
    ans = [{"math": f"A = {mtex(A)},\\quad b = {vtex(bs)}"}, {"math": f"A^TA = {mtex(AtA)},\\quad A^Tb = {vtex(list(Atb))}"},
           {"p": f"Normal equations $A^TA\\hat{{x}} = A^Tb$ give $C = {fx(xh[0])}$, $D = {fx(xh[1])}$: the line is $b = {fx(xh[0])} + ({fx(xh[1])})t$."},
           {"p": f"Error $e = b - A\\hat{{x}} = {vtex(list(b - A * xh))}$ is orthogonal to both columns of $A$."}]
    return _out("proj", topic, marks, blocks, ans, C=num(xh[0]), D=num(xh[1]))


def _proj_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_proj(rng, 2, difficulty, "projections")
    if "D" in q["data"]:
        v = q["data"]["D"]
        return [{"p": q["blocks"][0]["p"].replace("Find the least-squares line $b = C + Dt$", "The slope $D$ of the least-squares line $b = C + Dt$").rstrip(".") + " is ______ (up to 2 decimals)."}], v, q["answer"], [q["data"]["C"], round(v + 0.5, 2), round(-v, 2) if v else 1]
    v = q["data"]["xhat"]
    stem = q["blocks"][0]["p"].split(". Find")[0]
    return [{"p": stem + ". The scalar $\\hat{x}$ with $p = \\hat{x}a$ is ______ (up to 2 decimals)."}], v, q["answer"], [round(1 / v, 2) if v else 1, round(-v, 2) if v else 2, round(v + 1, 2)]


def gen_gs(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    while True:
        a = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        b = sp.Matrix([rng.randint(-2, 3) for _ in range(3)])
        c = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        k = 2 if difficulty == "easy" else rng.choice([2, 3])
        M = sp.Matrix.hstack(a, b, c)[:, :k]
        if M.rank() == k and a.dot(a) > 0:
            break
    vs = [a, b, c][:k]
    names = "ABC"
    orth = []
    lines = []
    for i, v in enumerate(vs):
        w = v
        terms = []
        for j, q in enumerate(orth):
            coef = q.dot(v) / q.dot(q)
            w = w - coef * q
            terms.append(f"\\frac{{{names[j]}^T {'abc'[i]}}}{{{names[j]}^T{names[j]}}}{names[j]}")
        orth.append(w)
        lines.append(f"{names[i]} = {'abc'[i]}" + ("".join(f" - {t}" for t in terms)) + f" = {vtex(list(w))}")
    qs = [w / sp.sqrt(w.dot(w)) for w in orth]
    Q = sp.Matrix.hstack(*qs)
    R = sp.simplify(Q.T * M)
    blocks = [{"p": "Apply Gram-Schmidt to " + ", ".join(f"${'abc'[i]} = {vtex(list(v))}$" for i, v in enumerate(vs)) + " to get orthonormal $q$'s, and write $A = QR$."}]
    ans = [{"math": "\\\\ ".join(lines)}, {"math": ",\\quad ".join(f"q_{i + 1} = \\frac{{{names[i]}}}{{\\|{names[i]}\\|}} = {vtex([sp.nsimplify(x) for x in q])}" for i, q in enumerate(qs))},
           {"math": f"R = Q^TA = {mtex(R)}"}]
    return _out("gs", topic, marks, blocks, ans, norms2=[int(w.dot(w)) if w.dot(w).is_Integer else str(w.dot(w)) for w in orth])


def _gs_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    while True:
        a = sp.Matrix([rng.randint(-2, 2) for _ in range(3)])
        b = sp.Matrix([rng.randint(-2, 3) for _ in range(3)])
        if a.dot(a) and sp.Matrix.hstack(a, b).rank() == 2:
            break
    B = b - a.dot(b) / a.dot(a) * a
    v = B.dot(B)
    return ([{"p": f"Gram-Schmidt is applied to $a = {vtex(list(a))}$ and $b = {vtex(list(b))}$ (in that order). Before normalising, the second vector is $B = b - \\frac{{a^Tb}}{{a^Ta}}a$. The value of $\\|B\\|^2$ is ______ (up to 2 decimals)."}],
            num(v), [{"p": f"$B = {vtex(list(B))}$, $\\|B\\|^2 = {fx(v)}$."}], [num(b.dot(b)), num(b.dot(b) - v + 1), num(a.dot(b))])


def _eig_matrix(rng: random.Random, n: int, symmetric: bool = False) -> tuple[sp.Matrix, list[int], sp.Matrix]:
    while True:
        lams = [rng.choice([-2, -1, 1, 2, 3, 4, 5]) for _ in range(n)]
        if len(set(lams)) < n:
            continue
        if symmetric:
            if n == 2:
                a, b = rng.choice([(1, 1), (1, -1), (1, 2), (2, 1)])
                S = sp.Matrix([[a, -b], [b, a]])
            else:
                S = sp.Matrix([[1, 1, 1], [1, -1, 0], [1, 1, -2]])
            Q = sp.Matrix.hstack(*[S[:, j] / sp.sqrt(S[:, j].dot(S[:, j])) for j in range(n)])
            A = sp.simplify(Q * sp.diag(*lams) * Q.T)
            if all(x.is_Rational for x in A):
                return A, lams, Q
            continue
        S = _unit_lower(rng, n, -1, 1) * _upper(rng, n, (1, -1))
        A = S * sp.diag(*lams) * S.inv()
        if all(x.is_Integer and abs(x) <= 12 for x in A):
            return A, lams, S


def gen_eig(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = 2 if difficulty == "easy" or rng.random() < 0.5 else 3
    A, lams, _S = _eig_matrix(rng, n)
    lam = sp.Symbol("lambda")
    cp = sp.expand((A - lam * sp.eye(n)).det())
    blocks = [{"p": "Find the eigenvalues and eigenvectors of"}, {"math": f"A = {mtex(A)}"}, {"p": "Check your eigenvalues with the trace and the determinant."}]
    ans = [{"p": f"$\\det(A - \\lambda I) = {sp.latex(cp)} = {sp.latex(sp.factor(cp))} = 0$, so $\\lambda = {', '.join(map(str, sorted(lams)))}$."}]
    for L in sorted(lams):
        v = (A - L * sp.eye(n)).nullspace()[0]
        v = v * sp.lcm([sp.fraction(x)[1] for x in v])
        shift = f"A - {L}I" if L > 0 else f"A + {-L}I"
        ans.append({"p": f"$\\lambda = {L}$: solve $({shift})x = 0$, eigenvector $x = {vtex(list(v))}$."})
    ssum = " + ".join(f"({x})" if x < 0 else str(x) for x in sorted(lams))
    ans.append({"p": f"Check: trace $= {fx(A.trace())} = {ssum}$, $\\det A = {fx(A.det())} = " + " \\cdot ".join(f"({x})" for x in sorted(lams)) + "$."})
    return _out("eig", topic, marks, blocks, ans, eigenvalues=sorted(lams))


def _eig_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    n = rng.choice([2, 3])
    A, lams, _S = _eig_matrix(rng, n)
    which = rng.choice(["largest", "smallest", "sum of squares of the"])
    v = max(lams) if which == "largest" else min(lams) if which == "smallest" else sum(x * x for x in lams)
    return ([{"p": f"The {which} eigenvalue{'s' if which.startswith('sum') else ''} of $ {mtex(A)} $ {'is' if not which.startswith('sum') else 'equals'} ______."}], v,
            [{"p": f"Eigenvalues: {', '.join(map(str, sorted(lams)))} (trace {fx(A.trace())}, det {fx(A.det())})."}], [int(A.trace()), int(A.det()), -v if v else 1])


def _eig_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    A, lams, _S = _eig_matrix(rng, 3)
    others = [x for x in range(-3, 7) if x not in lams]
    k = rng.choice([1, 2, 3])
    opts = rng.sample(lams, k) + rng.sample(others, 4 - k)
    rng.shuffle(opts)
    good = [i for i, x in enumerate(opts) if x in lams]
    return ([{"p": f"Which of the following are eigenvalues of $ {mtex(A)} $?"}], [str(x) for x in opts], good,
            [{"p": f"$\\det(A - \\lambda I) = {sp.latex(sp.factor((A - sp.Symbol('lambda') * sp.eye(3)).det()))}$: eigenvalues {', '.join(map(str, sorted(lams)))}."}])


def gen_diag(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A, lams, S = _eig_matrix(rng, 2)
    k = rng.choice([3, 4, 5, 10])
    Ak = A ** k
    Lk = sp.diag(*[x ** k for x in lams])
    blocks = [{"p": f"Diagonalize $A = {mtex(A)}$ as $A = S\\Lambda S^{{-1}}$ and use it to find $A^{{{k}}}$."}]
    ans = [{"p": f"Eigenvalues ${lams[0]}, {lams[1]}$ with eigenvectors as the columns of $S$:"}, {"math": f"S = {mtex(S)},\\ \\Lambda = {mtex(sp.diag(*lams))},\\ S^{{-1}} = {mtex(S.inv())}"},
           {"math": f"A^{{{k}}} = S\\Lambda^{{{k}}}S^{{-1}} = {mtex(S)}{mtex(Lk)}{mtex(S.inv())} = {mtex(Ak)}"}]
    return _out("diag", topic, marks, blocks, ans)


def _diag_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    A, lams, S = _eig_matrix(rng, 2)
    k = rng.choice([3, 4, 5, 6])
    Ak = A ** k
    i, j = rng.randrange(2), rng.randrange(2)
    return ([{"p": f"For $A = {mtex(A)}$, the entry in row {i + 1}, column {j + 1} of $A^{{{k}}}$ is ______."}], int(Ak[i, j]),
            [{"math": f"A = S\\Lambda S^{{-1}},\\ S = {mtex(S)},\\ \\Lambda = {mtex(sp.diag(*lams))};\\ A^{{{k}}} = {mtex(Ak)}"}], [int(A[i, j]) ** k, int(Ak[j, i]) if Ak[j, i] != Ak[i, j] else int(Ak[i, j]) + 1, sum(x ** k for x in lams)])


def gen_spectral(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    n = 2 if difficulty != "hard" else rng.choice([2, 3])
    A, lams, Q = _eig_matrix(rng, n, symmetric=True)
    blocks = [{"p": f"$A = {mtex(A)}$ is symmetric. Find an orthogonal $Q$ and diagonal $\\Lambda$ with $A = Q\\Lambda Q^T$, and write the spectral decomposition $A = \\sum \\lambda_i q_iq_i^T$."}]
    terms = " + ".join(f"({x}){mtex(sp.simplify(Q[:, i] * Q[:, i].T))}" for i, x in enumerate(lams))
    ans = [{"p": f"Eigenvalues: {', '.join(map(str, lams))} (real, as for every symmetric matrix)."}, {"math": f"Q = {mtex(sp.simplify(Q))},\\quad \\Lambda = {mtex(sp.diag(*lams))}"},
           {"p": "The columns of $Q$ are orthonormal: $Q^TQ = I$."}, {"math": f"A = {terms}"}]
    return _out("spectral", topic, marks, blocks, ans, eigenvalues=lams)


def _posdef_setup(rng: random.Random) -> tuple[sp.Matrix, sp.Expr, sp.Symbol]:
    k = sp.Symbol("k")
    if rng.random() < 0.5:
        a, b = rng.randint(1, 4), rng.randint(1, 4)
        A = sp.Matrix([[a, b], [b, k]])
        cond = sp.Rational(b * b, a)
    else:
        a, b, c = rng.randint(2, 4), rng.randint(-2, 2), rng.randint(-1, 2)
        A = sp.Matrix([[a, b, 0], [b, k, c], [0, c, a]])
        # leading minors: a > 0; a k - b^2 > 0; det > 0
        d3 = sp.solve(sp.Eq(A.det(), 0), k)[0]
        cond = max(sp.Rational(b * b, a), d3)
    return A, cond, k


def gen_posdef(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A, cond, k = _posdef_setup(rng)
    n = A.rows
    minors = [A[:i, :i].det() for i in range(1, n + 1)]
    blocks = [{"p": f"For which values of $k$ is $A = {mtex(A)}$ positive definite? Use the leading principal minors (or pivots)."}]
    ans = [{"p": "Leading principal minors: " + ", ".join(f"$D_{i + 1} = {sp.latex(sp.expand(m))}$" for i, m in enumerate(minors)) + "."},
           {"p": f"All must be $> 0$: this needs $k > {fx(cond)}$. So $A$ is positive definite exactly when $k > {fx(cond)}$."}]
    return _out("posdef", topic, marks, blocks, ans, kmin=num(cond))


def _posdef_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    A, cond, _k = _posdef_setup(rng)
    v = sp.floor(cond) + 1
    return ([{"p": f"The smallest INTEGER $k$ for which $ {mtex(A)} $ is positive definite is ______."}], int(v),
            [{"p": f"Leading minors positive $\\iff k > {fx(cond)}$; smallest integer: {int(v)}."}], [int(v) - 1, int(sp.ceiling(cond)), int(v) + 1])


def _posdef_msq(rng: random.Random, difficulty: str) -> tuple[list, list[str], list[int], list] | None:
    mats = []
    for _ in range(4):
        a, b, c = rng.randint(-1, 5), rng.randint(-3, 3), rng.randint(-1, 5)
        mats.append(sp.Matrix([[a, b], [b, c]]))
    good = [i for i, M in enumerate(mats) if M[0, 0] > 0 and M.det() > 0]
    if not 1 <= len(good) <= 3:
        return None
    return ([{"p": "Which of the following symmetric matrices are positive definite?"}], [f"${mtex(M)}$" for M in mats], good,
            [{"p": "Test $a > 0$ and $ac - b^2 > 0$: " + "; ".join(f"$a = {M[0, 0]}, \\det = {M.det()}$" for M in mats) + "."}])


def gen_svd(rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    A = rng.choice([sp.Matrix([[3, 0], [4, 5]]), sp.Matrix([[2, 2], [-1, 1]]), sp.Matrix([[0, 3], [2, 0]]), sp.Matrix([[1, 1], [1, 1]]) * rng.choice([1, 2]),
                    sp.Matrix([[2, 0], [0, -3]]), sp.Matrix([[1, 2], [2, 4]]), sp.Matrix([[4, 4], [-3, 3]]), sp.Matrix([[1, 1], [0, 0]]), sp.Matrix([[3, 1], [1, 3]])])
    AtA = A.T * A
    ev = sorted(AtA.eigenvals().keys(), key=lambda x: -float(x))
    sig = [sp.sqrt(x) for x in ev]
    vecs = []
    for e in ev:
        v = (AtA - e * sp.eye(2)).nullspace()[0]
        vecs.append(sp.simplify(v / sp.sqrt(v.dot(v))))
    if ev[0] == ev[-1]:
        vecs = [sp.Matrix([1, 0]), sp.Matrix([0, 1])]
    us = [sp.simplify(A * v / s) if s != 0 else None for v, s in zip(vecs, sig)]
    blocks = [{"p": f"Find the singular values of $A = {mtex(A)}$ and the singular vectors $v_i$ and $u_i$ for the nonzero ones."}]
    ans = [{"math": f"A^TA = {mtex(AtA)}"}, {"p": "Eigenvalues of $A^TA$: " + ", ".join(f"${fx(x)}$" for x in ev) + "; singular values $\\sigma = " + ", ".join(fx(s) for s in sig) + "$."}]
    for i, (v, s, u) in enumerate(zip(vecs, sig, us), 1):
        ans.append({"p": f"$\\sigma_{i} = {fx(s)}$: $v_{i} = {vtex(list(v))}$" + (f", $u_{i} = Av_{i}/\\sigma_{i} = {vtex(list(u))}$." if u is not None else " (zero singular value; $u_2$ completes an orthonormal basis).")})
    ans.append({"p": f"Rank of $A$ = number of nonzero singular values = {sum(1 for s in sig if s != 0)}."})
    return _out("svd", topic, marks, blocks, ans, sigma=[num(s) for s in sig])


def _svd_value(rng: random.Random, difficulty: str) -> tuple[list, Any, list, list]:
    q = gen_svd(rng, 2, difficulty, "svd")
    s = q["data"]["sigma"]
    what = q["blocks"][0]["p"].split("values of ")[1].split(" and the")[0]
    return [{"p": f"The largest singular value of {what} is ______ (up to 2 decimals)."}], s[0], q["answer"], [round(s[0] ** 2, 2), s[-1], round(s[0] / 2, 2)]


# =========================================================================== registry

GENERATORS: dict[str, Callable[..., dict[str, Any]]] = {
    "truth": gen_truth, "conditional": gen_conditional, "equiv": gen_equiv, "taut": gen_taut, "quant": gen_quant, "argument": gen_argument, "nf": gen_nf,
    "sets": gen_sets, "func": gen_func, "sum": gen_sum, "receval": gen_receval,
    "count": gen_count, "inclexcl": gen_inclexcl, "pigeon": gen_pigeon, "permcomb": gen_permcomb, "binom": gen_binom, "recur": gen_recur, "recurnh": gen_recurnh,
    "relprop": gen_relprop, "relcomp": gen_relcomp, "closure": gen_closure, "equivrel": gen_equivrel, "poset": gen_poset, "lattice": gen_lattice,
    "elim": gen_elim, "inverse": gen_inverse, "lu": gen_lu, "transpose": gen_transpose, "nullspace": gen_nullspace, "rank": gen_rank, "basis": gen_basis, "four": gen_four,
    "ortho": gen_ortho, "proj": gen_proj, "gs": gen_gs, "eig": gen_eig, "diag": gen_diag, "spectral": gen_spectral, "posdef": gen_posdef, "svd": gen_svd,
}

VALUE_GENERATORS: dict[str, Callable[..., tuple]] = {
    "truth": _taut_value, "taut": _taut_value, "nf": _nf_value, "sets": _sets_value, "func": _func_value, "sum": _sum_value, "receval": _receval_value,
    "count": _count_value, "inclexcl": _inclexcl_value, "pigeon": _pigeon_value, "permcomb": _permcomb_value, "binom": _binom_value,
    "recur": _recur_value, "recurnh": _recurnh_value, "relprop": _relcount_value, "relcomp": _relcount_value, "closure": _closure_value, "equivrel": _equivrel_value,
    "poset": _poset_value, "lattice": _poset_value, "elim": _elim_value, "inverse": _inverse_value, "lu": _lu_value, "transpose": _lu_value, "nullspace": _rank_value,
    "rank": _rank_value, "basis": _basis_value, "four": _four_value, "ortho": _ortho_value, "proj": _proj_value, "gs": _gs_value, "eig": _eig_value, "diag": _diag_value,
    "spectral": _eig_value, "posdef": _posdef_value, "svd": _svd_value,
}

# text-answer MCQs (no number to type): conditional statements
CHOICE_GENERATORS: dict[str, Callable[..., tuple]] = {"conditional": _cond_choice}

MSQ_GENERATORS: dict[str, Callable[..., Any]] = {
    "equiv": _equiv_msq, "taut": _taut_msq, "truth": _taut_msq, "quant": _quant_msq, "argument": _arg_msq, "relprop": _relprop_msq, "poset": _poset_msq,
    "lattice": _poset_msq, "nullspace": _nullspace_msq, "basis": _basis_msq, "eig": _eig_msq, "posdef": _posdef_msq,
}
MSQ_KINDS = tuple(MSQ_GENERATORS)
MCQ_KINDS = tuple(VALUE_GENERATORS) + tuple(CHOICE_GENERATORS)


def generate(kind: str, rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    return GENERATORS[kind](rng, marks, difficulty, topic)


def gen_nat(kind: str, rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    blocks, val, ans, _traps = VALUE_GENERATORS[kind](rng, difficulty)
    tol = 0.01 if isinstance(val, float) and val != int(val) else 0
    return {"kind": "nat", "topic": topic, "marks": marks, "blocks": blocks, "value": val, "tolerance": tol,
            "answer": ans + [{"p": f"Answer: {val:g}" + (f" (accepted range {val - tol:.2f} to {val + tol:.2f})" if tol else "")}]}


def gen_mcq_numeric(kind: str, rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any]:
    """A 4-option MCQ whose key is computed. Distractors are the answers of typical mistakes."""
    if kind in CHOICE_GENERATORS and (kind not in VALUE_GENERATORS or rng.random() < 0.5):
        blocks, options, correct, ans = CHOICE_GENERATORS[kind](rng, difficulty)
        return {"kind": "mcq", "topic": topic, "marks": marks, "blocks": blocks, "options": options, "correct": correct, "answer": ans}
    blocks, val, ans, traps = VALUE_GENERATORS[kind](rng, difficulty)
    isint = float(val) == int(val) and all(float(t) == int(t) for t in traps if t is not None)
    opts = {val}
    traps = [t for t in traps if t is not None and t != val]
    rng.shuffle(traps)
    for t in traps:
        if len(opts) < 4:
            opts.add(t)
    while len(opts) < 4:
        d = rng.choice([-3, -2, -1, 1, 2, 3]) if isint else round(rng.choice([-1, 1]) * rng.uniform(0.4, 3.0), 2)
        opts.add(int(val) + d if isint else round(val + d, 2))
    options = sorted(opts)
    fmt = (lambda x: str(int(x))) if isint else (lambda x: f"{float(x):.2f}")
    blocks = [{"p": blocks[0]["p"].replace(" is ______ (up to 2 decimals).", ":").replace(" is ______.", ":").replace(" equals ______.", ":").replace(" = ______.", ":")}] + blocks[1:]
    return {"kind": "mcq", "topic": topic, "marks": marks, "blocks": blocks, "options": [fmt(o) for o in options], "correct": options.index(val), "answer": ans}


def gen_msq_numeric(kind: str, rng: random.Random, marks: int, difficulty: str, topic: str) -> dict[str, Any] | None:
    if kind not in MSQ_GENERATORS:
        return None
    for _ in range(12):
        got = MSQ_GENERATORS[kind](rng, difficulty)
        if got:
            blocks, options, correct, ans = got
            return {"kind": "msq", "topic": topic, "marks": marks, "blocks": blocks, "options": options, "correct": correct, "answer": ans}
    return None
