"""IIIT Sri City DSMA previous question papers -> src/indiclearn/studio/courses/dsma_pyq.json (questions + worked answers).

    python training/build_dsma_pyq.py

Questions are typed from the institute's papers (2022 mid/end, 2024 quizzes, 2025 mid). Every computed answer (truth tables,
counts, recurrences, matrices) is produced here with the app's exact solvers / sympy, so it cannot be mistyped; proofs and
explanations are written by hand. Where a printed paper has a mistake, the answer says so and gives the correct result.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indiclearn.studio.courses import dsmacalc as D  # noqa: E402
from indiclearn.studio.courses.dsmacalc import BIN, NOT, V, mtex, tex, vtex  # noqa: E402

OUT = ROOT / "src" / "indiclearn" / "studio" / "courses" / "dsma_pyq.json"
p, q, r = V("p"), V("q"), V("r")
imp = lambda a, b: BIN("imp", a, b)  # noqa: E731
AND = lambda a, b: BIN("and", a, b)  # noqa: E731
OR = lambda a, b: BIN("or", a, b)  # noqa: E731
n = sp.Symbol("n", integer=True)


def P(t: str) -> dict:
    return {"p": t}


def M(t: str) -> dict:
    return {"math": t}


def Q(num: str, text: str, marks: int, topic: str, answer: list, qtype: str | None = None, **kw) -> dict:
    qt = qtype or ("short" if marks <= 2 else "medium" if marks <= 4 else "long")
    return {"n": num, "q": text, "marks": marks, "topic": topic, "qtype": qt, "answer_blocks": answer, **kw}


def taut_part(label: str, f: tuple) -> list:
    vs = D.vars_of(f)
    c = D.classify(f, vs)
    return [P(f"**({label})** $ {tex(f)} $"), {"table": D.truth_table(f, vs, [f])}, P(f"The last column is {'all T' if c == 'tautology' else 'all F' if c == 'contradiction' else 'mixed'}: **{c}**.")]


def mcq(num: str, text: str, options: list[str], correct: int | list[int], why: str, topic: str, note: str = "") -> dict:
    multi = isinstance(correct, list)
    d = Q(num, text, 1, topic, [P(why)] + ([P("⚠ " + note)] if note else []), qtype="msq" if multi else "mcq", options=options,
          correct=None if correct == -1 else correct)
    if correct == -1:
        d["key_note"] = "none of the printed options is correct"
    if note:
        d["paper_issue"] = note
    return d


papers: list[dict] = []

# ------------------------------------------------------------------------------------------- Mid-Sem Monsoon 2025
f1 = imp(AND(AND(OR(p, q), imp(p, r)), imp(q, r)), r)
assert D.classify(f1) == "tautology"
y = sp.Function("y")
y13 = sp.simplify(sp.rsolve(y(n) - 9 * y(n - 1) + 26 * y(n - 2) - 24 * y(n - 3) - 4**n, y(n), {y(0): 0, y(1): 1, y(2): 10}))
seq = [0, 1, 10]
for k in range(3, 7):
    seq.append(9 * seq[-1] - 26 * seq[-2] + 24 * seq[-3] + 4**k)
assert all(sp.simplify(y13.subs(n, k) - seq[k]) == 0 for k in range(7))
a11 = 1 + 3 * sp.summation(sp.Symbol("k") ** 2, (sp.Symbol("k"), 1, n))
papers.append({
    "id": "mid-2025", "title": "Mid-Sem Exam, Monsoon 2025", "exam": "mid", "year": 2025, "date": "08-10-2025", "minutes": 90, "marks": 25,
    "note": "UG1 (CSE/ECE/AI&DS), 14 questions.",
    "questions": [
        Q("1", "Simplify the compound proposition step by step to an equivalent expression containing no implications or negations of conjunctions/disjunctions: $((p \\vee q) \\wedge (p \\rightarrow r) \\wedge (q \\rightarrow r)) \\rightarrow r$. Then decide whether it is a tautology, contradiction or contingency, stating the law used at each step.", 1, "equivalence", [
            P("$(p \\rightarrow r) \\wedge (q \\rightarrow r) \\equiv (\\neg p \\vee r) \\wedge (\\neg q \\vee r) \\equiv (\\neg p \\wedge \\neg q) \\vee r$ (implication law, distributive law)."),
            P("So the whole formula is $\\neg[(p \\vee q) \\wedge ((\\neg p \\wedge \\neg q) \\vee r)] \\vee r$ (implication law)."),
            P("$(p \\vee q) \\wedge ((\\neg p \\wedge \\neg q) \\vee r) \\equiv [(p \\vee q) \\wedge \\neg(p \\vee q)] \\vee [(p \\vee q) \\wedge r] \\equiv (p \\vee q) \\wedge r$ (distributive, De Morgan, negation law $F \\vee x \\equiv x$)."),
            P("Negating (De Morgan): $\\neg(p \\vee q) \\vee \\neg r \\equiv (\\neg p \\wedge \\neg q) \\vee \\neg r$. Adding $\\vee\\, r$: $(\\neg p \\wedge \\neg q) \\vee \\neg r \\vee r \\equiv (\\neg p \\wedge \\neg q) \\vee T \\equiv T$ (negation law, domination law)."),
            P("Final expression: $T$ - the formula is a **tautology** (it is the rule of proof by cases / constructive dilemma)."),
            {"table": D.truth_table(f1, ["p", "q", "r"], [f1])}]),
        Q("2", "Universe: all real numbers. $P(x): x^2 > 4$, $Q(x): x > 2$, $R(x): x < -2$. Express in English, simplify, and decide true or false: (I) $\\forall x[P(x) \\rightarrow (Q(x) \\vee R(x))]$; (II) $\\exists x[P(x) \\wedge \\neg(Q(x) \\vee R(x))]$.", 1, "predicates", [
            P("$x^2 > 4 \\iff x > 2$ or $x < -2$, i.e. $P(x) \\equiv Q(x) \\vee R(x)$ for every real $x$."),
            P("(I) \"Every real number whose square exceeds 4 is greater than 2 or less than $-2$.\" **True** - it is exactly the equivalence above."),
            P("(II) \"Some real number has square greater than 4 but lies between $-2$ and $2$.\" $P(x) \\wedge \\neg(Q(x) \\vee R(x)) \\equiv P(x) \\wedge \\neg P(x)$, a contradiction for every $x$, so (II) is **false**. (II) is exactly the negation of (I).")]),
        mcq("3", "Which of the following statements are logically equivalent?",
            ["If it rains, then the ground gets wet.", "If the ground gets wet, then it rains.", "If the ground does not get wet, then it does not rain.",
             "It rains and the ground gets wet.", "Either it does not rain or the ground gets wet."], [0, 2, 4], "Let $p$: it rains, $q$: the ground gets wet. (a) $p \\rightarrow q$; (b) $q \\rightarrow p$ (converse); (c) $\\neg q \\rightarrow \\neg p$ (contrapositive); (d) $p \\wedge q$; (e) $\\neg p \\vee q$. Since $p \\rightarrow q \\equiv \\neg q \\rightarrow \\neg p \\equiv \\neg p \\vee q$, statements **(a), (c) and (e)** are equivalent. (b) and (d) are not.", "conditionals"),
        Q("4", "Express using predicates, quantifiers and logical connectives: (i) \"No cats are dogs.\" (ii) \"Every student in DSMA class has studied Math I.\" (domain: all students of IIIT Sri City) (iii) \"A negative real number does not have a square root that is a real number.\" (iv) \"Every insurance agent who sells a policy to at least one person who is not insured refuses to sell policies to anyone who is dishonest.\"", 2, "predicates", [
            P("(i) $C(x)$: $x$ is a cat, $D(x)$: $x$ is a dog. $\\forall x\\,(C(x) \\rightarrow \\neg D(x))$, equivalently $\\neg \\exists x\\,(C(x) \\wedge D(x))$."),
            P("(ii) $S(x)$: $x$ is in the DSMA class, $M(x)$: $x$ has studied Math I. $\\forall x\\,(S(x) \\rightarrow M(x))$ - note $\\rightarrow$, not $\\wedge$, because the domain is all students."),
            P("(iii) Domain: real numbers. $\\forall x\\,(x < 0 \\rightarrow \\neg \\exists y\\,(y^2 = x))$, equivalently $\\forall x\\,\\forall y\\,(x < 0 \\rightarrow y^2 \\ne x)$."),
            P("(iv) $A(x)$: $x$ is an insurance agent, $S(x,y)$: $x$ sells a policy to $y$, $I(y)$: $y$ is insured, $H(z)$: $z$ is dishonest, $R(x,z)$: $x$ refuses to sell policies to $z$. $\\forall x\\,[(A(x) \\wedge \\exists y\\,(S(x,y) \\wedge \\neg I(y))) \\rightarrow \\forall z\\,(H(z) \\rightarrow R(x,z))]$.")]),
        Q("5", "The $n$th statement in a list of 129 statements is \"Exactly $n$ of the statements in this list are true.\" What conclusions can you draw?", 1, "propositions", [
            P("Two different statements of the list cannot both be true (they claim different numbers), so **at most one statement is true**."),
            P("If exactly one is true, the number of true statements is 1, so it must be statement 1 (\"exactly 1 is true\") - and then statements 2 to 129 are false, which is consistent."),
            P("If none is true, statement 1 is false (there are 0, not 1, true statements) and so are all the others - also consistent."),
            P("Conclusion: statements 2, 3, ..., 129 are all **false**; statement 1 may be true or false, and no other statement can be true. (In Rosen's version, \"exactly $n$ are **false**\" with 100 statements, only statement 99 is true.)")]),
        Q("6", "Prove or disprove: (i) Is $\\forall x \\exists y \\forall z\\, P(x,y,z)$ equivalent to $\\exists y \\forall x \\forall z\\, P(x,y,z)$? (ii) Is $\\exists x \\forall y \\forall z\\, Q(x,y,z)$ equivalent to $\\forall y \\exists x \\exists z\\, Q(x,y,z)$?", 2, "predicates", [
            P("(i) **Not equivalent.** In $\\forall x \\exists y$ the $y$ may depend on $x$; $\\exists y \\forall x$ needs one $y$ for all $x$. Counterexample: domain $\\{1, 2\\}$, $P(x,y,z): x = y$. Then $\\forall x \\exists y \\forall z\\, P$ is true ($y = x$), but $\\exists y \\forall x \\forall z\\, P$ is false (no $y$ equals both 1 and 2). (The second implies the first.)"),
            P("(ii) **Not equivalent.** $\\exists x \\forall y \\forall z\\, Q \\Rightarrow \\forall y \\exists x \\exists z\\, Q$ always (keep the same $x$, pick any $z$), but not conversely. Counterexample: domain $\\{1, 2\\}$, $Q(x,y,z): x = y$. $\\forall y \\exists x \\exists z\\, Q$ is true ($x = y$), $\\exists x \\forall y \\forall z\\, Q$ is false.")]),
        Q("7", "Determine whether this argument is valid. SRI is an AI created to protect humanity. If SRI were both rational and ethical, she would prevent disasters. If SRI were not rational, she would be malfunctioning; if she were not ethical, she would be dangerous. However, SRI does not prevent disasters. If SRI exists, she is neither malfunctioning nor dangerous. Therefore, SRI does not exist.", 2, "inference", [
            P("Let $E$: SRI exists, $R$: rational, $T$: ethical, $P$: prevents disasters, $M$: malfunctioning, $D$: dangerous. Premises: $(R \\wedge T) \\rightarrow P$, $\\neg R \\rightarrow M$, $\\neg T \\rightarrow D$, $\\neg P$, $E \\rightarrow (\\neg M \\wedge \\neg D)$. Conclusion: $\\neg E$."),
            P("1. $\\neg P$ and $(R \\wedge T) \\rightarrow P$ give $\\neg(R \\wedge T)$, i.e. $\\neg R \\vee \\neg T$ (modus tollens, De Morgan)."),
            P("2. From $\\neg R \\rightarrow M$ and $\\neg T \\rightarrow D$ with $\\neg R \\vee \\neg T$: $M \\vee D$ (constructive dilemma)."),
            P("3. $M \\vee D \\equiv \\neg(\\neg M \\wedge \\neg D)$ (De Morgan). With $E \\rightarrow (\\neg M \\wedge \\neg D)$, modus tollens gives $\\neg E$."),
            P("The argument is **valid**: SRI does not exist.")]),
        Q("8", "If $3n + 2 \\equiv 0 \\pmod 4$, then $n \\equiv 2 \\pmod 4$. Prove this by contradiction. [$a \\equiv b \\pmod n$ means $n \\mid (a - b)$.]", 1, "proofs", [
            P("Suppose $3n + 2 \\equiv 0 \\pmod 4$ but $n \\not\\equiv 2 \\pmod 4$. Then $n \\equiv 0, 1$ or $3 \\pmod 4$."),
            P("$n \\equiv 0$: $3n + 2 \\equiv 2$; $n \\equiv 1$: $3n + 2 \\equiv 5 \\equiv 1$; $n \\equiv 3$: $3n + 2 \\equiv 11 \\equiv 3 \\pmod 4$."),
            P("In every case $3n + 2 \\not\\equiv 0 \\pmod 4$, contradicting the hypothesis. Hence $n \\equiv 2 \\pmod 4$. (Check: $n = 2$ gives $3n + 2 = 8 \\equiv 0$.)")]),
        Q("9", "$S$ is the set of integers defined by: basis step $4 \\in S$; recursive step: if $n \\in S$ then $n + 3 \\in S$ and $n^2 \\in S$. Use structural induction to prove that every $n \\in S$ satisfies $n \\equiv 1 \\pmod 3$.", 2, "recursion", [
            P("**Basis:** $4 = 3 \\cdot 1 + 1$, so $4 \\equiv 1 \\pmod 3$."),
            P("**Recursive step:** assume $n \\in S$ with $n \\equiv 1 \\pmod 3$, i.e. $n = 3k + 1$."),
            P("$n + 3 = 3(k + 1) + 1 \\equiv 1 \\pmod 3$, and $n^2 = 9k^2 + 6k + 1 = 3(3k^2 + 2k) + 1 \\equiv 1 \\pmod 3$."),
            P("Both elements built from $n$ keep the property, so by structural induction every element of $S$ is $\\equiv 1 \\pmod 3$.")]),
        Q("10", "Use mathematical induction to show that $\\dfrac{1 \\cdot 3 \\cdot 5 \\cdots (2n-1)}{2 \\cdot 4 \\cdot 6 \\cdots (2n)} \\le \\dfrac{1}{\\sqrt{n+1}}$ for $n \\ge 1$.", 2, "induction", [
            P("Let $P_n$ be the left side. **Basis** $n = 1$: $P_1 = \\frac{1}{2} \\le \\frac{1}{\\sqrt 2} \\approx 0.707$. True."),
            P("**Inductive step:** assume $P_k \\le \\frac{1}{\\sqrt{k+1}}$. Then $P_{k+1} = P_k \\cdot \\frac{2k+1}{2k+2} \\le \\frac{2k+1}{(2k+2)\\sqrt{k+1}}$."),
            P("It suffices that $\\frac{2k+1}{(2k+2)\\sqrt{k+1}} \\le \\frac{1}{\\sqrt{k+2}}$, i.e. $(2k+1)^2(k+2) \\le (2k+2)^2(k+1)$."),
            P("Expanding: $4k^3 + 12k^2 + 9k + 2 \\le 4k^3 + 12k^2 + 12k + 4$, i.e. $0 \\le 3k + 2$ - true for $k \\ge 1$. Hence $P_{k+1} \\le \\frac{1}{\\sqrt{k+2}}$ and the result holds for all $n \\ge 1$.")]),
        Q("11", "Let $a_0 = 1$ and $a_n = a_{n-1} + 3n^2$ for $n \\ge 1$. Find a formula for $a_n$ by iteration and prove it by mathematical induction.", 3, "recurrence-model", [
            P("Iterating: $a_n = a_{n-1} + 3n^2 = a_{n-2} + 3(n-1)^2 + 3n^2 = \\cdots = a_0 + 3(1^2 + 2^2 + \\cdots + n^2)$."),
            P(f"So $a_n = 1 + 3 \\cdot \\frac{{n(n+1)(2n+1)}}{{6}} = 1 + \\frac{{n(n+1)(2n+1)}}{{2}} = {sp.latex(sp.factor(a11))}$."),
            P("Check: $a_1 = 1 + 3 = 4$ and the formula gives $1 + \\frac{1 \\cdot 2 \\cdot 3}{2} = 4$; $a_2 = 4 + 12 = 16 = 1 + \\frac{2 \\cdot 3 \\cdot 5}{2}$."),
            P("**Induction:** basis $a_0 = 1 = 1 + 0$. If $a_k = 1 + \\frac{k(k+1)(2k+1)}{2}$, then $a_{k+1} = 1 + \\frac{k(k+1)(2k+1)}{2} + 3(k+1)^2 = 1 + \\frac{(k+1)[k(2k+1) + 6(k+1)]}{2} = 1 + \\frac{(k+1)(2k^2 + 7k + 6)}{2} = 1 + \\frac{(k+1)(k+2)(2k+3)}{2}$, which is the formula for $n = k + 1$.")]),
        Q("12", "Which of the following sets are uncountable? Justify. (1) The set of all functions from $\\mathbb{R}$ to $\\{0, 1\\}$. (2) The set of all functions from $\\mathbb{N}$ to $\\{0, 1\\}$. (3) The set of all finite subsets of $\\mathbb{N}$.", 3, "cardinality", [
            P("(2) **Uncountable.** A function $f: \\mathbb{N} \\to \\{0,1\\}$ is an infinite 0-1 sequence (equivalently a subset of $\\mathbb{N}$). If they were listed $f_1, f_2, \\dots$, the diagonal function $g(n) = 1 - f_n(n)$ differs from every $f_n$ at $n$ - Cantor's diagonal argument. So $|\\{0,1\\}^{\\mathbb{N}}| = |P(\\mathbb{N})| > |\\mathbb{N}|$."),
            P("(1) **Uncountable.** Restricting to $\\mathbb{N} \\subset \\mathbb{R}$ maps it onto the set in (2) (each $g: \\mathbb{N} \\to \\{0,1\\}$ extends, e.g. by 0 elsewhere), so it is at least as large as an uncountable set (in fact $|P(\\mathbb{R})| > |\\mathbb{R}|$)."),
            P("(3) **Countable.** Finite subsets of size $k$ are countable (finite tuples of naturals), and the set of all finite subsets is a countable union over $k = 0, 1, 2, \\dots$ of countable sets. Explicitly, $\\{a_1, \\dots, a_k\\} \\mapsto 2^{a_1} + \\cdots + 2^{a_k}$ is a bijection onto $\\mathbb{N}$ (binary representation).")]),
        Q("13", "Solve $y_n = 9y_{n-1} - 26y_{n-2} + 24y_{n-3} + 4^n$ with $y_0 = 0$, $y_1 = 1$, $y_2 = 10$.", 3, "recurrence-nonhomog", [
            P("Characteristic equation: $r^3 - 9r^2 + 26r - 24 = (r - 2)(r - 3)(r - 4) = 0$, roots $2, 3, 4$."),
            P("$F(n) = 4^n$ and 4 is a root of multiplicity 1, so try $a_n^{(p)} = A\\,n\\,4^n$. Substituting gives $A\\,4^n[n - \\tfrac{9}{4}(n-1) + \\tfrac{26}{16}(n-2) - \\tfrac{24}{64}(n-3)] = 4^n$, so $A \\cdot \\tfrac{1}{8} = 1$, $A = 8$."),
            P("General solution: $y_n = c_1 2^n + c_2 3^n + c_3 4^n + 8n\\,4^n$. From $y_0 = 0, y_1 = 1, y_2 = 10$: $c_1 = -\\frac{29}{2}$, $c_2 = 60$, $c_3 = -\\frac{91}{2}$."),
            P(f"$y_n = 8n\\,4^n - \\frac{{29}}{{2}}\\,2^n + 60 \\cdot 3^n - \\frac{{91}}{{2}}\\,4^n$. Check: $y_3 = {seq[3]}$, $y_4 = {seq[4]}$ from the recurrence agree with the formula.")]),
        Q("14", "Define a bijective function from $(0, 1)$ to $\\mathbb{R}$.", 1, "cardinality", [
            P("$f(x) = \\tan\\left(\\pi\\left(x - \\tfrac{1}{2}\\right)\\right)$: $x \\mapsto \\pi(x - \\frac12)$ is a bijection $(0,1) \\to (-\\frac{\\pi}{2}, \\frac{\\pi}{2})$ and $\\tan$ is a strictly increasing bijection from there onto $\\mathbb{R}$."),
            P("Another one: $f(x) = \\ln\\frac{x}{1 - x}$, with inverse $f^{-1}(y) = \\frac{1}{1 + e^{-y}}$.")]),
    ]})


# ------------------------------------------------------------------------------------------- Quiz 1, Oct 2024
papers.append({
    "id": "quiz1-2024", "title": "Scheduled Quiz 1, October 2024", "exam": "quiz", "year": 2024, "date": "03-10-2024", "minutes": 20, "marks": 15,
    "note": "15 MCQs, 1 mark each (no key was given; the answers below are worked out and checked).",
    "questions": [
        mcq("1", "Sets $A$ and $B$ have $|A| = |B| = n$ and $A \\cap B = \\emptyset$. What is $|A \\cup B|$?", ["$n$", "$2n$", "$n^2$", "$n(n-1)$"], 1,
            "$|A \\cup B| = |A| + |B| - |A \\cap B| = n + n - 0 = 2n$.", "sets"),
        mcq("2", "If $f: A \\to B$ and $g: B \\to C$ are both injective, which is necessarily true about $g \\circ f$?", ["$g \\circ f$ is injective but not surjective", "$g \\circ f$ is surjective but not injective", "$g \\circ f$ is both injective and surjective", "$g \\circ f$ is injective"], 3,
            "$g(f(x)) = g(f(y)) \\Rightarrow f(x) = f(y)$ ($g$ one-to-one) $\\Rightarrow x = y$ ($f$ one-to-one). Surjectivity is not guaranteed either way, so only (d) is necessarily true.", "functions"),
        mcq("3", "Which of the following is a characteristic of a countable set?", ["Every subset of a countable set is finite", "A countable set must be finite", "A countable set is either finite or has a bijection with $\\mathbb{N}$", "A power set of a countable set is countable"], 2,
            "Countable = finite or countably infinite (in bijection with $\\mathbb{N}$). $\\mathbb{N}$ itself is an infinite countable subset of $\\mathbb{N}$, and $P(\\mathbb{N})$ is uncountable.", "cardinality"),
        mcq("4", "$f(0) = 1$ and $f(n) = 3f(n-1) + 1$. What is $f(n)$ in terms of $n$?", ["$f(n) = 3^n + n$", "$f(n) = 3^{n+1} - 2$", "$f(n) = 3n + 1$", "$f(n) = 3^n - 1$"], -1,
            "$f(0..4) = 1, 4, 13, 40, 121$. Iterating, $f(n) = 3^n + 3^{n-1} + \\cdots + 1 = \\dfrac{3^{n+1} - 1}{2}$.", "recursion",
            "None of the printed options is correct: (a) gives $f(2) = 11$, (b) $f(0) = 1$ but $f(1) = 7$, (c) $f(2) = 7$, (d) $f(0) = 0$. The correct answer is $f(n) = \\frac{3^{n+1} - 1}{2}$."),
        mcq("5", "A set $S$ is defined recursively by $1 \\in S$; if $x \\in S$ then $x + 1 \\in S$. Which is true about $S$?", ["$S$ contains only even numbers", "$S = \\mathbb{N}$, the set of natural numbers", "$S$ is finite", "$S$ contains only powers of 2"], 1,
            "Starting from 1 and adding 1 repeatedly generates $\\{1, 2, 3, \\dots\\}$ - the natural numbers (taking $\\mathbb{N}$ to start at 1).", "recursion"),
        mcq("6", "Which statement about recursive algorithms is FALSE?", ["A recursive algorithm must always have a base case", "A recursive algorithm can always be transformed into an iterative algorithm", "Recursive algorithms can solve problems where the problem size is reduced at each step", "Recursive algorithms are always faster than their iterative counterparts"], 3,
            "Recursion is not always faster - e.g. naive recursive Fibonacci is exponential while the loop is linear, and calls add overhead.", "recursion"),
        mcq("7", "In a proof by strong induction, which step is necessary but not in a standard proof by induction?", ["Prove the base case for $n = 1$", "Assume the statement is true for $n = k$", "Assume the statement is true for all $k \\le n$ when proving for $n + 1$", "Prove that $P(k+1)$ follows from $P(k)$"], 2,
            "Strong induction's hypothesis is that $P(1), \\dots, P(n)$ all hold when proving $P(n+1)$.", "induction"),
        mcq("8", "Which statement about structural induction is true?", ["It is commonly used to prove sequences of real numbers", "It is a variation of mathematical induction that applies to recursively defined structures", "It requires proving a property for all elements less than $n$", "It can only be applied to finite sets of numbers"], 1,
            "Structural induction proves a property for the basis elements and shows the recursive rules preserve it - for recursively defined sets, strings, trees, formulas.", "recursion"),
        mcq("9", "Let $p, q$ be propositional variables. Which is a tautology?", ["$p \\wedge q$", "$p \\rightarrow q$", "$p \\vee q$", "$(p \\vee q) \\vee \\neg q$"], 3,
            "$(p \\vee q) \\vee \\neg q \\equiv p \\vee (q \\vee \\neg q) \\equiv p \\vee T \\equiv T$. The others are false for some row.", "tautology"),
        mcq("10", "When do $p \\wedge q$ and $p \\vee q$ have the same truth value?", ["$p$ and $q$ have the same truth value", "$p$ and $q$ have different truth values", "$p$ is the negation of $q$", "none of the above"], 0,
            "TT: both T; FF: both F; TF and FT: $\\wedge$ is F but $\\vee$ is T. So exactly when $p$ and $q$ agree.", "propositions"),
        mcq("11", "The premises $(p \\wedge q) \\vee r$ and $r \\rightarrow s$ imply which conclusion?", ["$p \\vee r$", "$q \\vee s$", "$q \\vee r$", "$p \\vee q$"], [0, 1, 2],
            "$(p \\wedge q) \\vee r \\equiv (p \\vee r) \\wedge (q \\vee r)$, so $p \\vee r$ and $q \\vee r$ both follow (simplification). From $q \\vee r$ and $r \\rightarrow s$ (i.e. $\\neg r \\vee s$), resolution gives $q \\vee s$. $p \\vee q$ does not follow ($p = q = F$, $r = s = T$).", "inference",
            "Three options are valid conclusions; the intended single answer is most likely (b) $q \\vee s$ (resolution). Checked with a full truth table."),
        mcq("12", "The inverse of a conditional statement is the ______ of its converse.", ["negation", "equivalence", "contrapositive", "none of the above"], 2,
            "For $p \\rightarrow q$: converse $q \\rightarrow p$, whose contrapositive is $\\neg p \\rightarrow \\neg q$ - exactly the inverse.", "conditionals"),
        mcq("13", "$\\vee$ has precedence over", ["$\\wedge$", "$\\neg$", "$\\Rightarrow$", "none of the above"], 2,
            "Precedence: $\\neg$, then $\\wedge$, then $\\vee$, then $\\rightarrow$, then $\\leftrightarrow$. So $\\vee$ binds tighter only than $\\rightarrow$ (and $\\leftrightarrow$).", "propositions"),
        mcq("14", "Which of the following can only be used in disproving statements?", ["Direct proof", "Contrapositive proofs", "Counter example", "Mathematical induction"], 2,
            "A counterexample shows a universal statement is false; it cannot prove one.", "proofs"),
        mcq("15", "Express \"The difference of a real number and itself is zero\".", ["$\\forall x \\exists y (x - y = 0)$", "$\\forall x \\forall y (x - y = 0)$", "$\\exists x (x - x = 0)$", "$\\forall x (x - x = 0)$"], 3,
            "It is about every real number and that same number: $\\forall x\\,(x - x = 0)$.", "predicates"),
    ]})

# ------------------------------------------------------------------------------------------- Quiz 2, Nov 2024 (answer key given; checked)
papers.append({
    "id": "quiz2-2024", "title": "Scheduled Quiz 2, November 2024", "exam": "quiz", "year": 2024, "date": "11-11-2024", "minutes": 20, "marks": 15,
    "note": "15 MCQs with the official key, each re-checked (one printed question has a typo - see Q1).",
    "questions": [
        mcq("1", "$x_1 + x_2 + x_3 + x_4 + x_5 = 21$ with nonnegative integers, $0 \\le x_1 \\le 3$, $1 \\le x_2 \\le 4$ and $x_3 \\ge 15$. How many solutions are there?", ["80", "100", "108", "106"], 3,
            "Substitute $x_2' = x_2 - 1$, $x_3' = x_3 - 15$: $x_1 + x_2' + x_3' + x_4 + x_5 = 5$ with $x_1 \\le 3$ and $x_2' \\le 2$ (for $x_2 < 4$). Count $\\binom{9}{4} = 126$, minus $x_1 \\ge 4$: $\\binom{5}{4} = 5$, minus $x_2' \\ge 3$: $\\binom{6}{4} = 15$, plus both: 0. $126 - 5 - 15 = 106$.", "incl-excl",
            "As printed ($1 \\le x_2 \\le 4$, so $x_2' \\le 3$) the count is $126 - 5 - \\binom{5}{4} = 116$ - not among the options. The key (d) 106 is for $1 \\le x_2 < 4$, as in Rosen. Both counts verified by brute force."),
        mcq("2", "How many positive integers not exceeding 1000 are divisible by 7 or 11?", ["220", "240", "100", "200"], 0,
            "$\\lfloor 1000/7 \\rfloor + \\lfloor 1000/11 \\rfloor - \\lfloor 1000/77 \\rfloor = 142 + 90 - 12 = 220$.", "incl-excl"),
        mcq("3", "$R$ and $S$ are reflexive relations on a set $A$. The relation $R \\cap S$ is", ["antisymmetric", "reflexive", "irreflexive", "symmetric", "none of the above"], 1,
            "Every $(a, a)$ is in $R$ and in $S$, hence in $R \\cap S$.", "relations"),
        mcq("4", "$R = \\{(0,0), (0,1), (1,1), (2,2)\\}$ on $\\{0, 1, 2\\}$, and $P$ is the property \"is not reflexive\". The closure of $R$ with respect to $P$", ["exists", "does not exist", "has an odd number of elements", "is the relation $R$ itself"], 1,
            "$R$ already contains $(0,0), (1,1), (2,2)$, so $R$ and every relation containing it is reflexive. No relation containing $R$ has property $P$, so the closure does not exist.", "closures"),
        mcq("5", "Which collection of subsets is a partition of $\\{1, 2, 3, 4, 5, 6\\}$?", ["$\\{\\{1,2\\},\\{2,3,4\\},\\{4,5,6\\}\\}$", "$\\{\\{1\\},\\{2,3,6\\},\\{4\\},\\{5\\}\\}$", "$\\{\\{2,4\\},\\{1,5\\}\\}$", "$\\{\\{1,4,5\\},\\{2,6\\}\\}$"], 1,
            "(b): nonempty, pairwise disjoint, union is the whole set. (a) overlaps (2 and 4 twice); (c) misses 3 and 6; (d) misses 3.", "equiv-rel"),
        mcq("6", "Which of these is a poset?", ["$(\\mathbb{Z}, \\not\\geq)$", "$(\\mathbb{Z}, \\ne)$", "$(\\mathbb{Z}, \\ge)$", "$(\\mathbb{Z}, \\nmid)$"], 2,
            "$\\ge$ is reflexive, antisymmetric and transitive. $\\not\\ge$ (i.e. $<$) and $\\ne$ are not reflexive; $\\nmid$ is not reflexive either.", "posets"),
        mcq("7", "Which statement is correct about the least upper bound (supremum) in a poset?", ["It is the smallest element in the set", "It is the maximum element in the set", "It is the least element that is greater than or equal to all elements in a subset", "It is always equal to the greatest lower bound"], 2,
            "The LUB of a subset is the least of its upper bounds.", "extremal"),
        mcq("8", "In a well-ordered set, which statement is true?", ["Every non-empty subset has a least element", "Every non-empty subset has a maximum element", "Every non-empty subset has a greatest lower bound", "Every element has an immediate successor"], 0,
            "That is the definition of a well-ordered set (e.g. $\\mathbb{N}$ with $\\le$).", "posets"),
        mcq("9", "Find the minimum number of students in a class to be sure that three of them are born in the same month.", ["13", "24", "25", "37"], 2,
            "Generalized pigeonhole: $12(3 - 1) + 1 = 25$.", "pigeonhole"),
        mcq("10", "Every person at a party shakes hands with every other person. If there were 91 handshakes, how many persons were present?", ["14", "15", "16", "17"], 0,
            "$\\binom{n}{2} = 91 \\Rightarrow n(n-1) = 182 \\Rightarrow n = 14$.", "perm-comb"),
        mcq("11", "An urn has 40 balls: 16 white, 10 red, 14 black. What is the minimum number drawn blindfolded (no replacement) to be sure of at least one ball of every colour?", ["21", "25", "27", "31"], 3,
            "Worst case: all 16 white and all 14 black first (30 balls), then the next ball must be red: $30 + 1 = 31$.", "pigeonhole"),
        mcq("12", "The recurrence relation $a_n = a_{n-2} + a_{n-5} + 5n$ is not", ["homogeneous", "linear", "having constant coefficients", "none of the above"], 0,
            "It is linear with constant coefficients, but the term $5n$ makes it non-homogeneous.", "recurrence-model"),
        mcq("13", "The solution of $a_n = a_{n-1} + 2$ with $a_0 = 2$ is", ["$2(2n + 1)$", "$3n^2$", "$2(1 + n)$", "$5(n + 1)/2$"], 2,
            "$a_n = a_0 + 2n = 2 + 2n = 2(1 + n)$.", "recurrence-homog"),
        mcq("14", "$a_k = \\dfrac{a_{k-1}}{1 + a_{k-1}}$ for $k \\ge 1$ and $a_0 = 1$. An explicit formula for $a_k$ is", ["$1/2^k$", "$1/(3^{k+1} - 2)$", "$1/(k+1)$", "$2/(k+2)$"], 2,
            "$a_1 = \\frac12, a_2 = \\frac13, \\dots$ In fact $\\frac{1}{a_k} = \\frac{1}{a_{k-1}} + 1$, so $\\frac{1}{a_k} = k + 1$.", "recurrence-model"),
        mcq("15", "On $\\mathbb{N} \\times \\mathbb{N}$ (nonzero naturals) define $(a,b)\\,R\\,(x,y)$ iff $\\gcd(a,b) = \\gcd(x,y)$. The relation $R$ is", ["Reflexive, Symmetric, Not Transitive", "Not Reflexive, Not Symmetric, Not Transitive", "Reflexive, Not Symmetric, Not Transitive", "Reflexive, Symmetric, Transitive"], 3,
            "\"Same value of a function\" is always an equivalence relation: equality of gcds is reflexive, symmetric and transitive.", "equiv-rel"),
    ]})

# ------------------------------------------------------------------------------------------- Mid 2022 (Group 6, Group 1)
def mid2022(group: int) -> dict:
    g6 = group == 6
    forms = ([AND(p, q), p], [p, OR(p, p)], [NOT(p), imp(p, q)], [AND(p, q), imp(p, q)], [NOT(imp(p, q)), p], [NOT(imp(p, q)), NOT(q)], [q, imp(p, q)]) if g6 else \
        ([NOT(AND(p, q)), p], [p, OR(p, q)], [p, imp(p, q)], [AND(p, q), imp(p, q)], [imp(p, q), p], [NOT(imp(p, q)), NOT(q)], [q, NOT(imp(p, q))])
    fs = [imp(a, b) for a, b in forms]
    q1_ans = sum((taut_part("abcdefg"[i], f) for i, f in enumerate(fs)), [])
    names = ("Prakash plays cricket", "Qadir plays football", "Ratan plays football", "Sonu plays football") if g6 else ("A reads", "B sleeps", "C sleeps", "D sleeps")
    arg_text = ("If Prakash plays Cricket, then either Qadir or Ratan will play Football. If Qadir plays Football, then Prakash will not play Cricket. If Sonu plays Football, then Ratan will not play Football. Prakash plays Cricket. Therefore, Sonu will not play Football." if g6 else
                "If A reads, then either B or C will sleep. If B sleeps, then A will not read. If D sleeps, then C will not sleep. A reads. Therefore, D will not sleep.")
    P_, Q_, R_, S_ = V("P"), V("Q"), V("R"), V("S")
    prem = [imp(P_, OR(Q_, R_)), imp(Q_, NOT(P_)), imp(S_, NOT(R_)), P_]
    ok, _bad = D.valid(prem, NOT(S_))
    assert ok
    fa, ga = (3, 4) if g6 else (1, 2)
    odd = fa * ga * 1 + 1
    jj = "j(j+2)" if g6 else "j(j+1)"
    j = sp.Symbol("j")
    expr = j * (j + 2) if g6 else j * (j + 1)
    closed = sp.factor(sp.summation(expr, (j, 0, n)))
    lo = 9 if g6 else 10
    val = sum(int(expr.subs(j, k)) for k in range(lo, 21))
    return {
        "id": f"mid-2022-g{group}", "title": f"Mid Exam 2022, Group {group}" if g6 else "Mid Exam 2021-22, Group 1", "exam": "mid", "year": 2022, "minutes": 90, "marks": 20,
        "note": "Online proctored exam, 5 questions.",
        "questions": [
            Q("1", "For each statement decide, using truth tables, whether it is a tautology, contradiction or contingency: " + "; ".join(f"({'abcdefg'[i]}) $ {tex(f)} $" for i, f in enumerate(fs)) + ".", 7, "tautology", q1_ans, qtype="long"),
            Q("2", f"Determine whether the argument is valid and explain why: {arg_text}", 4, "inference", [
                P(f"Let $P$: {names[0]}, $Q$: {names[1]}, $R$: {names[2]}, $S$: {names[3]}. Premises: $P \\rightarrow (Q \\vee R)$, $Q \\rightarrow \\neg P$, $S \\rightarrow \\neg R$, $P$. Conclusion: $\\neg S$."),
                P("1. $P$ and $P \\rightarrow (Q \\vee R)$: $Q \\vee R$ (modus ponens)."),
                P("2. $P$ and $Q \\rightarrow \\neg P$: $\\neg Q$ (modus tollens, since $\\neg\\neg P$)."),
                P("3. $Q \\vee R$ and $\\neg Q$: $R$ (disjunctive syllogism)."),
                P("4. $R$ and $S \\rightarrow \\neg R$: $\\neg S$ (modus tollens)."),
                P("The argument is **valid** (a truth table over $P, Q, R, S$ confirms: whenever all four premises are true, $\\neg S$ is true).")]),
            Q("3", f"$f, g, h$ map non-negative integers to non-negative integers: $f(n) = {'3n + 1' if g6 else 'n + 1'}$, $g(n) = {fa if False else ('4n' if g6 else '2n')}$, $h(n) = 1$ if $n$ is odd and $0$ otherwise. Find $(f \\circ g) \\circ h$.", 3, "functions", [
                P(f"$(f \\circ g)(m) = f(g(m)) = f({'4m' if g6 else '2m'}) = {'12m + 1' if g6 else '2m + 1'}$."),
                P(f"$((f \\circ g) \\circ h)(n) = {'12' if g6 else '2'}\\,h(n) + 1$, so it equals $ {odd if g6 else 3} $ if $n$ is odd and $1$ if $n$ is even."),
                P("Composition is associative, so $f \\circ (g \\circ h)$ gives the same: $g(h(n)) \\in \\{0, " + ("4" if g6 else "2") + "\\}$, then $f$ adds as above.")]),
            Q("4", f"Find the closed form of $\\sum_{{j=0}}^{{n}} {jj}$ and use it to evaluate $\\sum_{{j={lo}}}^{{20}} {jj}$.", 3, "sums", [
                P(f"$\\sum_{{j=0}}^{{n}} {jj} = \\sum j^2 + {'2' if g6 else ''}\\sum j = \\frac{{n(n+1)(2n+1)}}{{6}} + {'n(n+1)' if g6 else '\\frac{n(n+1)}{2}'} = {sp.latex(closed)}$."),
                P(f"$\\sum_{{j={lo}}}^{{20}} = S(20) - S({lo - 1}) = {int(closed.subs(n, 20))} - {int(closed.subs(n, lo - 1))} = {val}$.")]),
            Q("5", "Prove that $n^2 < 2^n$ for all integers $n \\ge 5$." if g6 else "Prove that $3^n < n!$ for all integers $n > 6$.", 3, "induction", ([
                P("**Basis** $n = 5$: $25 < 32$."),
                P("**Step:** assume $k^2 < 2^k$ for some $k \\ge 5$. Then $(k+1)^2 = k^2 + 2k + 1 < k^2 + k^2 = 2k^2$ because $2k + 1 < k^2$ for $k \\ge 3$."),
                P("So $(k+1)^2 < 2k^2 < 2 \\cdot 2^k = 2^{k+1}$. By induction $n^2 < 2^n$ for all $n \\ge 5$.")] if g6 else [
                P("**Basis** $n = 7$: $3^7 = 2187 < 7! = 5040$."),
                P("**Step:** assume $3^k < k!$ for some $k \\ge 7$. Then $3^{k+1} = 3 \\cdot 3^k < 3 \\cdot k! < (k+1)\\,k! = (k+1)!$ since $k + 1 \\ge 8 > 3$."),
                P("By induction $3^n < n!$ for all $n > 6$.")])),
        ]}


papers += [mid2022(6), mid2022(1)]

# ------------------------------------------------------------------------------------------- End 2022 (Groups 4, 6, 8)
S = sp.Function("S")


def orth_part(vs: list[list[int]]) -> list:
    qs = sp.GramSchmidt([sp.Matrix(v) for v in vs], True)
    perp = sp.Matrix(vs).nullspace()[0]
    perp = perp / perp.norm()
    return [P(f"Gram-Schmidt on $v_1 = {vtex(vs[0])}$, $v_2 = {vtex(vs[1])}$: $q_1 = v_1/\\|v_1\\|$, $B = v_2 - (q_1^Tv_2)\\,q_1$, $q_2 = B/\\|B\\|$."),
            M(f"q_1 = {vtex([sp.nsimplify(x) for x in qs[0]])},\\quad q_2 = {vtex([sp.nsimplify(x) for x in qs[1]])}"),
            P(f"$S^\\perp = N(A)$ for $A$ with rows $v_1, v_2$: solving $Ax = 0$ gives one direction, normalised: $q_3 = {vtex([sp.nsimplify(x) for x in perp])}$ (check: $q_3 \\perp v_1, v_2$). $\\{{q_1, q_2\\}}$ is an orthonormal basis of $S$ and $\\{{q_3\\}}$ of $S^\\perp$.")]


def poset_part(E: list[int]) -> list:
    mat = [[1 if b % a == 0 else 0 for b in E] for a in E]
    H = D.hasse(E)
    le = lambda a, b: b % a == 0  # noqa: E731
    mx = [x for x in E if not any(y != x and le(x, y) for y in E)]
    mn = [x for x in E if not any(y != x and le(y, x) for y in E)]
    gr = next((x for x in E if all(le(y, x) for y in E)), None)
    ls = next((x for x in E if all(le(x, y) for y in E)), None)
    return [P(f"(a) Zero-one matrix (rows and columns in the order {', '.join(map(str, E))}; entry 1 when the row element divides the column element):"), M(mtex(mat)),
            P("(b) Hasse diagram - covering pairs only (drawn upward): " + ", ".join(f"{a}–{b}" for a, b in H) + "."),
            P(f"(c) Maximal: {', '.join(map(str, mx))}. Minimal: {', '.join(map(str, mn))}. Greatest element: {gr if gr is not None else 'none'}. Least element: {ls if ls is not None else 'none'}.")]


def equiv_part(A: list[int], m: int) -> list:
    cls: dict[int, list[int]] = {}
    for x in A:
        cls.setdefault(x % m, []).append(x)
    return [P(f"(a) Reflexive: $x - x = 0$ is a multiple of {m}. Symmetric: if $x - y = {m}k$ then $y - x = {m}(-k)$. Transitive: $x - y = {m}k$, $y - z = {m}l$ give $x - z = {m}(k + l)$. So $R$ is an equivalence relation."),
            P("(b) Equivalence classes: " + "; ".join(f"$[{v[0]}] = \\{{{', '.join(map(str, v))}\\}}$" for v in cls.values()) + f". The partition of $A$ is $\\{{{', '.join('{' + ', '.join(map(str, v)) + '}' for v in cls.values())}\\}}$.")]


def null_proj(Mx: list[list[int]]) -> list:
    A = sp.Matrix(Mx)
    N = sp.Matrix.hstack(*A.nullspace())
    Pm = N * (N.T * N).inv() * N.T
    return [P(f"Solve $Ax = 0$: $N(A)$ has basis " + ", ".join(f"${vtex(list(c))}$" for c in A.nullspace()) + f" (dimension {N.cols}). Put them as the columns of $B$."),
            P("Projection onto $N(A) = C(B)$: $P = B(B^TB)^{-1}B^T$. (Equivalently $P = I - A^T(AA^T)^{-1}A$ when $A$ has independent rows, since $N(A) \\perp C(A^T)$.)"),
            M(f"P = {mtex(Pm)}"), P("Check: $P^2 = P$, $P = P^T$, and $AP = 0$.")]


gr6_S = sp.simplify(sp.rsolve(S(n + 2) + 3 * S(n + 1) + 2 * S(n) - 3**n, S(n), {S(0): 1, S(1): 1}))
A6 = sp.Matrix([[1, -1, 0, 1], [-1, -1, -2, -3], [-1, -1, -2, -3], [1, -1, 0, 1]])
B3 = sp.Matrix([[1, 3, 2], [0, 4, 1], [5, 0, 6]])
L3, U3, _ = B3.LUdecomposition()
D3 = sp.diag(*[U3[i, i] for i in range(3)])
papers.append({
    "id": "end-2022-g6", "title": "End Exam 2022, Group 6", "exam": "end", "year": 2022, "date": "03-2022", "minutes": 90, "marks": 20, "note": "Online proctored exam, 4 questions.",
    "questions": [
        Q("1a", "Solve the recurrence $S_{n+2} + 3S_{n+1} + 2S_n = 3^n$ for $n \\ge 0$ with $S_0 = 1$, $S_1 = 1$.", 3, "recurrence-nonhomog", [
            P("Characteristic equation $r^2 + 3r + 2 = (r+1)(r+2) = 0$: roots $-1, -2$. Homogeneous part $c_1(-1)^n + c_2(-2)^n$."),
            P("Particular: try $A\\,3^n$: $9A + 9A + 2A = 20A = 1$, so $A = \\frac{1}{20}$."),
            P("$S_0 = c_1 + c_2 + \\frac{1}{20} = 1$ and $S_1 = -c_1 - 2c_2 + \\frac{3}{20} = 1$ give $c_1 = \\frac{11}{4}$, $c_2 = -\\frac{9}{5}$."),
            P(f"$S_n = {sp.latex(gr6_S)}$. Check: $S_2 = 3^0 - 3 - 2 = -4$ from the recurrence, and the formula gives $\\frac{{11}}{{4}} - \\frac{{36}}{{5}} + \\frac{{9}}{{20}} = -4$.")]),
        Q("1b", "Let $S = \\text{span}\\{[1,-1,1], [1,0,1]\\}$. Find an orthonormal basis of $S$ and of $S^\\perp$.", 4, "gram-schmidt", orth_part([[1, -1, 1], [1, 0, 1]])),
        Q("2a", "Find the dimension of the column space of $A = " + mtex(A6) + "$.", 3, "rank", [
            M(f"\\operatorname{{rref}}(A) = {mtex(A6.rref()[0])}"), P(f"There are {A6.rank()} pivots, so $\\dim C(A) = \\operatorname{{rank}} A = {A6.rank()}$ (rows 2 and 3 are equal, and row 4 equals row 1).")]),
        Q("2b", "Find the projection matrix that projects a vector onto the null space of $A = " + mtex([[1, 1, 2], [2, 1, 1]]) + "$.", 4, "projections", null_proj([[1, 1, 2], [2, 1, 1]])),
        Q("3", "Let $A = \\{2, 3, 5, 10, 11, 15, 25\\}$ and $aRb$ iff $a$ divides $b$. (a) Give the zero-one matrix of $R$. (b) Draw the Hasse diagram. (c) Find the maximal and minimal elements. Is there a greatest and a least element?", 3, "posets", poset_part([2, 3, 5, 10, 11, 15, 25])),
        Q("4", "$R$ on $A = \\{1,2,3,4,5,6,7,8,9,10,12\\}$: $xRy$ iff $x - y$ is a multiple of 4. (a) Verify that $R$ is an equivalence relation. (b) Find the equivalence classes and the partition of $A$.", 3, "equiv-rel", equiv_part([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], 4)),
    ]})
papers.append({
    "id": "end-2022-g4", "title": "End Exam 2022, Group 4", "exam": "end", "year": 2022, "date": "03-2022", "minutes": 90, "marks": 20, "note": "Online proctored exam, 4 questions.",
    "questions": [
        Q("1a", "Solve the recurrence $S_n + 4S_{n-1} + 4S_{n-2} = 5(-2)^n$ for $n \\ge 2$.", 3, "recurrence-nonhomog", [
            P("Characteristic equation $r^2 + 4r + 4 = (r+2)^2 = 0$: $r = -2$ repeated. Homogeneous part $(c_1 + c_2 n)(-2)^n$."),
            P("$F(n) = 5(-2)^n$ and $-2$ is a root of multiplicity 2, so try $A n^2 (-2)^n$. Dividing the substituted equation by $(-2)^{n-2}$: $A[4n^2 - 8(n-1)^2 + 4(n-2)^2] = 20$, i.e. $8A = 20$, $A = \\frac{5}{2}$."),
            P("General solution: $S_n = (c_1 + c_2 n)(-2)^n + \\frac{5}{2}n^2(-2)^n$. (No initial values are given on the paper, so $c_1, c_2$ stay arbitrary.)")]),
        Q("1b", "Let $S = \\text{span}\\{[1,2,0], [8,1,-6]\\}$. Find an orthonormal basis of $S$ and of $S^\\perp$.", 4, "gram-schmidt", orth_part([[1, 2, 0], [8, 1, -6]])),
        Q("2a", "Decompose $A = " + mtex(B3) + "$ into $LDU$, giving intermediate steps.", 3, "lu", [
            P("Elimination: $\\ell_{21} = 0$ (already 0). $\\ell_{31} = 5$: $R_3 \\leftarrow R_3 - 5R_1 = [0, -15, -4]$. $\\ell_{32} = -\\frac{15}{4}$: $R_3 \\leftarrow R_3 + \\frac{15}{4}R_2 = [0, 0, -\\frac{1}{4}]$."),
            M(f"L = {mtex(L3)},\\quad D = {mtex(D3)},\\quad U = {mtex(D3.inv() * U3)}"),
            P("$D$ holds the pivots $1, 4, -\\frac{1}{4}$; $U$ is the upper-triangular factor divided by its pivots (1s on the diagonal). Check: $LDU = A$, and $\\det A = 1 \\cdot 4 \\cdot (-\\frac14) = -1$.")]),
        Q("2b", "Find the projection matrix that projects a vector onto the null space of $A = " + mtex([[1, 2, 3], [2, 4, 6]]) + "$.", 4, "projections",
          null_proj([[1, 2, 3], [2, 4, 6]]) + [P("Here $A$ has rank 1 (row 2 = 2 × row 1), so $N(A)$ is a plane and $P = I - \\frac{aa^T}{a^Ta}$ with $a = (1, 2, 3)$ gives the same matrix.")]),
        Q("3", "Let $A = \\{1, 2, 4, 8, 16, 32, 64\\}$ and $aRb$ iff $a$ divides $b$. (a) Zero-one matrix. (b) Hasse diagram. (c) Maximal, minimal, greatest and least elements.", 3, "posets", poset_part([1, 2, 4, 8, 16, 32, 64]) + [P("This poset is a chain (totally ordered).")]),
        Q("4", "$R$ on $A = \\{1, 2, \\dots, 10\\}$: $xRy$ iff $x - y$ is a multiple of 2. (a) Verify that $R$ is an equivalence relation. (b) Equivalence classes and partition.", 3, "equiv-rel", equiv_part(list(range(1, 11)), 2)),
    ]})
A8 = sp.Matrix([[1, 2], [2, 4]])
b8 = sp.Matrix([1, 3])
papers.append({
    "id": "end-2022-g8", "title": "End Exam 2022, Group 8", "exam": "end", "year": 2022, "date": "24-03-2022", "minutes": 90, "marks": 20, "note": "Online proctored exam, 4 questions.",
    "questions": [
        Q("1a", "Solve the recurrence $S_n + 5S_{n-1} + 6S_{n-2} = 42 \\cdot 4^n$ for $n \\ge 2$.", 3, "recurrence-nonhomog", [
            P("Characteristic equation $r^2 + 5r + 6 = (r+2)(r+3) = 0$: roots $-2, -3$."),
            P("Particular: try $A\\,4^n$. Dividing by $4^{n-2}$: $16A + 20A + 6A = 42 \\cdot 16$, so $42A = 672$, $A = 16$."),
            P("General solution: $S_n = c_1(-2)^n + c_2(-3)^n + 16 \\cdot 4^n$ (no initial values are given, so $c_1, c_2$ are arbitrary).")]),
        Q("1b", "Let $S = \\text{span}\\{[1,0,0], [2,0,3]\\}$. Find an orthonormal basis of $S$ and of $S^\\perp$.", 4, "gram-schmidt", orth_part([[1, 0, 0], [2, 0, 3]]) + [P("So $S$ is the $xz$-plane and $S^\\perp$ is the $y$-axis.")]),
        Q("2", "Let $A = \\{1, 3, 5, 15, 25, 30, 45\\}$ and $aRb$ iff $a$ divides $b$. (a) Zero-one matrix. (b) Hasse diagram. (c) Maximal, minimal, greatest and least elements.", 3, "posets", poset_part([1, 3, 5, 15, 25, 30, 45])),
        Q("3", "$R$ on $A = \\{1,2,3,4,5,6,7,8,9,10,12\\}$: $xRy$ iff $x - y$ is a multiple of 2. (a) Verify that $R$ is an equivalence relation. (b) Equivalence classes and partition.", 3, "equiv-rel", equiv_part([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12], 2)),
        Q("4a", "By the Gauss-Jordan method find the inverse of $A = " + mtex(B3) + "$. Give intermediate steps.", 3, "inverse", [
            M(f"[A \\mid I] = {mtex(B3.row_join(sp.eye(3)))}"),
            P("$R_3 \\leftarrow R_3 - 5R_1$; $R_3 \\leftarrow R_3 + \\frac{15}{4}R_2$ (now upper triangular, pivots $1, 4, -\\frac14$); scale $R_2 \\div 4$, $R_3 \\times (-4)$; clear above the pivots: $R_2 \\leftarrow R_2 - \\frac14 R_3$, $R_1 \\leftarrow R_1 - 2R_3 - 3R_2$."),
            M(f"[I \\mid A^{{-1}}] = {mtex(B3.row_join(sp.eye(3)).rref()[0])}"), M(f"A^{{-1}} = {mtex(B3.inv())}"), P("Check: $AA^{-1} = I$ ($\\det A = -1$).")]),
        Q("4b", "$Ax = b$ may have no solution when $b \\notin C(A)$; projecting $b$ onto $C(A)$ gives the best approximate solution. Find it for $" + mtex(A8) + "\\begin{bmatrix} x_1 \\\\ x_2 \\end{bmatrix} = " + vtex([1, 3]) + "$.", 4, "projections", [
            P("$C(A)$ is the line through $a = (1, 2)$ (column 2 = 2 × column 1), and $b = (1, 3) \\notin C(A)$."),
            P("Projection: $p = \\frac{a^Tb}{a^Ta}a = \\frac{7}{5}(1, 2) = (\\frac{7}{5}, \\frac{14}{5})$. Solve $Ax = p$: $x_1 + 2x_2 = \\frac{7}{5}$."),
            P("Note $A^TA = " + mtex(A8.T * A8) + "$ is singular, so the normal equations have infinitely many solutions: every $x$ with $x_1 + 2x_2 = \\frac75$ is a best approximate (least-squares) solution."),
            P(f"The shortest one (minimum norm, $x \\in C(A^T)$) is $x = {vtex(list(A8.pinv() * b8))}$. Error $e = b - p = (-\\frac25, \\frac15) \\perp C(A)$.")]),
    ]})

# ------------------------------------------------------------------------------------------- write
for pp in papers:
    for qq in pp["questions"]:
        qq["paper"] = pp["id"]
    pp["marks_sum"] = sum(qq["marks"] for qq in pp["questions"])
OUT.write_text(json.dumps({"subject": "dsma", "institute": "IIIT Sri City", "papers": papers}, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"{len(papers)} papers, {sum(len(x['questions']) for x in papers)} questions -> {OUT.relative_to(ROOT)}")
