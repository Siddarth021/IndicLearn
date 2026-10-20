"""OS course lessons: index, one lesson (English or Telugu) with a live solved example and a quick check, progress."""

from __future__ import annotations

import copy
import json
import random
import re
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

from indiclearn.studio.courses import oscalc
from indiclearn.studio.courses import papers as P
from indiclearn.studio.db import Database, jdump, jload, now

HERE = Path(__file__).resolve().parent


def _s(x: Any) -> str:
    """A model sometimes writes {"tip": "..."} where a plain string belongs."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return " - ".join(str(v) for v in x.values() if isinstance(v, (str, int, float)))
    return str(x or "")


def normalize(les: dict | None) -> dict | None:
    if not isinstance(les, dict) or not les.get("summary"):
        return None
    out = dict(les)
    out["summary"] = _s(les.get("summary"))
    out["sections"] = [{"heading": _s(x.get("heading", "")), "body": _s(x.get("body", ""))} for x in les.get("sections") or [] if isinstance(x, dict)]
    out["key_terms"] = [{"term": _s(x.get("term", "")), "meaning": _s(x.get("meaning", ""))} for x in les.get("key_terms") or [] if isinstance(x, dict) and x.get("term")]
    ex = les.get("example") if isinstance(les.get("example"), dict) else {"title": "", "body": _s(les.get("example"))}
    out["example"] = {"title": _s(ex.get("title", "")), "body": _s(ex.get("body", ""))}
    for k in ("exam_tips", "mistakes"):
        out[k] = [_s(x) for x in les.get(k) or [] if _s(x).strip()]
    out["diagram"] = _s(les.get("diagram", ""))
    out["title"] = _s(les.get("title", ""))
    return out


@lru_cache(maxsize=8)
def _load(subject: str, mtime: float) -> dict[str, dict]:
    path = HERE / f"{subject}_lessons.jsonl"
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue  # a half-written line while the lesson builder is saving
        d["en"] = normalize(d.get("en"))
        d["te"] = normalize(d.get("te"))
        if d["en"]:
            out[d["topic"]] = d
    return out


def lessons(subject: str = "os") -> dict[str, dict]:
    path = HERE / f"{subject}_lessons.jsonl"
    return _load(subject, path.stat().st_mtime) if path.exists() else {}


def index(db: Database, user_id: str, subject: str = "os") -> dict[str, Any]:
    syl = P.syllabus(subject)
    ls = lessons(subject)
    done = {r["topic"]: r["status"] for r in db.all("SELECT topic, status FROM lesson_progress WHERE user_id=? AND subject=?", (user_id, subject))}
    units = []
    for u in syl["units"]:
        units.append({"id": u["id"], "title": u["title"], "topics": [
            {"id": t["id"], "label": t["label"], "has_lesson": t["id"] in ls, "telugu": bool((ls.get(t["id"]) or {}).get("te")),
             "summary": ((ls.get(t["id"]) or {}).get("en") or {}).get("summary", ""), "numerical": bool(t.get("numerical")),
             "status": done.get(t["id"], "new")} for t in u["topics"]]})
    total = sum(len(u["topics"]) for u in units)
    return {"id": subject, "title": syl["title"], "short": syl.get("short", syl["title"]), "units": units,
            "done": sum(1 for s in done.values() if s == "learned"), "total": total,
            "books": syl.get("books", []), "links": syl.get("links", [])}


def _cached_te(db: Database, d: dict, subject: str) -> dict | None:
    if d.get("te"):
        return d["te"]
    r = db.one("SELECT payload_json FROM lesson_i18n WHERE subject=? AND topic=? AND lang='te'", (subject, d["topic"]))
    return normalize(jload(r["payload_json"])) if r else None


def _hosted(brain: Any) -> Any:
    """Only online engines translate a whole lesson: a laptop model would keep the student waiting for many minutes."""
    if brain is None:
        return None
    hosted = [p for p in getattr(brain, "providers", []) if not p.local]
    if not any(p.ok() for p in hosted):
        return None
    view = copy.copy(brain)
    view.providers = hosted
    return view


def _telugu(db: Database, d: dict, brain: Any, subject: str = "os") -> tuple[dict | None, bool]:
    """(Telugu lesson or None, being prepared). Never blocks the page: a missing translation is made in the background,
    once, by whichever online engine answers, and cached for every student."""
    te = _cached_te(db, d, subject)
    if te:
        return te, False
    key = subject + "/" + d["topic"]
    if key in _TE_RUNNING:
        return None, True
    view = _hosted(brain)
    if view is None:
        return None, False

    def work() -> None:
        try:
            with _TE_LOCKS.setdefault(key, threading.Lock()):
                if _cached_te(db, d, subject):
                    return
                out = translate(view, d["en"], "te")
                if out:
                    db.run("INSERT OR REPLACE INTO lesson_i18n(subject,topic,lang,payload_json,created_at) VALUES (?,?,?,?,?)",
                           (subject, d["topic"], "te", jdump(out), now()))
        except Exception:  # pragma: no cover - English stays available
            pass
        finally:
            _TE_RUNNING.discard(key)

    _TE_RUNNING.add(key)
    threading.Thread(target=work, daemon=True).start()
    return None, True


_TE_RUNNING: set[str] = set()
_TE_LOCKS: dict[str, threading.Lock] = {}


# The exact rules the solver (oscalc.py) uses - written by hand, so the lesson can never contradict the worked example.
RULES = {
    "cpu": ["Completion time (CT) = the moment the process finishes. Turnaround time TAT = CT - AT. Waiting time WT = TAT - BT. "
            "Response time = first time on the CPU - AT.",
            "Ties: the process that arrived earlier goes first; if they arrived together, the lower process number (P1 before P2).",
            "Round Robin: a process that arrives exactly when a time slice ends joins the ready queue BEFORE the preempted process.",
            "If no process is ready, the CPU is idle - show the idle gap in the Gantt chart; averages are over all processes."],
    "pages": ["A page fault happens whenever the referenced page is not in a frame - the first loads into empty frames are faults too.",
              "FIFO replaces the page that was loaded earliest; LRU the page used least recently; Optimal the page whose next use is "
              "farthest in the future (a page never used again is replaced first).",
              "Belady's anomaly: with FIFO, giving more frames can sometimes cause MORE page faults. LRU and Optimal never show it."],
    "bankers": ["Need = Max - Allocation. A process can finish if its Need <= Work (Work starts as Available).",
                "When it finishes it releases its allocation: Work = Work + Allocation. Repeat, each time taking the lowest-numbered "
                "process that can finish.",
                "If every process can finish, the state is SAFE and the order is a safe sequence; otherwise it is unsafe."],
    "disk": ["Total head movement = sum of |next cylinder - current cylinder| along the path.",
             "SCAN travels to the last cylinder before reversing; LOOK reverses at the last request in that direction.",
             "C-SCAN and C-LOOK jump back to the other end and continue in the same direction - say in your answer whether you "
             "count the jump (this app counts it)."],
    "fit": ["First fit: the first hole big enough. Best fit: the smallest hole that fits. Worst fit: the largest hole.",
            "The unused part of a hole stays as a smaller hole for the next process; a process that fits nowhere must wait."],
    "paging": ["Page number = logical address div page size; offset = logical address mod page size.",
               "Physical address = frame number x page size + offset.",
               "Offset bits = log2(page size); page-number bits = log2(number of pages)."],
    "eat": ["TLB hit: t + m (TLB lookup, then the data). TLB miss: t + 2m (TLB, page table in memory, then the data).",
            "EAT = h x (t + m) + (1 - h) x (t + 2m), where h is the hit ratio."],
}
DSMA_RULES = {
    "truth": ["A conditional $p \\rightarrow q$ is false ONLY when $p$ is true and $q$ is false; $p \\leftrightarrow q$ is true when both have the same value.",
              "Truth-table rows are listed starting from all true (T T, T F, F T, F F for two variables) - the order used in Rosen.",
              "Precedence: $\\neg$ first, then $\\wedge$, then $\\vee$, then $\\rightarrow$, then $\\leftrightarrow$."],
    "conditional": ["For $p \\rightarrow q$: converse $q \\rightarrow p$, inverse $\\neg p \\rightarrow \\neg q$, contrapositive $\\neg q \\rightarrow \\neg p$.",
                    "A conditional is equivalent to its contrapositive; the converse is equivalent to the inverse. A conditional is NOT equivalent to its converse.",
                    "\"p only if q\" means $p \\rightarrow q$; \"p if q\" means $q \\rightarrow p$; \"p is necessary for q\" means $q \\rightarrow p$; \"p is sufficient for q\" means $p \\rightarrow q$."],
    "equiv": ["Two formulas are logically equivalent when they have the same truth value in EVERY row of the truth table (equivalently, $P \\leftrightarrow Q$ is a tautology).",
              "$p \\rightarrow q \\equiv \\neg p \\vee q$; De Morgan: $\\neg(p \\wedge q) \\equiv \\neg p \\vee \\neg q$ and $\\neg(p \\vee q) \\equiv \\neg p \\wedge \\neg q$."],
    "taut": ["Tautology: true in every row. Contradiction: false in every row. Contingency: true in some rows and false in others.",
             "To decide, build the full truth table (all $2^n$ rows for $n$ variables) or simplify with the laws of logic."],
    "quant": ["$\\neg \\forall x\\, P(x) \\equiv \\exists x\\, \\neg P(x)$ and $\\neg \\exists x\\, P(x) \\equiv \\forall x\\, \\neg P(x)$.",
              "Nested quantifiers are read left to right: $\\forall x \\exists y$ lets $y$ depend on $x$; $\\exists y \\forall x$ needs ONE $y$ that works for every $x$.",
              "Over a finite domain, $\\forall$ is a big AND and $\\exists$ is a big OR over its elements."],
    "argument": ["An argument is valid when the conclusion is true in every row where ALL premises are true (premises $\\rightarrow$ conclusion is a tautology).",
                 "Modus ponens: $p, p \\rightarrow q \\vdash q$. Modus tollens: $\\neg q, p \\rightarrow q \\vdash \\neg p$. Affirming the conclusion and denying the hypothesis are fallacies."],
    "nf": ["PDNF (sum of minterms): OR together one minterm for every row where the formula is TRUE; in a minterm a variable appears plain if it is T in that row and negated if F.",
           "PCNF (product of maxterms): AND together one maxterm for every row where the formula is FALSE; in a maxterm a variable appears negated if it is T in that row and plain if F.",
           "Rows are numbered from 0 with T = 1 and the first variable as the most significant bit, so $\\Sigma m(\\dots)$ and $\\Pi M(\\dots)$ use complementary row numbers."],
    "sets": ["$|A \\cup B| = |A| + |B| - |A \\cap B|$; $A - B$ = elements of A not in B; $A \\oplus B = (A - B) \\cup (B - A)$.",
             "$|P(A)| = 2^{|A|}$ and $|A \\times B| = |A| \\cdot |B|$."],
    "func": ["From an $m$-element set to an $n$-element set: $n^m$ functions; $n!/(n-m)!$ one-to-one functions when $m \\le n$ (else 0); onto functions $\\sum_{i=0}^{n} (-1)^i \\binom{n}{i} (n-i)^m$.",
             "One-to-one: different inputs give different outputs. Onto: every element of the codomain is hit. Bijection: both - only then does $f^{-1}$ exist."],
    "sum": ["$\\sum_{k=1}^{n} k = n(n+1)/2$, $\\sum_{k=1}^{n} k^2 = n(n+1)(2n+1)/6$, $\\sum_{k=1}^{n} k^3 = (n(n+1)/2)^2$, $\\sum_{k=0}^{n} ar^k = a(r^{n+1}-1)/(r-1)$ for $r \\ne 1$."],
    "receval": ["Apply the recursive rule from the initial values upward, one term at a time; write every intermediate term."],
    "count": ["Product rule: a task done in a sequence of steps with $n_1, n_2, \\dots$ ways has $n_1 n_2 \\cdots$ ways. Sum rule: disjoint alternatives add.",
              "When the alternatives overlap, subtract the overlap (inclusion-exclusion)."],
    "inclexcl": ["$|A \\cup B \\cup C| = |A|+|B|+|C| - |A \\cap B| - |A \\cap C| - |B \\cap C| + |A \\cap B \\cap C|$.",
                 "Multiples of $k$ in $1..N$: $\\lfloor N/k \\rfloor$; multiples of both $a$ and $b$: $\\lfloor N/\\mathrm{lcm}(a,b) \\rfloor$.",
                 "Derangements: $D_n = n! \\sum_{i=0}^{n} (-1)^i / i!$ ($D_1=0, D_2=1, D_3=2, D_4=9, D_5=44$)."],
    "pigeon": ["If $N$ objects go into $k$ boxes, some box holds at least $\\lceil N/k \\rceil$ objects.",
               "The fewest objects that GUARANTEE $r$ in one box is $k(r-1)+1$."],
    "permcomb": ["$P(n,r) = n!/(n-r)!$ (order matters); $C(n,r) = n!/(r!(n-r)!)$ (order does not matter).",
                 "With repetition: $n^r$ ordered; $C(n+r-1, r)$ unordered (stars and bars). Arrangements of a word with repeated letters: $n!/(n_1! n_2! \\cdots)$."],
    "binom": ["$(x+y)^n = \\sum_{k=0}^{n} \\binom{n}{k} x^{n-k} y^k$, so the coefficient of $x^{n-k} y^k$ in $(ax+by)^n$ is $\\binom{n}{k} a^{n-k} b^k$.",
              "Pascal: $\\binom{n+1}{k} = \\binom{n}{k-1} + \\binom{n}{k}$; $\\sum_k \\binom{n}{k} = 2^n$."],
    "recur": ["For $a_n = c_1 a_{n-1} + c_2 a_{n-2}$ the characteristic equation is $r^2 - c_1 r - c_2 = 0$.",
              "Distinct roots $r_1, r_2$: $a_n = \\alpha_1 r_1^n + \\alpha_2 r_2^n$. Repeated root $r$: $a_n = (\\alpha_1 + \\alpha_2 n) r^n$. Find the $\\alpha$'s from the initial conditions."],
    "recurnh": ["$a_n = a_n^{(h)} + a_n^{(p)}$: solve the associated homogeneous part, then guess a particular solution of the same form as $F(n)$.",
                "If $F(n) = (\\text{polynomial}) \\cdot s^n$ and $s$ is a root of the characteristic equation with multiplicity $m$, multiply the guess by $n^m$.",
                "Find the constants of the homogeneous part AFTER adding the particular solution, using the initial conditions."],
    "relprop": ["On a set $A$: reflexive if $(a,a) \\in R$ for EVERY $a \\in A$; symmetric if $(a,b) \\in R \\Rightarrow (b,a) \\in R$; antisymmetric if $(a,b),(b,a) \\in R \\Rightarrow a = b$; transitive if $(a,b),(b,c) \\in R \\Rightarrow (a,c) \\in R$.",
                "A relation can be both symmetric and antisymmetric (e.g. only loops), or neither.",
                "Counts on an $n$-element set: all relations $2^{n^2}$, reflexive $2^{n(n-1)}$, symmetric $2^{n(n+1)/2}$, antisymmetric $2^n 3^{n(n-1)/2}$."],
    "relcomp": ["$S \\circ R = \\{(a,c) : (a,b) \\in R \\text{ and } (b,c) \\in S \\text{ for some } b\\}$ - apply $R$ first. Its matrix is the Boolean product $M_R \\odot M_S$.",
                "$R^{-1} = \\{(b,a) : (a,b) \\in R\\}$ - transpose the matrix."],
    "closure": ["Reflexive closure: add every $(a,a)$. Symmetric closure: add $(b,a)$ for every $(a,b)$.",
                "Warshall: $W_k[i][j] = W_{k-1}[i][j] \\vee (W_{k-1}[i][k] \\wedge W_{k-1}[k][j])$ for $k = 1..n$; $W_n$ is the transitive closure."],
    "equivrel": ["An equivalence relation is reflexive, symmetric and transitive; its equivalence classes partition the set.",
                 "Congruence mod $m$ on integers has exactly $m$ classes $[0], [1], \\dots, [m-1]$."],
    "poset": ["A partial order is reflexive, antisymmetric and transitive. A Hasse diagram keeps only covering pairs: no loops, no edges implied by transitivity, larger elements drawn higher.",
              "Maximal: nothing above it; greatest: above EVERY element (unique if it exists). Minimal and least likewise.",
              "LUB (join) = the least of the common upper bounds; GLB (meet) = the greatest of the common lower bounds. For divisibility, LUB = lcm and GLB = gcd."],
    "lattice": ["A lattice is a poset in which every pair has both a LUB and a GLB. The divisors of $n$ under divisibility form a lattice with join = lcm, meet = gcd.",
                "Complement of $a$ in a bounded lattice: $b$ with $a \\vee b = 1$ and $a \\wedge b = 0$; in $D_n$ that needs $\\mathrm{lcm} = n$ and $\\gcd = 1$."],
    "elim": ["Multiplier $\\ell_{ij}$ = (entry to eliminate) / (pivot); subtract $\\ell_{ij}$ times the pivot row. Pivots are the first nonzero entries of the rows of $U$.",
             "A zero in a pivot position needs a row exchange; if no exchange helps, the matrix is singular."],
    "inverse": ["Gauss-Jordan: reduce $[A \\mid I]$ to $[I \\mid A^{-1}]$. $A^{-1}$ exists iff $A$ has $n$ (nonzero) pivots iff $\\det A \\ne 0$.",
                "$(AB)^{-1} = B^{-1}A^{-1}$; for $2 \\times 2$, $\\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}^{-1} = \\frac{1}{ad-bc}\\begin{bmatrix} d & -b \\\\ -c & a \\end{bmatrix}$."],
    "lu": ["$L$ has 1s on its diagonal and the multipliers $\\ell_{ij}$ below it; $U$ is the upper-triangular matrix left by elimination (no row exchanges). $\\det A$ = product of the pivots."],
    "transpose": ["$(AB)^T = B^T A^T$, $(A^{-1})^T = (A^T)^{-1}$; $A^TA$ is always symmetric. A permutation matrix $P$ has $P^{-1} = P^T$; with row exchanges, $PA = LU$."],
    "nullspace": ["Reduce to RREF $R$; pivot columns give pivot variables, the others are free. Each special solution sets one free variable to 1 and the other free variables to 0.",
                  "$N(A)$ is spanned by the special solutions; there are $n - r$ of them."],
    "rank": ["rank $r$ = number of pivots. $Ax = b$ is solvable iff elimination gives no row $0 = c$ with $c \\ne 0$.",
             "Complete solution $x = x_p + x_n$. $r = n$: at most one solution; $r < n$: infinitely many when solvable; $r = m$: solvable for every $b$."],
    "basis": ["Vectors are independent iff the matrix with them as columns has rank equal to the number of vectors. A basis is an independent spanning set; dimension = number of basis vectors.",
              "More than $n$ vectors in $\\mathbb{R}^n$ are always dependent."],
    "four": ["For an $m \\times n$ matrix of rank $r$: $\\dim C(A) = r$, $\\dim N(A) = n - r$, $\\dim C(A^T) = r$, $\\dim N(A^T) = m - r$.",
             "$N(A) \\perp C(A^T)$ in $\\mathbb{R}^n$ and $N(A^T) \\perp C(A)$ in $\\mathbb{R}^m$."],
    "ortho": ["$x \\perp y$ iff $x^T y = 0$. Every row of $A$ is orthogonal to every vector of $N(A)$."],
    "proj": ["Projection onto a line through $a$: $p = \\dfrac{a^T b}{a^T a}\\, a$. Onto $C(A)$: $\\hat{x} = (A^TA)^{-1}A^Tb$, $p = A\\hat{x}$, $P = A(A^TA)^{-1}A^T$.",
             "Least squares: solve the normal equations $A^TA\\hat{x} = A^Tb$; the error $e = b - p$ is orthogonal to $C(A)$."],
    "gs": ["Gram-Schmidt: $A = a$, $B = b - \\dfrac{A^T b}{A^T A} A$, $C = c - \\dfrac{A^T c}{A^T A} A - \\dfrac{B^T c}{B^T B} B$; then divide each by its length. $A = QR$ with $R = Q^TA$ upper triangular."],
    "eig": ["Solve $\\det(A - \\lambda I) = 0$; for each $\\lambda$ find $N(A - \\lambda I)$.",
            "Sum of eigenvalues = trace; product = determinant. A triangular matrix has its eigenvalues on the diagonal."],
    "diag": ["$A = S\\Lambda S^{-1}$ with eigenvectors as the columns of $S$; then $A^k = S\\Lambda^k S^{-1}$.",
             "$A$ is diagonalizable iff it has $n$ independent eigenvectors; $n$ distinct eigenvalues are enough."],
    "spectral": ["A real symmetric matrix has real eigenvalues and orthonormal eigenvectors: $A = Q\\Lambda Q^T = \\sum \\lambda_i q_i q_i^T$."],
    "posdef": ["A symmetric $A$ is positive definite iff all eigenvalues $> 0$ iff all pivots $> 0$ iff all leading principal minors $> 0$ iff $x^TAx > 0$ for every $x \\ne 0$.",
               "For $\\begin{bmatrix} a & b \\\\ b & c \\end{bmatrix}$: positive definite iff $a > 0$ and $ac - b^2 > 0$."],
    "svd": ["$A = U\\Sigma V^T$: the singular values $\\sigma_i$ are the square roots of the eigenvalues of $A^TA$ (largest first); the columns of $V$ are eigenvectors of $A^TA$ and $u_i = Av_i/\\sigma_i$.",
            "The number of nonzero singular values is the rank of $A$."],
}
SUBJECT_RULES = {"os": RULES, "dsma": DSMA_RULES}
RULES_HEADING = {"en": "Exact rules for numericals", "te": "Numericals కి ఖచ్చితమైన నియమాలు"}


def rules_for(subject: str, kind: str | None) -> list[str] | None:
    return SUBJECT_RULES.get(subject, {}).get(kind or "") or None


def _with_rules(body: dict, kind: str | None, lang: str, subject: str = "os") -> dict:
    """Numerical topics: drop any model-written 'conventions' section and put the solver's exact rules in its place."""
    rl = rules_for(subject, kind)
    if not rl:
        return body
    secs = [x for x in body["sections"] if not re.search(r"tie|convention|rule", x.get("heading", ""), re.IGNORECASE)]
    rules = "\n".join("- " + r for r in rl)
    return {**body, "sections": secs + [{"heading": RULES_HEADING.get(lang, RULES_HEADING["en"]), "body": rules, "exact": True}]}


def lesson(db: Database, user_id: str, topic_id: str, lang: str = "en", seed: int | None = None, brain: Any = None, subject: str = "os") -> dict[str, Any]:
    order = [(u, t) for u, t in P.topics_of(None, subject)]
    pos = next((i for i, (_u, t) in enumerate(order) if t["id"] == topic_id), None)
    if pos is None:
        raise P.PaperError("Lesson not found.")
    u, t = order[pos]
    d = lessons(subject).get(topic_id)
    if not d:
        raise P.PaperError("This lesson is not ready yet.")
    te, te_pending = _telugu(db, d, brain, subject) if lang == "te" else (None, False)
    body = _with_rules(te if te else d["en"], t.get("numerical"), "te" if te else "en", subject)
    rng = random.Random(seed)
    example = None
    calc = P.calc(subject)
    if t.get("numerical") and t["numerical"] in calc.GENERATORS:  # a fresh, exactly solved problem every time ("New example")
        g = calc.generate(t["numerical"], rng, 4, "medium", topic_id)
        example = {"blocks": g["blocks"], "answer": g["answer"]}
    bank = [q for q in P.bank(subject) if q["topic"] == topic_id and q["qtype"] == "mcq"]
    rng.shuffle(bank)
    check = [{"q": q["q"], "options": q["options"], "correct": q["correct"], "why": q["answer"], "id": q["id"]} for q in bank[:3]]
    st = db.one("SELECT status FROM lesson_progress WHERE user_id=? AND subject=? AND topic=?", (user_id, subject, topic_id))
    if not st:
        db.run("INSERT OR IGNORE INTO lesson_progress(user_id,subject,topic,status,updated_at) VALUES (?,?,?,?,?)", (user_id, subject, topic_id, "opened", now()))
    nav = lambda i: {"id": order[i][1]["id"], "label": order[i][1]["label"]} if 0 <= i < len(order) else None  # noqa: E731
    return {"subject": subject, "topic": topic_id, "unit": {"id": u["id"], "title": u["title"]}, "label": t["label"], "lang": "te" if te else "en",
            "has_telugu": bool(d.get("te")), "te_pending": te_pending, "lesson": body, "example": example, "check": check, "status": (st["status"] if st else "opened"),
            "refs": t.get("refs", []), "prereq": [{"id": x, "label": lbl} for x in t.get("prereq", []) for lbl in [_label(subject, x)] if lbl],
            "prev": nav(pos - 1), "next": nav(pos + 1)}


def _label(subject: str, topic_id: str) -> str | None:
    return next((t["label"] for _u, t in P.topics_of(None, subject) if t["id"] == topic_id), None)


def mark(db: Database, user_id: str, topic_id: str, status: str, subject: str = "os") -> dict[str, Any]:
    if status not in ("learned", "opened"):
        raise P.PaperError("Unknown status.")
    db.run("INSERT INTO lesson_progress(user_id,subject,topic,status,updated_at) VALUES (?,?,?,?,?) "
           "ON CONFLICT(user_id,subject,topic) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at", (user_id, subject, topic_id, status, now()))
    return {"ok": True, "status": status}


# ---------------------------------------------------------------- Telugu (shared with training/make_os_lessons.py)


def flatten(lesson: dict) -> dict[str, str]:
    f = {"title": lesson.get("title", ""), "summary": lesson["summary"], "diagram": lesson.get("diagram", "")}
    for i, s in enumerate(lesson["sections"]):
        f[f"s{i}h"], f[f"s{i}b"] = s.get("heading", ""), s.get("body", "")
    for i, k in enumerate(lesson.get("key_terms", [])):
        f[f"k{i}"] = k.get("meaning", "")
    ex = lesson.get("example") or {}
    f["exh"], f["exb"] = ex.get("title", ""), ex.get("body", "")
    for i, t in enumerate(lesson.get("exam_tips", [])):
        f[f"t{i}"] = t
    for i, t in enumerate(lesson.get("mistakes", [])):
        f[f"m{i}"] = t
    return {k: v for k, v in f.items() if v}


def unflatten(lesson: dict, f: dict[str, str]) -> dict:
    g = lambda k, d="": f.get(k, d)  # noqa: E731
    return {"title": g("title", lesson.get("title", "")), "summary": g("summary"), "diagram": g("diagram"),
            "sections": [{"heading": g(f"s{i}h", s.get("heading", "")), "body": g(f"s{i}b", s.get("body", ""))} for i, s in enumerate(lesson["sections"])],
            "key_terms": [{"term": k["term"], "meaning": g(f"k{i}", k.get("meaning", ""))} for i, k in enumerate(lesson.get("key_terms", []))],
            "example": {"title": g("exh"), "body": g("exb")},
            "exam_tips": [g(f"t{i}", t) for i, t in enumerate(lesson.get("exam_tips", []))],
            "mistakes": [g(f"m{i}", t) for i, t in enumerate(lesson.get("mistakes", []))]}


SYSTEM_TE = """Translate the JSON values into Telugu the way an Andhra/Telangana engineering lecturer explains in class:
Telugu sentences written in TELUGU SCRIPT (తెలుగు లిపి), with every technical term kept in ENGLISH LETTERS exactly as in the
source - e.g. process, CPU, scheduling, time quantum, preemptive, context switch, ready queue, page fault, semaphore, deadlock,
kernel, thread, Gantt chart, turnaround time. Never spell an English technical term in Telugu letters (write "scheduling",
not "షెడ్యూలింగ్"), and never write Telugu in English letters. Keep **bold**, "- " bullets, numbers and symbols, and copy
everything between $ signs (mathematics) exactly as it is.
Return ONLY JSON with exactly the same keys."""


def _te_ok(src: str, out: str) -> bool:
    te = len(re.findall(r"[\u0C00-\u0C7F]", out))
    letters = len(re.findall(r"[A-Za-z\u0C00-\u0C7F]", out)) or 1
    return te / letters >= 0.2 or len(src) < 25  # technical Telugu is English-term heavy; short labels may stay English


SYSTEM_TE_TEXT = SYSTEM_TE.replace("Translate the JSON values", "Translate the text").replace(
    "Return ONLY JSON with exactly the same keys.", "Output ONLY the Telugu translation, nothing else.")


def _one(brain: Any, src: str, heading: bool) -> str | None:
    for _ in range(3):
        r = brain._run(SYSTEM_TE_TEXT, src, False, 2500)
        text = (r[0] if r else "").strip().strip('"')
        if text and (heading or _te_ok(src, text)):
            return text
    return src if heading else None


def translate(brain: Any, lesson: dict, lang: str) -> dict | None:
    """Whole lesson in one plain-text call with ### markers (fast), then field-by-field retries for anything that came back wrong."""
    f = flatten(lesson)
    body = "\n\n".join(f"### {k}\n{v}" for k, v in f.items())
    sys_all = SYSTEM_TE_TEXT + "\nThe text has sections that start with lines like '### s0b'. Keep every '### key' line exactly as it is and translate only the text under it."
    r = brain._run(sys_all, body, False, 12000)
    got: dict[str, str] = {}
    if r:
        for part in re.split(r"(?m)^###\s*", r[0]):
            key, _, text = part.partition("\n")
            if key.strip() in f and text.strip():
                got[key.strip()] = text.strip()
    out: dict[str, str] = {}
    for k, src in f.items():
        heading = k in ("title", "exh") or (k[0] == "s" and k.endswith("h"))
        if k in got and (heading or _te_ok(src, got[k])):
            out[k] = got[k]
        else:
            one = _one(brain, src, heading)
            if one is None:
                return None
            out[k] = one
    return unflatten(lesson, out)
