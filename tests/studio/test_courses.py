"""Courses (OS, DSMA): the course study document (default + personal map, chat, quiz), exact DSMA solvers, LaTeX handling."""

import itertools
import random

import pytest

from indiclearn.studio.courses import dsmacalc as D
from indiclearn.studio.courses import papers as P


# ------------------------------------------------------------------ exact DSMA generators
@pytest.mark.parametrize("kind", sorted(D.GENERATORS))
def test_every_dsma_generator_works(kind):
    for s in range(2):
        for diff in ("easy", "hard"):
            g = D.generate(kind, random.Random(s), 5, diff, "t")
            assert g["blocks"] and g["answer"]
            if kind in D.VALUE_GENERATORS:
                n = D.gen_nat(kind, random.Random(s), 2, diff, "t")
                assert isinstance(n["value"], (int, float))
            if kind in D.MCQ_KINDS:
                m = D.gen_mcq_numeric(kind, random.Random(s), 2, diff, "t")
                assert len(set(m["options"])) == 4 and 0 <= m["correct"] < 4
            q = D.gen_msq_numeric(kind, random.Random(s), 2, diff, "t")
            if q:
                assert 1 <= len(q["correct"]) <= 3 and len(q["options"]) == 4


def test_dsma_counting_answers_match_brute_force():
    for m in range(1, 6):
        for n in range(1, 5):
            assert D.surjections(m, n) == sum(1 for f in itertools.product(range(n), repeat=m) if len(set(f)) == n)
    assert [D.bell(n) for n in range(1, 7)] == [1, 2, 5, 15, 52, 203]
    A = [1, 2, 3, 4, 5]
    rng = random.Random(1)
    for _ in range(50):
        R = {(rng.choice(A), rng.choice(A)) for _ in range(5)}
        T = set(R)
        while True:
            new = {(a, c) for a, b in T for b2, c in T if b == b2} - T
            if not new:
                break
            T |= new
        W = D.warshall(R, A)[-1]
        assert {(A[i], A[j]) for i in range(5) for j in range(5) if W[i][j]} == T


def test_dsma_logic_and_matrices_are_exact():
    for name, fs in D.KNOWN.items():
        assert all(D.classify(f) == name for f in fs)
    for name, prem, concl in D.ARGS:
        assert D.valid(prem, concl)[0] == ("fallacy" not in name and "invalid" not in name)
    rng = random.Random(3)
    for _ in range(20):
        A, lams, _S = D._eig_matrix(rng, rng.choice([2, 3]))
        assert sorted(A.eigenvals(multiple=True)) == sorted(lams)
        L, U, _steps = D.eliminate(D._unit_lower(rng, 3) * D._upper(rng, 3))
        assert (L * U - (L * U)).is_zero_matrix


# ------------------------------------------------------------------ LaTeX in model JSON and in PDFs
def test_latex_inside_model_json_survives():
    from indiclearn.studio.brain import parse_json

    got = parse_json(r'{"a": "Use $\frac{1}{2}$, $\neg p \rightarrow q$, $\binom{n}{k}$, $\lambda \le 3$", "b": "line\nnext"}')
    assert got["a"] == r"Use $\frac{1}{2}$, $\neg p \rightarrow q$, $\binom{n}{k}$, $\lambda \le 3$" and got["b"] == "line\nnext"


def test_pdf_text_has_no_raw_latex():
    from indiclearn.studio.courses.texplain import split_matrices, text

    assert text(r"$F = (p \rightarrow q) \wedge \neg r$") == "F = (p → q) ∧ ¬r"
    assert text(r"$\binom{5}{2} = \frac{5!}{2!\,3!}$") == "C(5, 2) = (5!)/(2! 3!)"
    assert [k for k, _v in split_matrices(r"A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}")] == ["text", "matrix"]


# ------------------------------------------------------------------ courses as study documents
def test_courses_list_has_os_and_dsma(client, auth):
    cs = client.get("/api/courses").json()
    assert [c["id"] for c in cs] == ["os", "dsma"]
    assert any(p["name"] == "GATE-style (DSMA)" for p in cs[1]["presets"])
    idx = client.get("/api/courses/dsma/lessons", headers=auth).json()
    assert idx["total"] == 47 and len(idx["units"]) == 6 and any(b["kind"] == "text" for b in idx["books"])
    assert client.get("/api/courses/nope/lessons", headers=auth).status_code == 404


def test_course_is_a_study_document_with_map_chat_quiz(client, auth):
    doc = client.post("/api/courses/os/open", headers=auth).json()
    assert doc["course"] == "os" and doc["status"] == "ready"
    assert all(d["id"] != doc["id"] for d in client.get("/api/docs", headers=auth).json())  # not an upload
    g = client.get(f"/api/docs/{doc['id']}/graph?mode=default", headers=auth).json()
    ids = {n["id"] for n in g["nodes"]}
    assert {"root", "u1", "round-robin", "page-replacement"} <= ids
    assert any(e["type"] == "prereq" for e in g["edges"])
    # a lesson opened, a quick check answered right, the lesson marked learned -> the personal map shows it
    les = client.get("/api/courses/os/lessons/round-robin", headers=auth).json()
    assert les["doc_id"] == doc["id"]
    assert client.post("/api/courses/os/lessons/round-robin/check?correct=true", headers=auth).json()["ok"]
    client.post("/api/courses/os/lessons/round-robin/mark?status=learned", headers=auth)
    pg = client.get(f"/api/docs/{doc['id']}/graph?mode=personal", headers=auth).json()
    rr = next(n for n in pg["nodes"] if n["id"] == "round-robin")
    assert rr["mastery"]["status"] in ("learning", "mastered") and rr["mastery"]["evidence"] >= 2
    # doubts and quizzes work on the course exactly as on notes
    r = client.post(f"/api/docs/{doc['id']}/chat", json={"message": "what is a time quantum?", "lang": "en", "focus_node": "round-robin"}, headers=auth).json()
    assert r["grounded"] and any(c["section_id"] == "round-robin" for c in r["citations"])
    q = client.post(f"/api/docs/{doc['id']}/quiz", json={"scope": "u2", "n": 5, "lang": "en"}, headers=auth).json()
    assert len(q["questions"]) == 5
    # opening again is the same document (progress kept)
    assert client.post("/api/courses/os/open", headers=auth).json()["id"] == doc["id"]


def test_dsma_papers_print(client, auth):
    for preset in ("gate", "end"):
        bp = {**next(p for p in P.PRESETS if p["id"] == preset), "subject": "dsma", "units": None}
        p = client.post("/api/papers", json={"blueprint": bp}, headers=auth)
        assert p.status_code == 200, p.text
        p = p.json()
        assert p["course"] == "dsma" and p["total"] == bp["total"]
        pdf = client.get(f"/api/papers/{p['id']}/pdf?answers=true", headers=auth)
        assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-" and "DSMA" in pdf.headers["content-disposition"]


def test_past_paper_subject_is_detected(client, auth):
    text = ("IIIT Sri City  Mid Semester Examination  Discrete Structures and Matrix Algebra  Time: 90 minutes  Max Marks: 25\n"
            "1. Construct the truth table of (p -> q) and decide whether it is a tautology. [3 marks]\n"
            "2. Solve the recurrence relation a_n = 5a_{n-1} - 6a_{n-2}. [3 marks]\n"
            "3. Prove by mathematical induction that the sum is n(n+1)/2. [3 marks]\n"
            "4. Find the rank and the null space of the matrix A using elimination. [4 marks]\n"
            "5. Find the eigenvalues and eigenvectors of A. [4 marks]\n"
            "6. Draw the Hasse diagram of the poset of divisors of 24. [4 marks]\n"
            "7. Use the pigeonhole principle to show ... [4 marks]\n")
    r = client.post("/api/papers/pattern", files={"file": ("mid.txt", text.encode(), "text/plain")}, headers=auth)
    assert r.status_code == 200, r.text
    assert r.json()["blueprint"]["subject"] == "dsma"


def test_telugu_never_blocks_the_lesson(client, auth, monkeypatch):
    """No online engine: the lesson opens in English at once (no waiting on a slow laptop model)."""
    import time

    t = time.time()
    r = client.get("/api/courses/dsma/lessons/eigen?lang=te", headers=auth).json()
    assert time.time() - t < 5 and r["lang"] == "en" and r["te_pending"] is False
    r = client.get("/api/courses/os/lessons/round-robin?lang=te", headers=auth).json()
    assert r["lang"] == "te"  # built-in Telugu


def test_previous_papers_are_in_the_course(client, auth):
    """The institute's DSMA papers: listed, readable with worked answers, printable, on lessons, in the bank and in papers."""
    lst = client.get("/api/courses/dsma/pyq", headers=auth).json()
    assert {p["id"] for p in lst} >= {"mid-2025", "quiz1-2024", "quiz2-2024", "mid-2022-g6", "end-2022-g8"}
    for p in lst:
        full = client.get(f"/api/courses/dsma/pyq/{p['id']}?answers=true", headers=auth).json()
        assert sum(q["marks"] for q in full["questions"]) == p["marks"]  # every question of the paper is there
        assert all(q["answer_blocks"] for q in full["questions"])
    bare = client.get("/api/courses/dsma/pyq/mid-2025", headers=auth).json()
    assert "answer_blocks" not in bare["questions"][0]  # answers only when asked for
    pdf = client.get("/api/courses/dsma/pyq/end-2022-g4/pdf?answers=true", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"
    les = client.get("/api/courses/dsma/lessons/posets", headers=auth).json()
    assert any("End Exam 2022" in q["paper_title"] for q in les["pyq"])
    assert sum(1 for q in P.bank("dsma") if q.get("source") == "pyq") >= 60
    # keys that were checked: the paper's own typo is flagged, a question with no right option never reaches a quiz
    q2 = client.get("/api/courses/dsma/pyq/quiz2-2024?answers=true", headers=auth).json()["questions"][0]
    assert "116" in q2["paper_issue"]
    assert not any(q["id"] == "pyq-quiz1-2024-4" for q in P.bank("dsma"))


def test_pyq_worked_answers_are_exact():
    import sympy as sp

    from indiclearn.studio.courses import pyq

    qs = {(p["id"], q["n"]): q for p in pyq.papers("dsma") for q in p["questions"]}
    txt = lambda k: " ".join(b.get("p", "") + b.get("math", "") for b in qs[k]["answer_blocks"])  # noqa: E731
    assert "3014" in txt(("mid-2022-g6", "4")) and "2750" in txt(("mid-2022-g1", "4"))
    B = sp.Matrix([[1, 3, 2], [0, 4, 1], [5, 0, 6]])
    assert all(str(x) in txt(("end-2022-g8", "4a")) for x in (-24, 18, 5, -15, -4))
    assert B.inv() == sp.Matrix([[-24, 18, 5], [-5, 4, 1], [20, -15, -4]])
