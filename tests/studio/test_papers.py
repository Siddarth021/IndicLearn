"""Exam papers: blueprint validation, notes-coverage intersection, answers hidden until asked, printable PDFs."""

from __future__ import annotations

import pytest

from indiclearn.studio.courses import papers as P

MID = {"name": "Mid-semester", "total": 25, "minutes": 90, "units": ["u2", "u3"], "difficulty": "college",
       "rows": [{"count": 3, "marks": 3, "qtype": "short"}, {"count": 2, "marks": 4, "qtype": "long"}, {"count": 2, "marks": 4, "qtype": "numerical"}]}


def test_blueprint_must_add_up():
    with pytest.raises(P.PaperError, match="add up to 24"):
        P.validate({**MID, "rows": [{"count": 4, "marks": 6, "qtype": "auto"}]})
    with pytest.raises(P.PaperError, match="1 or 2 marks"):
        P.validate({"total": 5, "rows": [{"count": 1, "marks": 5, "qtype": "mcq"}]})
    assert P.validate(MID)["total"] == 25
    for preset in P.PRESETS:
        assert P.validate(preset)["total"] == preset["total"]


def test_coverage_matches_whole_words_only():
    cov = P.coverage("Round robin uses a time quantum. A semaphore has wait and signal. The semaphore is counting.", ["u3", "u4"])
    assert "round-robin" in cov["covered"] and "semaphores" in cov["covered"]
    assert "bankers" not in cov["covered"]


def test_numerical_bank_questions_never_reach_a_paper():
    assert P.numericish({"qtype": "medium", "q": "Calculate the average waiting time for P1 (5), P2 (3), P3 (8)."})
    assert not P.numericish({"qtype": "short", "q": "Why does FCFS give a high average waiting time?"})


def test_paper_flow_hides_answers_until_asked(client, auth):
    c = client.get("/api/courses").json()[0]
    assert c["id"] == "os" and len(c["units"]) == 8
    r = client.post("/api/papers", json={"blueprint": MID}, headers=auth)
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["total"] == 25 and len(p["questions"]) == 7
    assert not p["answers_included"] and all("answer_text" not in q and "correct" not in q for q in p["questions"])
    assert sum(q["kind"] == "numerical" for q in p["questions"]) == 2
    full = client.get(f"/api/papers/{p['id']}?answers=true", headers=auth).json()
    assert all(q.get("answer_text") or q.get("answer_blocks") for q in full["questions"])
    for ans in ("false", "true"):
        pdf = client.get(f"/api/papers/{p['id']}/pdf?answers={ans}", headers=auth)
        assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-" and len(pdf.content) > 3000
    assert client.get("/api/papers", headers=auth).json()[0]["id"] == p["id"]
    other = {"Authorization": "Bearer " + client.post("/api/auth/guest", json={}).json()["token"]}
    assert client.get(f"/api/papers/{p['id']}", headers=other).status_code == 404  # papers are private


def test_bad_blueprint_is_a_clear_400(client, auth):
    r = client.post("/api/papers", json={"blueprint": {**MID, "total": 30}}, headers=auth)
    assert r.status_code == 400 and "add up to 25" in r.json()["detail"]
