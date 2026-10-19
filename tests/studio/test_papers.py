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


GATE = next(p for p in P.PRESETS if p["id"] == "gate")


def test_gate_paper_has_mcq_msq_nat_with_exact_keys(client, auth):
    p = client.post("/api/papers", json={"blueprint": {**GATE, "units": None}}, headers=auth).json()
    kinds = [q["kind"] for q in p["questions"]]
    assert p["total"] == 15 and {"mcq", "msq", "nat"} <= set(kinds)
    assert all(q["negative"] == (round(q["marks"] / 3, 2) if q["kind"] == "mcq" else 0) for q in p["questions"])
    full = client.get(f"/api/papers/{p['id']}?answers=true", headers=auth).json()
    for q in full["questions"]:
        if q["kind"] == "nat":
            assert isinstance(q["value"], (int, float))
        if q["kind"] == "msq":
            assert 1 <= len(q["correct"]) <= 3
    pdf = client.get(f"/api/papers/{p['id']}/pdf?answers=true", headers=auth)
    assert pdf.status_code == 200 and pdf.content[:5] == b"%PDF-"


def test_numeric_mcq_and_msq_keys_are_computed():
    import random

    from indiclearn.studio.courses import oscalc as C

    for s in range(40):
        m = C.gen_msq_numeric("pages", random.Random(s), 2, "medium", "page-replacement")
        if m:
            vals = {name: int(part.split(": ")[1].split()[0]) for part in m["answer"][0]["p"].split(". At most")[0].split("; ") for name in [part.split(": ")[0]]}
            k = int(m["blocks"][0]["p"].split("at most ")[1].split()[0])
            assert sorted(i for i, o in enumerate(m["options"]) if vals[o] <= k) == m["correct"]


def test_pattern_from_a_real_midterm_photo(client, auth):
    from pathlib import Path

    photo = Path(__file__).resolve().parents[1] / "fixtures" / "iiits_os_midterm_photo.jpg"
    r = client.post("/api/papers/pattern", files={"file": ("mid.jpg", photo.read_bytes(), "image/jpeg")}, headers=auth)
    assert r.status_code == 200, r.text
    bp = r.json()["blueprint"]
    assert bp["total"] == 25 and bp["minutes"] == 90
    assert sorted((x["count"], x["marks"], x["qtype"]) for x in bp["rows"]) == [(2, 4, "long"), (2, 4, "numerical"), (3, 3, "short")]
    assert {"u1", "u2", "u3"} <= set(bp["units"])
    assert client.post("/api/papers", json={"blueprint": bp}, headers=auth).status_code == 200


def test_pattern_rejects_unreadable_files(client, auth):
    r = client.post("/api/papers/pattern", files={"file": ("x.exe", b"MZ....", "application/octet-stream")}, headers=auth)
    assert r.status_code == 400
