"""OS course lessons: index with progress, a lesson with a live exactly-solved example, marking learned."""

from __future__ import annotations


def test_course_index_lesson_and_progress(client, auth):
    idx = client.get("/api/courses/os/lessons", headers=auth).json()
    assert idx["total"] == 38 and all(t["has_lesson"] for u in idx["units"] for t in u["topics"])
    r = client.get("/api/courses/os/lessons/round-robin?seed=7", headers=auth).json()
    assert r["lesson"]["summary"] and len(r["lesson"]["sections"]) >= 2
    assert r["example"]["blocks"] and r["example"]["answer"]  # numerical topic: live solved example
    assert r["prev"]["id"] == "priority-sched" and r["next"]["id"] == "mlq"
    again = client.get("/api/courses/os/lessons/round-robin?seed=8", headers=auth).json()
    assert again["example"]["blocks"] != r["example"]["blocks"]  # "New example" really is new
    theory = client.get("/api/courses/os/lessons/threads", headers=auth).json()
    assert theory["example"] is None and theory["lesson"]["example"]["body"]
    assert client.post("/api/courses/os/lessons/threads/mark?status=learned", headers=auth).json()["status"] == "learned"
    idx = client.get("/api/courses/os/lessons", headers=auth).json()
    assert idx["done"] == 1
    assert client.get("/api/courses/os/lessons/nope", headers=auth).status_code == 404


def test_telugu_falls_back_to_english_without_ai(client, auth, monkeypatch):
    from indiclearn.studio.courses import lessons

    real = lessons.lessons
    missing = "paging"
    monkeypatch.setattr(lessons, "lessons", lambda subject="os": {k: ({**v, "te": None} if k == missing else v) for k, v in real(subject).items()})
    r = client.get(f"/api/courses/os/lessons/{missing}?lang=te", headers=auth).json()
    assert r["lang"] == "en" and r["lesson"]["summary"]  # no AI engine in tests: English, never an error
    done = next((t for t, d in lessons.lessons().items() if d.get("te")), None)
    if done:
        assert client.get(f"/api/courses/os/lessons/{done}?lang=te", headers=auth).json()["lang"] == "te"
