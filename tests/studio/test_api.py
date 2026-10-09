import time

import pytest
from fastapi.testclient import TestClient

from indiclearn.studio.app import create_app
from indiclearn.studio.config import Settings


@pytest.fixture()
def client(tmp_path):
    s = Settings()
    s.data_dir = tmp_path
    s.max_upload_mb = 1
    return TestClient(create_app(s))


def guest(c):
    return {"Authorization": "Bearer " + c.post("/api/auth/guest", json={}).json()["token"]}


def ready_doc(c, h, key="en"):
    d = c.post(f"/api/docs/sample/{key}", headers=h).json()
    for _ in range(80):
        d = c.get(f"/api/docs/{d['id']}", headers=h).json()
        if d["status"] != "processing":
            break
        time.sleep(0.25)
    assert d["status"] == "ready", d
    return d["id"]


def test_requires_auth(client):
    assert client.get("/api/docs").status_code == 401


def test_register_login(client):
    r = client.post("/api/auth/register", json={"username": "asha", "password": "secret1"})
    assert r.status_code == 200
    assert client.post("/api/auth/register", json={"username": "asha", "password": "secret1"}).status_code == 409
    assert client.post("/api/auth/login", json={"username": "asha", "password": "bad"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "asha", "password": "secret1"}).status_code == 200


def test_upload_validation(client):
    h = guest(client)
    assert client.post("/api/docs", headers=h, files={"file": ("x.exe", b"abc")}).status_code == 415
    assert client.post("/api/docs", headers=h, files={"file": ("x.md", b"")}).status_code == 400
    assert client.post("/api/docs", headers=h, files={"file": ("big.txt", b"a" * (1024 * 1024 + 5))}).status_code == 413


def test_full_student_flow(client):
    h = guest(client)
    i = ready_doc(client, h)
    r = client.post(f"/api/docs/{i}/chat", headers=h, json={"message": "What is aliasing?", "lang": "en"}).json()
    assert r["grounded"] and r["touched"]
    r = client.post(f"/api/docs/{i}/chat", headers=h, json={"message": "who invented pizza", "lang": "en"}).json()
    assert r["new_doubt"]
    r = client.post(f"/api/docs/{i}/chat", headers=h, json={"message": "quiz me on sampling", "lang": "en"}).json()
    q = r["quiz"]
    assert q and "answer" not in q["questions"][0]
    a = client.post(f"/api/quiz/{q['quiz_id']}/answer", headers=h, json={"question_id": q["questions"][0]["id"], "choice": 0}).json()
    assert "correct" in a and a["state"]["evidence"] >= 1
    again = client.post(f"/api/quiz/{q['quiz_id']}/answer", headers=h, json={"question_id": q["questions"][0]["id"], "choice": 0})
    assert again.status_code == 409
    g = client.get(f"/api/docs/{i}/graph?mode=personal", headers=h).json()
    assert any(n["type"] == "doubt" for n in g["nodes"])
    assert any(n.get("mastery", {}).get("status") != "new" for n in g["nodes"] if n["type"] == "concept")
    d = client.get(f"/api/docs/{i}/graph", headers=h).json()
    assert not any(n["type"] == "doubt" for n in d["nodes"])  # the default map never changes
    p = client.get(f"/api/docs/{i}/progress", headers=h).json()
    assert p["doubts"] == 1 and p["quiz_answered"] == 1
    assert client.get(f"/api/docs/{i}/reader", headers=h).json()["pages"]


def test_users_are_isolated(client):
    a, b = guest(client), guest(client)
    i = ready_doc(client, a)
    for path in (f"/api/docs/{i}", f"/api/docs/{i}/graph", f"/api/docs/{i}/reader", f"/api/docs/{i}/progress", f"/api/docs/{i}/file"):
        assert client.get(path, headers=b).status_code == 404
    assert client.post(f"/api/docs/{i}/chat", headers=b, json={"message": "hi"}).status_code == 404
    assert client.delete(f"/api/docs/{i}", headers=b).status_code == 404


@pytest.mark.parametrize("key,q", [("te", "సిగ్నల్ అంటే ఏమిటి?"), ("hi", "सैंपलिंग क्या है?")])
def test_indic_documents_are_understood(client, key, q):
    h = guest(client)
    i = ready_doc(client, h, key)
    doc = client.get(f"/api/docs/{i}", headers=h).json()
    assert doc["lang"] == key and doc["stats"]["concepts"] >= 5
    r = client.post(f"/api/docs/{i}/chat", headers=h, json={"message": q, "lang": key}).json()
    assert r["grounded"], r
