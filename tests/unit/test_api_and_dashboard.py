"""Unit tests for FastAPI endpoints and Dashboard Application Layer (Phase 8)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from indiclearn.api.app import create_app
from indiclearn.api.service import DashboardApplicationService
from indiclearn.student_model.models import EvidenceType


@pytest.fixture
def test_client() -> TestClient:
    """Fixture providing a FastAPI TestClient configured with default test fixtures."""
    service = DashboardApplicationService.create_default()
    app = create_app(dashboard_service=service)
    return TestClient(app)


def test_get_subjects(test_client: TestClient) -> None:
    """Test GET /api/subjects returns subjects list."""
    response = test_client.get("/api/subjects")
    assert response.status_code == 200
    subjects = response.json()
    assert len(subjects) >= 1
    assert subjects[0]["subject_id"] == "signals_and_systems"
    assert subjects[0]["concept_count"] >= 5


def test_get_knowledge_map(test_client: TestClient) -> None:
    """Test GET /api/subjects/{subject_id}/graph returns knowledge map with overlay."""
    response = test_client.get("/api/subjects/signals_and_systems/graph?student_id=test_student")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) >= 5
    assert len(data["edges"]) >= 5
    first_node = data["nodes"][0]
    assert "knowledge_score" in first_node
    assert "confidence" in first_node
    assert "mastery_status" in first_node


def test_get_concept_details(test_client: TestClient) -> None:
    """Test GET /api/concepts/{concept_id} returns complete concept details."""
    response = test_client.get(
        "/api/concepts/cpt_signals_and_systems_lti_system?student_id=test_student"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["concept_id"] == "cpt_signals_and_systems_lti_system"
    assert data["canonical_name"] == "LTI System"
    assert "prerequisites" in data
    assert "dependents" in data
    assert "explanation_reason" in data


def test_get_concept_not_found(test_client: TestClient) -> None:
    """Test GET /api/concepts/{concept_id} returns 404 for nonexistent concept."""
    response = test_client.get("/api/concepts/nonexistent_concept_123")
    assert response.status_code == 404
    assert "detail" in response.json()


def test_student_dashboard_overview(test_client: TestClient) -> None:
    """Test GET /api/students/{student_id}/dashboard returns analytics summary."""
    response = test_client.get("/api/students/fresh_student/dashboard")
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "fresh_student"
    assert data["total_concepts_tracked"] >= 5
    assert data["concepts_assessed"] == 0
    assert data["unknown_count"] == data["total_concepts_tracked"]


def test_student_recommendations(test_client: TestClient) -> None:
    """Test GET /api/students/{student_id}/recommendations returns prioritized cards."""
    response = test_client.get("/api/students/fresh_student/recommendations?top_k=3")
    assert response.status_code == 200
    recs = response.json()
    assert len(recs) <= 3
    if recs:
        assert "recommendation_id" in recs[0]
        assert "action_type" in recs[0]
        assert "priority_score" in recs[0]
        assert "reason_codes" in recs[0]


def test_assessment_lifecycle(test_client: TestClient) -> None:
    """Test end-to-end adaptive assessment session: start, get, submit."""
    # 1. Start Session
    start_resp = test_client.post(
        "/api/assessment/sessions",
        json={"student_id": "assess_student", "subject_id": "signals_and_systems"},
    )
    assert start_resp.status_code == 200
    session_data = start_resp.json()
    session_id = session_data["session_id"]
    assert session_data["is_active"] is True
    assert session_data["current_item"] is not None
    item = session_data["current_item"]

    # 2. Get Session by ID
    get_resp = test_client.get(f"/api/assessment/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id

    # 3. Submit Response
    selected_ans = item["options"][0] if item.get("options") else "True"
    sub_resp = test_client.post(
        f"/api/assessment/sessions/{session_id}/responses",
        json={
            "session_id": session_id,
            "item_id": item["item_id"],
            "selected_answer": selected_ans,
        },
    )
    assert sub_resp.status_code == 200
    result_data = sub_resp.json()
    assert "correct" in result_data
    assert "feedback" in result_data
    assert "updated_knowledge_score" in result_data
    assert "updated_confidence" in result_data


def test_chunk_inspection(test_client: TestClient) -> None:
    """Test inspecting academic chunk and missing chunk 404."""
    # Valid chunk
    resp = test_client.get("/api/chunks/CHUNK_SS_001")
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunk_id"] == "CHUNK_SS_001"
    assert "text" in data
    assert len(data["text"]) > 10

    # Nonexistent chunk
    resp_missing = test_client.get("/api/chunks/NONEXISTENT_CHUNK_000")
    assert resp_missing.status_code == 404


def test_student_isolation(test_client: TestClient) -> None:
    """Test that student state updates are completely isolated per student_id."""
    student_a = "student_iso_1"
    student_b = "student_iso_2"
    cid = "cpt_signals_and_systems_lti_system"

    # Populate Student A
    service = test_client.app.state.dashboard_service
    for _ in range(4):
        service.knowledge_service.record_interaction(
            student_id=student_a,
            concept_id=cid,
            event_type=EvidenceType.QUIZ_CORRECT,
            score=1.0,
            source="test_fixture",
        )

    # Inspect Student A
    resp_a = test_client.get(f"/api/concepts/{cid}?student_id={student_a}")
    assert resp_a.json()["mastery_status"] == "MASTERED"
    assert resp_a.json()["evidence_count"] == 4

    # Inspect Student B
    resp_b = test_client.get(f"/api/concepts/{cid}?student_id={student_b}")
    assert resp_b.json()["mastery_status"] == "UNKNOWN"
    assert resp_b.json()["evidence_count"] == 0


def test_demo_data_loader(test_client: TestClient) -> None:
    """Test loading demo fixture data."""
    student_demo = "student_demo_user"
    resp = test_client.post(f"/api/demo/load?student_id={student_demo}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "SUCCESS"
    assert len(data["demo_concepts_modified"]) >= 4

    # Verify student now has evidence
    dash_resp = test_client.get(f"/api/students/{student_demo}/dashboard")
    assert dash_resp.status_code == 200
    assert dash_resp.json()["concepts_with_evidence"] >= 3


def test_upload_document_arbitrary_topic(test_client: TestClient) -> None:
    """Test uploading an arbitrary research paper / custom topic and verifying its dynamic knowledge map."""
    paper_content = (
        b"# Attention Mechanisms\n\n"
        b"Attention is defined as a mechanism that maps a query and a set of key-value pairs to an output.\n\n"
        b"Multi-Head Attention is a method that computes attention in parallel.\n\n"
        b"Transformer Architecture is a deep learning model based on Self-Attention.\n"
    )
    files = {"file": ("attention_paper.txt", paper_content, "text/plain")}
    data = {
        "subject_name": "Attention Is All You Need",
        "subject_id": "Attention Is All You Need",
    }

    resp = test_client.post("/api/documents/upload", files=files, data=data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "SUCCESS"
    assert result["subject_id"] == "attention_is_all_you_need"
    assert result["subject_name"] == "Attention Is All You Need"
    assert result["chunks_indexed"] >= 1

    # Verify subject is dynamically listed
    sub_resp = test_client.get("/api/subjects")
    assert sub_resp.status_code == 200
    subject_ids = [s["subject_id"] for s in sub_resp.json()]
    assert "attention_is_all_you_need" in subject_ids

    # Verify knowledge map can be retrieved for the new subject
    graph_resp = test_client.get("/api/subjects/attention_is_all_you_need/graph")
    assert graph_resp.status_code == 200
    graph_data = graph_resp.json()
    assert graph_data["subject_id"] == "attention_is_all_you_need"
    assert "nodes" in graph_data
    assert "edges" in graph_data


def test_upload_invalid_file_extension(test_client: TestClient) -> None:
    """Test rejection of unsupported file types."""
    files = {"file": ("script.exe", b"binary_data", "application/octet-stream")}
    resp = test_client.post("/api/documents/upload", files=files)
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.json()["detail"]

