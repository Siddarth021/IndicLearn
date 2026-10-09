"""FastAPI routes for Student Dashboard and Academic Knowledge Map (Phase 8).

Exposes REST endpoints that act as a strict read/write VIEW over backend research domains:
- Academic Knowledge Graph
- Student Knowledge Model
- Adaptive Assessment Engine
- Personalized Recommendation Planner
- Academic Corpus Provenance
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from indiclearn.api.schemas import (
    AssessmentResultView,
    AssessmentSessionView,
    ChunkDetailView,
    ConceptDetailView,
    KnowledgeMapView,
    RecommendationCardView,
    StartAssessmentRequest,
    StudentDashboardOverview,
    SubjectSummary,
    SubmitResponseRequest,
)
from indiclearn.api.service import DashboardApplicationService

router = APIRouter()


def get_dashboard_service(request: Request) -> DashboardApplicationService:
    """Dependency provider for application service."""
    if hasattr(request.app.state, "dashboard_service"):
        return request.app.state.dashboard_service
    # Fallback to default singleton if not explicitly configured on app
    if not hasattr(get_dashboard_service, "_default_instance"):
        get_dashboard_service._default_instance = DashboardApplicationService.create_default()
    return get_dashboard_service._default_instance


ServiceDep = Annotated[DashboardApplicationService, Depends(get_dashboard_service)]
FileDep = Annotated[UploadFile, File(description="Curriculum document (PDF, PPTX, TXT, MD)")]
SubjectFormDep = Annotated[str, Form(description="Target academic subject ID or name")]
SubjectNameDep = Annotated[str | None, Form(description="Optional display title or paper name")]


# ---------------------------------------------------------------------------
# Subject & Academic Knowledge Map
# ---------------------------------------------------------------------------


@router.get("/subjects", response_model=list[SubjectSummary])
def get_subjects(
    service: ServiceDep,
) -> list[SubjectSummary]:
    """Retrieve list of academic subjects."""
    return service.get_subjects()


@router.get("/subjects/{subject_id}/graph", response_model=KnowledgeMapView)
def get_knowledge_map(
    subject_id: str,
    service: ServiceDep,
    student_id: str = Query("default_student", description="Student ID for knowledge overlay"),
) -> KnowledgeMapView:
    """Retrieve academic concept graph with student knowledge overlay."""
    return service.get_knowledge_map(student_id=student_id, subject_id=subject_id)


@router.get("/concepts", response_model=list[dict[str, Any]])
def list_concepts(
    service: ServiceDep,
    subject_id: str = Query("signals_and_systems"),
) -> list[dict[str, Any]]:
    """List all concepts for a subject."""
    return service.list_concepts(subject_id=subject_id)


@router.get("/concepts/{concept_id}", response_model=ConceptDetailView)
def get_concept_details(
    concept_id: str,
    service: ServiceDep,
    student_id: str = Query("default_student"),
    subject_id: str = Query("signals_and_systems"),
) -> ConceptDetailView:
    """Retrieve granular concept details including student state and recommendation."""
    try:
        return service.get_concept_details(
            student_id=student_id,
            concept_id=concept_id,
            subject_id=subject_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Student Analytics & Recommendations
# ---------------------------------------------------------------------------


@router.get("/students/{student_id}/knowledge", response_model=KnowledgeMapView)
def get_student_knowledge(
    student_id: str,
    service: ServiceDep,
    subject_id: str = Query("signals_and_systems"),
) -> KnowledgeMapView:
    """Retrieve student knowledge map overlay for a subject."""
    return service.get_knowledge_map(student_id=student_id, subject_id=subject_id)


@router.get("/students/{student_id}/dashboard", response_model=StudentDashboardOverview)
def get_student_dashboard(
    student_id: str,
    service: ServiceDep,
    subject_id: str = Query("signals_and_systems"),
) -> StudentDashboardOverview:
    """Retrieve high-level student learning analytics overview."""
    return service.get_student_dashboard(student_id=student_id, subject_id=subject_id)


@router.get(
    "/students/{student_id}/recommendations", response_model=list[RecommendationCardView]
)
def get_recommendations(
    student_id: str,
    service: ServiceDep,
    subject_id: str = Query("signals_and_systems"),
    top_k: int = Query(5, ge=1, le=20),
) -> list[RecommendationCardView]:
    """Retrieve personalized recommendations produced by Phase 7B RecommendationService."""
    return service.get_recommendations(student_id=student_id, subject_id=subject_id, top_k=top_k)


@router.get("/students/{student_id}/concepts/{concept_id}", response_model=ConceptDetailView)
def get_student_concept_details(
    student_id: str,
    concept_id: str,
    service: ServiceDep,
    subject_id: str = Query("signals_and_systems"),
) -> ConceptDetailView:
    """Alias for student concept details."""
    try:
        return service.get_concept_details(
            student_id=student_id,
            concept_id=concept_id,
            subject_id=subject_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Adaptive Assessment Integration
# ---------------------------------------------------------------------------


@router.post("/assessment/sessions", response_model=AssessmentSessionView)
def start_assessment_session(
    payload: StartAssessmentRequest,
    service: ServiceDep,
) -> AssessmentSessionView:
    """Initiate an adaptive assessment session."""
    return service.start_assessment(
        student_id=payload.student_id,
        subject_id=payload.subject_id,
        concept_id=payload.concept_id,
        requested_language=payload.requested_language,
    )


@router.post(
    "/assessment/sessions/{session_id}/responses", response_model=AssessmentResultView
)
def submit_assessment_response(
    session_id: str,
    payload: SubmitResponseRequest,
    service: ServiceDep,
) -> AssessmentResultView:
    """Submit answer to active item, updating student knowledge state in real time."""
    try:
        return service.submit_assessment_response(
            session_id=session_id,
            item_id=payload.item_id,
            answer=payload.selected_answer,
            response_time_ms=payload.response_time_ms,
            confidence_if_provided=payload.confidence_if_provided,
            requested_language=payload.requested_language,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/assessment/sessions/{session_id}", response_model=AssessmentSessionView)
def get_assessment_session(
    session_id: str,
    service: ServiceDep,
) -> AssessmentSessionView:
    """Retrieve existing assessment session status."""
    session = service.get_assessment_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Assessment session '{session_id}' not found.",
        )
    return session


# ---------------------------------------------------------------------------
# Source Provenance & Demo Fixtures
# ---------------------------------------------------------------------------


@router.get("/chunks/{chunk_id}", response_model=ChunkDetailView)
def get_chunk_detail(
    chunk_id: str,
    service: ServiceDep,
) -> ChunkDetailView:
    """Retrieve grounded academic textbook chunk text and page/section citation."""
    chunk = service.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Curriculum chunk '{chunk_id}' not found.",
        )
    return chunk


@router.post("/demo/load")
def load_demo_data(
    service: ServiceDep,
    student_id: str = Query(..., description="Student ID to populate with demo records"),
) -> dict[str, Any]:
    """Load isolated demonstration data for UI testing.

    Explicit Policy: Never mixed with real student evaluation benchmarks.
    """
    return service.load_demo_data(student_id=student_id)


@router.post("/documents/upload")
def upload_document(
    service: ServiceDep,
    file: FileDep,
    subject_id: SubjectFormDep = "signals_and_systems",
    subject_name: SubjectNameDep = None,
) -> dict[str, Any]:
    """Upload and ingest a document into the curriculum knowledge graph and chunk store."""
    import shutil
    from pathlib import Path

    allowed_exts = {".pdf", ".pptx", ".ppt", ".txt", ".md"}
    filename = file.filename or "uploaded_document"
    ext = Path(filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported extensions: {', '.join(sorted(allowed_exts))}",
        )

    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_target = upload_dir / filename

    try:
        with open(temp_target, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = service.ingest_document(
            file_path=temp_target,
            filename=filename,
            subject_id=subject_id,
            subject_name=subject_name or subject_id,
        )

        if result.get("status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=result.get("message", "Document ingestion failed."),
            )

        return result
    finally:
        file.file.close()

