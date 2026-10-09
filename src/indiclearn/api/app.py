"""FastAPI Application Factory for IndicLearn Student Learning Dashboard (Phase 8).

Wires the REST API routes, static asset hosting for the frontend SPA,
CORS middleware, and dependency injection of the DashboardApplicationService.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from indiclearn.api.routes import router
from indiclearn.api.service import DashboardApplicationService

logger = logging.getLogger(__name__)


def create_app(
    dashboard_service: DashboardApplicationService | None = None,
    frontend_dir: str | Path | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="IndicLearn Student Learning Dashboard",
        description="Phase 8 presentation and interaction layer for IndicLearn research architecture.",
        version="0.8.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Enable CORS for local dev flexibility
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach application service to app state
    if dashboard_service is None:
        dashboard_service = DashboardApplicationService.create_default()
    app.state.dashboard_service = dashboard_service

    # Mount API routes under /api and also at root prefix for flexibility
    app.include_router(router, prefix="/api", tags=["API"])
    app.include_router(router, tags=["API-Direct"])

    # Locate frontend directory
    if frontend_dir is None:
        frontend_path = Path(__file__).resolve().parents[3] / "frontend"
    else:
        frontend_path = Path(frontend_dir)

    if frontend_path.exists() and frontend_path.is_dir():
        # Serve static assets
        app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon() -> Response:
            return Response(status_code=204)

        @app.get("/", include_in_schema=False)
        def serve_index() -> FileResponse:
            index_file = frontend_path / "index.html"
            if index_file.exists():
                return FileResponse(str(index_file))
            return FileResponse(str(frontend_path / "index.html"))

    else:
        logger.info("Frontend directory not found at %s. Running in API-only mode.", frontend_path)

    return app


# Default app instance for ASGI servers (uvicorn src.indiclearn.api.app:app)
app = create_app()
