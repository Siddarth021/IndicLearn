"""ASGI app factory:  ``uvicorn indiclearn.studio.app:app``  (or ``python -m indiclearn.studio``)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from indiclearn.studio import __version__
from indiclearn.studio.api import build_router
from indiclearn.studio.config import Settings
from indiclearn.studio.service import StudioService


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings()
    svc = StudioService(s)
    app = FastAPI(title="IndicLearn Studio", version=__version__, docs_url="/api/swagger", openapi_url="/api/openapi.json")
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["*"], allow_headers=["*"])
    app.state.service = svc
    app.include_router(build_router(svc))
    fe = s.frontend_dir or Path(__file__).resolve().parents[3] / "frontend"
    if fe.is_dir():
        app.mount("/static", StaticFiles(directory=fe), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(fe / "index.html")

    return app


def __getattr__(name: str):  # lazy ``app`` so importing this module has no side effects
    if name == "app":
        global _APP
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)


_APP: FastAPI | None = None
