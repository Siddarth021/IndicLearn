"""HTTP API (FastAPI).  All routes live under ``/api``; the static frontend is mounted by ``app.py``."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from indiclearn.studio.auth import AuthError
from indiclearn.studio.service import StudioService


class Creds(BaseModel):
    username: str
    password: str
    display_name: str | None = None
    lang: str = "en"


class Guest(BaseModel):
    lang: str = "en"


class ChatIn(BaseModel):
    message: str = Field(max_length=1200)
    lang: str = "en"
    focus_node: str | None = None
    auto_lang: bool = True


class QuizIn(BaseModel):
    scope: str | None = None
    n: int = 5
    lang: str = "en"


class AnswerIn(BaseModel):
    question_id: str
    choice: int
    lang: str = "en"


class MarkIn(BaseModel):
    node_id: str
    kind: str


class LangIn(BaseModel):
    lang: str


def build_router(svc: StudioService) -> APIRouter:
    r = APIRouter(prefix="/api")

    def wrap(fn, *a, **k):
        try:
            return fn(*a, **k)
        except AuthError as e:
            raise HTTPException(e.status, str(e)) from e

    def current(authorization: str | None = Header(default=None), token: str | None = Query(default=None)) -> dict[str, Any]:
        tok = None
        if authorization and authorization.lower().startswith("bearer "):
            tok = authorization[7:].strip()
        tok = tok or token
        user = svc.auth.user_for_token(tok)
        if user is None:
            raise HTTPException(401, "Please sign in.")
        user["_token"] = tok
        return user

    @r.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @r.get("/info")
    def info() -> dict[str, Any]:
        return svc.info()

    # ---- auth
    @r.post("/auth/register")
    def register(b: Creds) -> dict[str, Any]:
        if not svc.s.allow_registration:
            raise HTTPException(403, "Registration is disabled.")
        return wrap(svc.auth.register, b.username, b.password, b.display_name, b.lang)

    @r.post("/auth/login")
    def login(b: Creds) -> dict[str, Any]:
        return wrap(svc.auth.login, b.username, b.password)

    @r.post("/auth/guest")
    def guest(b: Guest) -> dict[str, Any]:
        if not svc.s.allow_guest:
            raise HTTPException(403, "Guest mode is disabled.")
        return svc.auth.guest(b.lang)

    @r.get("/me")
    def me(u: dict = Depends(current)) -> dict[str, Any]:
        return {k: v for k, v in u.items() if k != "_token"}

    @r.post("/auth/logout")
    def logout(u: dict = Depends(current)) -> dict[str, bool]:
        svc.auth.logout(u["_token"])
        return {"ok": True}

    @r.put("/me/lang")
    def set_lang(b: LangIn, u: dict = Depends(current)) -> dict[str, bool]:
        svc.auth.set_lang(u["id"], b.lang)
        return {"ok": True}

    # ---- documents
    @r.get("/docs")
    def docs(u: dict = Depends(current)) -> list[dict[str, Any]]:
        return svc.list_docs(u["id"])

    @r.post("/docs")
    async def upload(file: UploadFile = File(...), u: dict = Depends(current)) -> dict[str, Any]:
        limit = svc.s.max_upload_mb * 1024 * 1024
        data = await file.read(limit + 1)
        return wrap(svc.add_upload, u["id"], file.filename or "upload", data)

    @r.post("/docs/sample/{key}")
    def sample(key: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.add_sample, u["id"], key)

    @r.get("/docs/{doc_id}")
    def doc(doc_id: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(lambda: svc._doc_public(svc._doc(u["id"], doc_id)))

    @r.delete("/docs/{doc_id}")
    def delete(doc_id: str, u: dict = Depends(current)) -> dict[str, bool]:
        wrap(svc.delete_doc, u["id"], doc_id)
        return {"ok": True}

    @r.get("/docs/{doc_id}/file")
    def original(doc_id: str, u: dict = Depends(current)) -> FileResponse:
        path, name = wrap(svc.file_path, u["id"], doc_id)
        mt = {".pdf": "application/pdf"}.get(path.suffix.lower(), "application/octet-stream")
        return FileResponse(path, media_type=mt, filename=name, content_disposition_type="inline")

    @r.get("/docs/{doc_id}/reader")
    def reader(doc_id: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.reader, u["id"], doc_id)

    @r.get("/docs/{doc_id}/graph")
    def graph(doc_id: str, mode: str = "default", u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.graph, u["id"], doc_id, "personal" if mode == "personal" else "default")

    @r.get("/docs/{doc_id}/progress")
    def progress(doc_id: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.progress, u["id"], doc_id)

    @r.post("/docs/{doc_id}/reset")
    def reset(doc_id: str, u: dict = Depends(current)) -> dict[str, bool]:
        wrap(svc.reset_progress, u["id"], doc_id)
        return {"ok": True}

    @r.post("/docs/{doc_id}/mark")
    def mark(doc_id: str, b: MarkIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.mark, u["id"], doc_id, b.node_id, b.kind)

    # ---- tutor
    @r.get("/docs/{doc_id}/chat")
    def history(doc_id: str, u: dict = Depends(current)) -> list[dict[str, Any]]:
        return wrap(svc.history, u["id"], doc_id)

    @r.delete("/docs/{doc_id}/chat")
    def clear(doc_id: str, u: dict = Depends(current)) -> dict[str, bool]:
        wrap(svc.clear_history, u["id"], doc_id)
        return {"ok": True}

    @r.post("/docs/{doc_id}/chat")
    def chat(doc_id: str, b: ChatIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.chat, u["id"], doc_id, b.message, b.lang, b.focus_node, b.auto_lang)

    # ---- quizzes
    @r.post("/docs/{doc_id}/quiz")
    def quiz(doc_id: str, b: QuizIn, u: dict = Depends(current)) -> dict[str, Any]:
        out = wrap(svc.quiz_new, u["id"], doc_id, b.scope, b.n, b.lang)
        return out

    @r.post("/quiz/{quiz_id}/answer")
    def answer(quiz_id: str, b: AnswerIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.quiz_answer, u["id"], quiz_id, b.question_id, b.choice, b.lang)

    return r
