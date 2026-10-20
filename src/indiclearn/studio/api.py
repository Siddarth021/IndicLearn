"""HTTP API (FastAPI).  All routes live under ``/api``; the static frontend is mounted by ``app.py``."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from indiclearn.studio.auth import AuthError
from indiclearn.studio.courses.papers import PaperError

STARTED = __import__("time").time()
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


class RenameIn(BaseModel):
    title: str = Field(max_length=200)


class PaperIn(BaseModel):
    blueprint: dict[str, Any]
    doc_ids: list[str] | None = None


class LangIn(BaseModel):
    lang: str


class TranslateIn(BaseModel):
    target: str
    text: str = Field(max_length=8000)
    reasoning: list[str] = Field(default_factory=list)
    key_terms: list[dict[str, Any]] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    beyond_notes: str = ""
    source: str = ""


class SpeakIn(BaseModel):
    text: str = Field(max_length=4000)
    lang: str = "en"


def build_router(svc: StudioService) -> APIRouter:
    r = APIRouter(prefix="/api")

    def wrap(fn, *a, **k):
        try:
            return fn(*a, **k)
        except AuthError as e:
            raise HTTPException(e.status, str(e)) from e
        except PaperError as e:
            raise HTTPException(404 if "not found" in str(e) else 400, str(e)) from e

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
        return {"ok": True, "app": "indiclearn", "started": STARTED}

    @r.get("/ai/status")
    def ai_status(test: int = 0) -> dict[str, Any]:
        """Which AI engines are configured, why one failed, and (with ?test=1) a live check of each."""
        from indiclearn.studio import neural, ocr

        brain = getattr(svc.models, "brain", None)
        out: dict[str, Any] = {"brain": brain.status() if brain else {"chain": ["off"]}, "ocr": ocr.status(),
                               "models": neural.status()}
        if test and brain is not None:
            out["live_test"] = brain.diagnose()
        return out

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

    @r.get("/me/stats")
    def me_stats(u: dict = Depends(current)) -> dict[str, Any]:
        return svc.me_stats(u["id"])

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

    @r.post("/asr")
    async def asr(file: UploadFile = File(...), lang: str = Form(default=""), doc_id: str = Form(default=""), u: dict = Depends(current)) -> dict[str, Any]:
        """Spoken doubt -> text. The browser then sends that text to /chat like a typed one.
        ``lang`` is the language the student chose (picks a Telugu-tuned model for Telugu speech)."""
        data = await file.read(25 * 1024 * 1024 + 1)
        return wrap(svc.transcribe, file.filename or "clip.webm", data, lang, u["id"], doc_id or None)

    @r.get("/languages")
    def languages() -> list[dict[str, str]]:
        """Every language a reply can be converted into (the per-reply dropdown)."""
        from indiclearn.studio.brain import TARGET_LANGS

        return [{"code": c, "name": en, "native": nat} for c, (en, nat) in TARGET_LANGS.items()]

    @r.post("/translate")
    def translate(b: TranslateIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.translate, b.model_dump(exclude={"target", "source"}), b.target, b.source)

    @r.post("/tts")
    def tts(b: SpeakIn, u: dict = Depends(current)) -> Response:
        return Response(wrap(svc.speak, b.text, b.lang), media_type="audio/wav")

    # ---- exam papers (printable practice papers per subject course)
    @r.get("/courses")
    def courses_list() -> list[dict[str, Any]]:
        from indiclearn.studio.courses.papers import courses

        return courses()

    @r.get("/courses/{subject}/lessons")
    def lessons_index(subject: str, u: dict = Depends(current)) -> dict[str, Any]:
        from indiclearn.studio.courses import lessons

        return wrap(lessons.index, svc.db, u["id"], subject)

    @r.post("/courses/{subject}/open")
    def course_open(subject: str, u: dict = Depends(current)) -> dict[str, Any]:
        """The course as a study document (for the chat, mind map, quiz and progress views)."""
        return wrap(svc.course_doc, u["id"], subject)

    @r.get("/courses/{subject}/lessons/{topic}")
    def lesson_get(subject: str, topic: str, lang: str = "en", seed: int | None = None, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.course_lesson, u["id"], subject, topic, lang, seed)

    @r.post("/courses/{subject}/lessons/{topic}/mark")
    def lesson_mark(subject: str, topic: str, status: str = "learned", u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.course_mark, u["id"], subject, topic, status)

    @r.post("/courses/{subject}/lessons/{topic}/check")
    def lesson_check(subject: str, topic: str, correct: bool = False, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.course_check, u["id"], subject, topic, correct)

    @r.get("/papers")
    def papers_list(u: dict = Depends(current)) -> list[dict[str, Any]]:
        return svc.exams.list(u["id"])

    @r.post("/papers/pattern")
    async def paper_pattern(file: UploadFile = File(...), subject: str | None = Form(default=None), u: dict = Depends(current)) -> dict[str, Any]:
        """A past question paper (PDF / Word / photo) -> a blueprint the student can edit."""
        from indiclearn.studio.courses.pattern import blueprint_from

        data = await file.read(15 * 1024 * 1024 + 1)
        if len(data) > 15 * 1024 * 1024:
            raise HTTPException(413, "That file is larger than 15 MB.")
        return wrap(blueprint_from, data, file.filename or "paper.jpg", getattr(svc.models, "brain", None), subject)

    @r.post("/papers")
    def paper_new(b: PaperIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.exams.new, u["id"], b.blueprint, b.doc_ids)

    @r.get("/papers/{pid}")
    def paper_get(pid: str, answers: bool = False, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.exams.get, u["id"], pid, answers)

    @r.get("/papers/{pid}/pdf")
    def paper_pdf(pid: str, answers: bool = False, u: dict = Depends(current)) -> Response:
        data, name = wrap(svc.exams.pdf, u["id"], pid, answers)
        return Response(data, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"'})

    @r.delete("/papers/{pid}")
    def paper_delete(pid: str, u: dict = Depends(current)) -> dict[str, bool]:
        svc.exams.delete(u["id"], pid)
        return {"ok": True}

    @r.post("/docs/sample/{key}")
    def sample(key: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.add_sample, u["id"], key)

    @r.get("/docs/{doc_id}")
    def doc(doc_id: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(lambda: svc._doc_public(svc._doc(u["id"], doc_id)))

    @r.put("/docs/{doc_id}")
    def rename(doc_id: str, b: RenameIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.rename_doc, u["id"], doc_id, b.title)

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

    @r.post("/docs/{doc_id}/chat/stream")
    def chat_stream(doc_id: str, b: ChatIn, u: dict = Depends(current)) -> StreamingResponse:
        """Same as /chat, but the answer arrives while it is being written (Server-Sent Events):
        ``delta`` {"text": answer so far} ... ``reset`` (a retry started) ... ``done`` {full reply} | ``error`` {"detail"}."""
        wrap(svc._doc, u["id"], doc_id)  # 404 / 401 before the stream starts
        q: queue.Queue = queue.Queue()
        last = {"t": ""}

        def on_delta(text: str | None) -> None:
            if text is None:
                last["t"] = ""
                q.put(("reset", {}))
            elif len(text) - len(last["t"]) >= 12 or text.endswith((".", "\n", "।")):  # don't flood the browser
                last["t"] = text
                q.put(("delta", {"text": text}))

        def work() -> None:
            try:
                q.put(("done", svc.chat(u["id"], doc_id, b.message, b.lang, b.focus_node, b.auto_lang, on_delta)))
            except AuthError as e:
                q.put(("error", {"detail": str(e), "status": e.status}))
            except Exception:  # pragma: no cover
                q.put(("error", {"detail": "Something went wrong while answering - please try again."}))

        threading.Thread(target=work, daemon=True).start()

        def events():
            yield ": ok\n\n"
            while True:
                try:
                    kind, data = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
                if kind in ("done", "error"):
                    return

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---- quizzes
    @r.post("/docs/{doc_id}/quiz")
    def quiz(doc_id: str, b: QuizIn, u: dict = Depends(current)) -> dict[str, Any]:
        out = wrap(svc.quiz_new, u["id"], doc_id, b.scope, b.n, b.lang)
        return out

    @r.post("/quiz/{quiz_id}/retry")
    def quiz_retry(quiz_id: str, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.quiz_retry, u["id"], quiz_id)

    @r.post("/quiz/{quiz_id}/answer")
    def answer(quiz_id: str, b: AnswerIn, u: dict = Depends(current)) -> dict[str, Any]:
        return wrap(svc.quiz_answer, u["id"], quiz_id, b.question_id, b.choice, b.lang)

    return r
