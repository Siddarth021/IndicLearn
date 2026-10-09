"""Runtime configuration for IndicLearn Studio (12-factor: everything via environment variables)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("INDICLEARN_DATA_DIR", "data/studio")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("INDICLEARN_MAX_UPLOAD_MB", "40")))
    max_docs_per_user: int = field(default_factory=lambda: int(os.environ.get("INDICLEARN_MAX_DOCS", "25")))
    token_ttl_days: int = field(default_factory=lambda: int(os.environ.get("INDICLEARN_TOKEN_TTL_DAYS", "30")))
    allow_guest: bool = field(default_factory=lambda: _bool("INDICLEARN_ALLOW_GUEST", True))
    allow_registration: bool = field(default_factory=lambda: _bool("INDICLEARN_ALLOW_REGISTRATION", True))
    cors_origins: list[str] = field(default_factory=lambda: [o.strip() for o in os.environ.get("INDICLEARN_CORS_ORIGINS", "").split(",") if o.strip()])
    # --- local model backends (all optional; the app works with none of them) ---
    llm_backend: str = field(default_factory=lambda: os.environ.get("INDICLEARN_LLM_BACKEND", "extractive").lower())  # extractive | ollama | transformers
    llm_model: str = field(default_factory=lambda: os.environ.get("INDICLEARN_LLM_MODEL", "gemma3:4b"))
    ollama_url: str = field(default_factory=lambda: os.environ.get("INDICLEARN_OLLAMA_URL", "http://127.0.0.1:11434"))
    embedding_backend: str = field(default_factory=lambda: os.environ.get("INDICLEARN_EMBEDDING_BACKEND", "none").lower())  # none | sentence_transformers
    embedding_model: str = field(default_factory=lambda: os.environ.get("INDICLEARN_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    translator_backend: str = field(default_factory=lambda: os.environ.get("INDICLEARN_TRANSLATOR_BACKEND", "none").lower())  # none | nllb
    translator_model: str = field(default_factory=lambda: os.environ.get("INDICLEARN_TRANSLATOR_MODEL", "facebook/nllb-200-distilled-600M"))
    local_files_only: bool = field(default_factory=lambda: _bool("INDICLEARN_LOCAL_FILES_ONLY", False))
    frontend_dir: Path | None = field(default_factory=lambda: Path(os.environ["INDICLEARN_FRONTEND_DIR"]) if os.environ.get("INDICLEARN_FRONTEND_DIR") else None)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "indiclearn.sqlite3"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
