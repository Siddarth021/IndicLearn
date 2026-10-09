"""Configuration loading.

Purpose: load configs/config.yaml into validated settings; allow env overrides of the form
``INDICLEARN_<SECTION>__<KEY>=value`` (value parsed as YAML so ints/bools/lists work).
Nothing tunable is hard-coded elsewhere in the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class AppSettings(BaseModel):
    log_level: str = "INFO"


class DatabaseSettings(BaseModel):
    url: str = "sqlite:///data/indiclearn.sqlite3"


class IngestionSettings(BaseModel):
    supported_extensions: list[str] = Field(default_factory=lambda: [".txt", ".md", ".pdf", ".pptx"])
    max_file_mb: int = 50


class PreprocessingSettings(BaseModel):
    unicode_form: str = "NFC"
    join_hyphenated_lines: bool = True


class ChunkingSettings(BaseModel):
    strategy: str = "recursive_char"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)


class EmbeddingSettings(BaseModel):
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    dimension: int = 384
    batch_size: int = 32
    device: str | None = None
    cache_path: str | None = "data/cache/embeddings.json"


class VectorStoreSettings(BaseModel):
    provider: str = "faiss"
    index_path: str = "data/indices/signals_systems"


class RetrievalSettings(BaseModel):
    top_k: int = 5
    strategy: str = "bm25"
    hybrid_strategy: str = "rrf"
    weight_lexical: float = 0.5
    weight_dense: float = 0.5
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


class StudentModelSettings(BaseModel):
    prior_knowledge_score: float = 0.50
    prior_confidence: float = 0.10
    prior_weight: float = 1.0
    evidence_weights: dict[str, float] = Field(
        default_factory=lambda: {"strong": 1.00, "medium": 0.40, "weak": 0.10}
    )
    confidence_deltas: dict[str, float] = Field(
        default_factory=lambda: {"strong": 0.20, "medium": 0.08, "weak": 0.02}
    )
    recency_half_life_days: float = 30.0
    mastery_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "mastered_min_score": 0.80,
            "mastered_min_confidence": 0.60,
            "developing_min_score": 0.50,
            "weak_max_score": 0.40,
            "weak_min_confidence": 0.40,
        }
    )


class AssessmentSettings(BaseModel):
    enabled: bool = True
    default_subject_id: str = "signals_and_systems"
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "weakness": 0.35,
            "uncertainty": 0.25,
            "prerequisite": 0.20,
            "coverage": 0.12,
            "recency": 0.08,
        }
    )
    recency_decay_minutes: float = 60.0
    repeat_avoidance_window: int = 5
    exclude_mastered: bool = True
    default_language: str = "en"


class RecommendationSettings(BaseModel):
    enabled: bool = True
    default_subject_id: str = "signals_and_systems"
    top_k: int = 3
    priority_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "weakness": 0.35,
            "uncertainty": 0.15,
            "prerequisite_deficit": 0.15,
            "downstream_blocker": 0.15,
            "review_need": 0.10,
            "assessment_gap": 0.10,
            "recent_action_penalty": 0.25,
        }
    )
    mastered_damping_factor: float = 0.35
    recent_decay_minutes: float = 30.0
    review_half_life_hours: float = 72.0
    default_language: str = "en"


class Settings(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    student_model: StudentModelSettings = Field(default_factory=StudentModelSettings)
    assessment: AssessmentSettings = Field(default_factory=AssessmentSettings)
    recommendation: RecommendationSettings = Field(default_factory=RecommendationSettings)



ENV_PREFIX = "INDICLEARN_"


def _apply_env_overrides(raw: dict[str, Any], environ: dict[str, str]) -> dict[str, Any]:
    for key, value in environ.items():
        if not key.startswith(ENV_PREFIX) or "__" not in key:
            continue
        section, _, field = key[len(ENV_PREFIX):].lower().partition("__")
        raw.setdefault(section, {})[field] = yaml.safe_load(value)
    return raw


def load_settings(path: str | Path | None = None, environ: dict[str, str] | None = None) -> Settings:
    """Load settings from YAML (if the file exists) then apply env overrides."""
    environ = dict(os.environ) if environ is None else environ
    cfg_path = Path(path or environ.get("INDICLEARN_CONFIG", "configs/config.yaml"))
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return Settings(**_apply_env_overrides(raw, environ))
