"""Build IndicLearn_handoff_<date>.zip with HANDOFF_SUMMARY.txt, then validate it.

Excludes secrets, caches, venvs, node_modules, .git, model weights, data payloads.
Usage: python scripts/make_handoff.py [output_dir]
"""

from __future__ import annotations

import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
                ".mypy_cache", ".ruff_cache", "dist", ".idea", ".vscode"}
EXCLUDE_NAMES = {".env"}
EXCLUDE_SUFFIXES = {".pyc", ".zip", ".sqlite3", ".bin", ".safetensors", ".pt", ".pth", ".onnx"}
REQUIRED = [
    "README.md",
    "docs/PROJECT_CONTEXT.md",
    "docs/HANDOFF.md",
    "docs/ARCHITECTURE.md",
    "docs/KNOWLEDGE_GRAPH.md",
    "docs/NLP_PIPELINE.md",
    "docs/RAG_PIPELINE.md",
    "docs/KNOWLEDGE_MODEL.md",
    "docs/DECISIONS.md",
    "docs/TODO.md",
    "docs/CHANGELOG.md",
    ".env.example",
    "configs/config.yaml",
    "src/indiclearn/__init__.py",
    "tests/conftest.py",
    "HANDOFF_SUMMARY.txt",
    "data/evaluation/rag_eval.json",
    "data/evaluation/rag_eval_expanded.json",
    "data/evaluation/exp005a_results.json",
    "data/evaluation/scope_eval.json",
    "data/evaluation/exp005b_results.json",
    "data/evaluation/exp006_student_eval.json",
    "data/evaluation/exp006_results.json",
    "docs/ADAPTIVE_ASSESSMENT.md",
    "data/evaluation/signals_systems_items.json",
    "data/evaluation/exp007_assessment_eval.json",
    "data/evaluation/exp007_results.json",
    "docs/RECOMMENDATION_ENGINE.md",
    "data/evaluation/exp008_recommendation_eval.json",
    "data/evaluation/exp008_results.json",
    "docs/STUDENT_DASHBOARD.md",
    "data/evaluation/exp009_dashboard_eval.json",
    "data/evaluation/exp009_results.json",
    "docs/RESEARCH_EVALUATION.md",
    "docs/HUMAN_EVALUATION_PROTOCOL.md",
    "docs/LEARNING_EFFECTIVENESS_STUDY.md",
    "data/evaluation/experiment_registry.json",
    "data/evaluation/exp010_recommendation_eval.json",
    "data/evaluation/exp010_results.json",
    "data/evaluation/exp011_assessment_eval.json",
    "data/evaluation/exp011_results.json",
    "data/evaluation/exp012_usability_template.json",
    "data/evaluation/exp012_results.json",
    "data/evaluation/exp013_annotation_template.json",
    "data/evaluation/exp013_results.json",
    "data/evaluation/exp014_multilingual_template.json",
    "data/evaluation/exp014_results.json",
    "data/evaluation/human/annotation_guidelines.md",
    "data/evaluation/human/usability_questionnaire.json",
]

SUMMARY = """INDICLEARN PROJECT HANDOFF
==========================

READ THIS FIRST.

The project is:
IndicLearn - a research-grade prototype of a dynamic, personalized, concept-level knowledge map
for engineering students using Indic-language (initially Telugu, incl. Romanized and code-mixed)
NLP + RAG. Academic knowledge graph and student knowledge model are strictly separate layers.

Current status:
Phase 0 (architecture docs), Phase 1 (ingestion/cleaning/chunking), and Phase 2 (embeddings,
retrieval layer, FAISS vector store, BM25 baseline, evaluation metrics, and EXP-002 benchmark)
are complete. 60 unit/integration tests passing.

Current phase:
Phase 2 complete -> Phase 3 (Concept extraction + Academic Knowledge Graph) next.

Last completed task:
Phase 2 retrieval layer: EmbeddingProvider, VectorStore (FAISS + NumPy fallback), BM25, TF-IDF,
DenseRetriever, HybridRetriever (RRF), retrieval metrics, cross-lingual eval benchmark, and EXP-002.

Next task:
Concept & ConceptRelation domain entities, GraphStore protocol, NetworkX adapter, and concept extraction.

Important files:
docs/PROJECT_CONTEXT.md, docs/HANDOFF.md, docs/DECISIONS.md, docs/EXPERIMENTS.md, docs/RAG_PIPELINE.md,
src/indiclearn/retrieval/base.py, src/indiclearn/embeddings/base.py, configs/config.yaml

How to run:
python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]" && cp .env.example .env

How to test:
python -m pytest -q ; ruff check . ; python scripts/run_exp002.py

Known problems:
Real Telugu PDF extraction unverified; no OCR; dedupe state in-memory; mypy not yet run.

Important research findings:
EXP-002 evaluated: multilingual-e5-small achieved 1.0000 Recall@3 across EN/Telugu/Romanized/code-mixed.
BM25 achieved 1.0000 on Romanized/code-mixed but degraded to 0.4000 on pure Telugu script.
"""


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return not (set(rel.parts) & EXCLUDE_DIRS or path.name in EXCLUDE_NAMES
                or path.suffix in EXCLUDE_SUFFIXES)


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"IndicLearn_handoff_{datetime.now(timezone.utc).date().isoformat()}.zip"
    summary_content = (ROOT / "HANDOFF_SUMMARY.txt").read_text(encoding="utf-8") if (ROOT / "HANDOFF_SUMMARY.txt").exists() else SUMMARY
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("HANDOFF_SUMMARY.txt", summary_content)
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.name != "HANDOFF_SUMMARY.txt" and included(p):
                zf.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = [r for r in REQUIRED if r not in names]
        assert not missing, f"missing from archive: {missing}"
        assert not any(n.endswith(".env") for n in names), ".env leaked"
        with tempfile.TemporaryDirectory() as td:
            zf.extractall(td)
            assert (Path(td) / "docs/HANDOFF.md").read_text(encoding="utf-8")
    print(f"OK {zip_path} ({len(names)} files)")


if __name__ == "__main__":
    main()
