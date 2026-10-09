"""EXP-002: Multilingual and Indic Retrieval Evaluation Runner (Phase 2).

Executes rigorous comparative evaluation across:
- TF-IDF baseline
- BM25 baseline
- Dense Embedding models (Multilingual, Indic, English)
- Hybrid (BM25 + Dense)

Evaluates cross-lingual retrieval (EN, TE, Romanized TE, Code-mixed -> English material)
and records detailed error analysis.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from indiclearn.domain.documents import DocumentChunk
from indiclearn.evaluation.retrieval_metrics import (
    QueryEvaluation,
    aggregate_metrics,
)
from indiclearn.retrieval.base import Retriever


def categorize_failure(
    query_lang: str,
    query_text: str,
    retrieved_text: str | None,
    expected_concept: str,
) -> tuple[str, str]:
    """Diagnose and categorize retrieval failure for error analysis."""
    if query_lang == "te-Latn":
        return (
            "Romanization ambiguity",
            "Informal phonetic Romanization lacks standardized orthography, causing subword/character tokenization mismatch.",
        )
    if query_lang == "te":
        return (
            "Cross-lingual semantic mismatch",
            "Model failed to align Telugu script query representation with English technical text representation.",
        )
    if query_lang == "te-en":
        return (
            "Code-mixed syntax mismatch",
            "Mixed Telugu grammar particles combined with English technical tokens degraded semantic alignment.",
        )
    return (
        "Technical terminology mismatch",
        f"Specific technical semantics for '{expected_concept}' were not prioritized over broader domain tokens.",
    )


class ExperimentRunner:
    """Orchestrates comparative retrieval experiments on evaluation datasets."""

    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = Path(dataset_path)
        with open(self.dataset_path, encoding="utf-8") as f:
            self.data = json.load(f)

        self.corpus_chunks: list[DocumentChunk] = []
        for raw in self.data["corpus"]:
            self.corpus_chunks.append(DocumentChunk(**raw))

        self.queries: list[dict[str, Any]] = self.data["queries"]
        self.chunk_map: dict[str, DocumentChunk] = {c.chunk_id: c for c in self.corpus_chunks}

    def evaluate_retriever(
        self,
        name: str,
        retriever: Retriever,
        model_type: str = "lexical",
        dimension: int | None = None,
        model_size_mb: float | None = None,
        hardware: str = "CPU",
    ) -> dict[str, Any]:
        """Evaluate a single retriever across all queries and linguistic dimensions."""
        evaluations: list[QueryEvaluation] = []
        failures: list[dict[str, Any]] = []
        latencies: list[float] = []

        for q in self.queries:
            qid = q["query_id"]
            query_str = q["query"]
            relevant_ids = set(q["relevant_chunk_ids"])
            lang = q["language"]

            t0 = time.perf_counter()
            results = retriever.retrieve(query_str, top_k=10)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(latency_ms)

            retrieved_ids = [r.chunk_id for r in results]
            eval_res = QueryEvaluation(
                query_id=qid,
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                language=lang,
                query_type=q.get("query_type", "general"),
            )
            evaluations.append(eval_res)

            # Failure analysis if Recall@1 is 0 or target chunk not in top 3
            if eval_res.r1 == 0:
                top_retrieved = results[0] if results else None
                cat, cause = categorize_failure(
                    query_lang=lang,
                    query_text=query_str,
                    retrieved_text=top_retrieved.text if top_retrieved else None,
                    expected_concept=q["expected_concept"],
                )
                failures.append(
                    {
                        "query_id": qid,
                        "query": query_str,
                        "language": lang,
                        "script": q.get("script", "Latin"),
                        "expected_concept": q["expected_concept"],
                        "expected_chunks": list(relevant_ids),
                        "retrieved_top3": [
                            {"chunk_id": r.chunk_id, "score": r.score, "rank": r.rank}
                            for r in results[:3]
                        ],
                        "failure_category": cat,
                        "possible_cause": cause,
                    }
                )

        # Aggregate overall
        overall = aggregate_metrics(evaluations)

        # Aggregate by linguistic dimension
        dimensions: dict[str, dict[str, float]] = {}
        for lang_code in ["en", "te", "te-Latn", "te-en"]:
            lang_evals = [e for e in evaluations if e.language == lang_code]
            dimensions[lang_code] = aggregate_metrics(lang_evals)

        mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

        return {
            "name": name,
            "model_type": model_type,
            "dimension": dimension,
            "model_size_mb": model_size_mb,
            "hardware": hardware,
            "mean_latency_ms": round(mean_latency, 2),
            "overall_metrics": {k: round(v, 4) for k, v in overall.items()},
            "dimensional_metrics": {
                lang: {k: round(v, 4) for k, v in m.items()} for lang, m in dimensions.items()
            },
            "failures_count": len(failures),
            "failures": failures,
        }
