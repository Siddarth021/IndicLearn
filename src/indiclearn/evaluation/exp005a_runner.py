"""EXP-005A: Research-Hardened RAG Evaluation, Real-LLM Validation & Benchmark Expansion Runner.

This module implements Phase 5.1 Research Hardening:
1. Expanded Benchmark Evaluation (60 queries across 12 categories, 4 language varieties, 3 difficulties).
2. Per-Language and Per-Difficulty Disaggregated Retrieval Metrics.
3. Knowledge Graph Contribution Ablation (System A vs System B vs System C).
4. Concept Mapping Query Formulation Ablation (Raw vs Normalized vs Concept-Expanded).
5. Benchmark Leakage Verification.
6. Answerability and Evidence Sufficiency Matrix.
7. Citation Integrity Metrics (Validity, Coverage, Unsupported Rate) clearly distinguished from factual truth.
8. Claim-Level Propositional Audit via ClaimLevelAuditor.
9. Real LLM Opt-in Provider Evaluation (offline-safe: reports NOT RUN if unconfigured).
10. Human Evaluation Rubric Specification (reports NOT YET EVALUATED).

Scientific Interpretation Rule:
    Pipeline Validation != Real LLM Answer Quality != Human-Judged Correctness.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from indiclearn.domain.concepts import Concept
from indiclearn.domain.documents import DocumentChunk
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.llm.base import LLMProvider
from indiclearn.llm.mock import MockLLMProvider
from indiclearn.llm.providers.openai_compatible import OpenAICompatibleProvider
from indiclearn.nlp.pipeline import NLPPipeline
from indiclearn.rag.claims import ClaimLevelAuditor
from indiclearn.rag.models import RAGRequest
from indiclearn.rag.pipeline import RAGPipeline
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.dense import DenseRetriever
from indiclearn.retrieval.hybrid import HybridRetriever
from indiclearn.retrieval.in_memory_vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class Exp005aRunner:
    """Evaluation runner for EXP-005A (Hardened RAG Benchmark & Multi-Dimensional Ablations)."""

    def __init__(
        self,
        eval_path: str = "data/evaluation/rag_eval_expanded.json",
        corpus_path: str = "data/evaluation/signals_and_systems_eval.json",
        concept_graph_path: str = "data/evaluation/concept_graph_gold.json",
        subject_id: str = "signals_and_systems",
        provider_mode: str = "mock",
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.eval_path = Path(eval_path)
        self.corpus_path = Path(corpus_path)
        self.concept_graph_path = Path(concept_graph_path)
        self.subject_id = subject_id
        self.provider_mode = provider_mode

        # Configure LLM Provider
        if llm_provider is not None:
            self.llm_provider = llm_provider
        elif provider_mode == "real":
            real_p = OpenAICompatibleProvider.from_env()
            self.llm_provider = real_p if real_p.is_configured else MockLLMProvider()
        else:
            self.llm_provider = MockLLMProvider()

        # Load datasets
        self.eval_data = self._load_json(self.eval_path)
        self.corpus_data = self._load_json(self.corpus_path)
        self.concepts = self._load_concepts(self.concept_graph_path)
        self.chunks = self._load_chunks(self.corpus_data)

        # Initialize knowledge graph store
        self.graph_store = NetworkXGraphStore(default_subject_id=self.subject_id)
        for cpt in self.concepts:
            self.graph_store.add_concept(cpt)

        # Initialize NLP pipeline
        self.nlp_pipeline = NLPPipeline.from_graph_store(self.graph_store)

        # Initialize retrievers
        self.bm25_retriever = BM25Retriever()
        self.bm25_retriever.index(self.chunks)

        self.vector_store = InMemoryVectorStore()
        from indiclearn.embeddings.mock import MockEmbeddingProvider

        self.mock_embedder = MockEmbeddingProvider(dimension=384)
        embeddings = self.mock_embedder.embed_documents([c.text for c in self.chunks])
        self.vector_store.add(self.chunks, embeddings)
        self.dense_retriever = DenseRetriever(
            embedding_provider=self.mock_embedder,
            vector_store=self.vector_store,
        )

        self.hybrid_retriever = HybridRetriever(
            lexical_retriever=self.bm25_retriever,
            dense_retriever=self.dense_retriever,
            strategy="rrf",
            rrf_k=60,
        )

        # Full RAG Pipeline
        self.pipeline = RAGPipeline(
            nlp_pipeline=self.nlp_pipeline,
            retriever=self.hybrid_retriever,
            llm_provider=self.llm_provider,
            graph_store=self.graph_store,
        )

        # Claim-Level Auditor
        self.claim_auditor = ClaimLevelAuditor()

    def _load_json(self, path: Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_concepts(self, path: Path) -> list[Concept]:
        data = self._load_json(path)
        concepts = []
        for item in data.get("concepts", []):
            concepts.append(
                Concept(
                    concept_id=item["concept_id"],
                    canonical_name=item["canonical_name"],
                    subject_id=self.subject_id,
                    concept_type=item.get("concept_type", "concept"),
                    aliases=item.get("aliases", []),
                    description=item.get("description", ""),
                    source_chunk_ids=item.get("source_chunk_ids", []),
                )
            )
        return concepts

    def _load_chunks(self, corpus_json: dict[str, Any]) -> list[DocumentChunk]:
        chunks = []
        for item in corpus_json.get("corpus", []):
            chunks.append(
                DocumentChunk(
                    chunk_id=item["chunk_id"],
                    document_id=item["document_id"],
                    subject_id=self.subject_id,
                    page_number=item.get("page_number", 1),
                    section=item.get("section"),
                    text=item["text"],
                    char_start=item.get("char_start", 0),
                    char_end=item.get("char_end", len(item["text"])),
                    language=item.get("language", "en"),
                    concept_ids=item.get("concept_ids", []),
                )
            )
        return chunks

    def check_benchmark_leakage(self) -> dict[str, Any]:
        """Check for verbatim text leakage between benchmark queries and corpus chunks."""
        queries = self.eval_data.get("queries", [])
        chunk_texts = [c.text.lower() for c in self.chunks]

        leakage_detected = False
        leaked_queries: list[dict[str, str]] = []

        for q in queries:
            q_text = q["query"].strip().lower()
            # If query length > 25 chars and is an exact substring of any chunk
            if len(q_text) >= 25:
                for idx, c_text in enumerate(chunk_texts):
                    if q_text in c_text:
                        leakage_detected = True
                        leaked_queries.append({
                            "query_id": q.get("query_id", "unknown"),
                            "query": q["query"],
                            "matching_chunk_id": self.chunks[idx].chunk_id,
                        })
                        break

        return {
            "leakage_detected": leakage_detected,
            "total_queries_checked": len(queries),
            "total_chunks_checked": len(self.chunks),
            "leaked_queries_count": len(leaked_queries),
            "leaked_queries": leaked_queries,
            "status": "PASS (No verbatim leakage detected)" if not leakage_detected else "FAIL (Leakage detected)",
        }

    def _retrieve_for_query(self, query_text: str, top_k: int = 5) -> list[str]:
        """Execute default concept-aware hybrid retrieval matching RAG pipeline."""
        nlp_res = self.nlp_pipeline.analyze(query_text, subject_id=self.subject_id)
        norm_query = nlp_res.normalization.normalized_text

        candidates = self.hybrid_retriever.retrieve(
            query=norm_query,
            top_k=top_k * 2,
            filters={"subject_id": self.subject_id},
        )
        if nlp_res.concept_mapping.primary_concept:
            cpt_name = nlp_res.concept_mapping.primary_concept.canonical_name
            if cpt_name.lower() not in norm_query.lower():
                cpt_cands = self.hybrid_retriever.retrieve(
                    query=cpt_name,
                    top_k=top_k,
                    filters={"subject_id": self.subject_id},
                )
                candidates.extend(cpt_cands)

        # Deduplicate preserving highest score
        dedup: dict[str, Any] = {}
        for c in candidates:
            if c.chunk_id not in dedup or c.score > dedup[c.chunk_id].score:
                dedup[c.chunk_id] = c

        sorted_cands = sorted(dedup.values(), key=lambda x: x.score, reverse=True)[:top_k]
        return [c.chunk_id for c in sorted_cands]

    def _compute_metrics_for_subset(
        self, query_items: list[dict[str, Any]], top_k: int = 5
    ) -> dict[str, Any]:
        """Compute standard IR metrics for a specified subset of queries."""
        total = len(query_items)
        if total == 0:
            return {
                "queries_count": 0,
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "mrr": 0.0,
                "gold_evidence_recall_at_5": 0.0,
            }

        recall_1 = 0
        recall_3 = 0
        recall_5 = 0
        reciprocal_ranks: list[float] = []
        coverages: list[float] = []

        for q in query_items:
            gold_chunks = set(q.get("gold_evidence_chunk_ids", []))
            retrieved_chunk_ids = self._retrieve_for_query(q["query"], top_k=top_k)

            # R@1
            r1 = 1 if (retrieved_chunk_ids and retrieved_chunk_ids[0] in gold_chunks) else 0
            recall_1 += r1

            # R@3
            r3 = 1 if any(cid in retrieved_chunk_ids[:3] for cid in gold_chunks) else 0
            recall_3 += r3

            # R@5
            r5 = 1 if any(cid in retrieved_chunk_ids[:5] for cid in gold_chunks) else 0
            recall_5 += r5

            # MRR
            rank = 0
            for idx, cid in enumerate(retrieved_chunk_ids, start=1):
                if cid in gold_chunks:
                    rank = idx
                    break
            reciprocal_ranks.append(1.0 / rank if rank > 0 else 0.0)

            # Gold Evidence Recall
            if gold_chunks:
                cov = len(set(retrieved_chunk_ids[:top_k]) & gold_chunks) / len(gold_chunks)
                coverages.append(cov)
            else:
                coverages.append(1.0)

        return {
            "queries_count": total,
            "recall_at_1": round(recall_1 / total, 4),
            "recall_at_3": round(recall_3 / total, 4),
            "recall_at_5": round(recall_5 / total, 4),
            "mrr": round(sum(reciprocal_ranks) / total, 4),
            "gold_evidence_recall_at_5": round(sum(coverages) / len(coverages), 4) if coverages else 0.0,
        }

    def evaluate_retrieval(self, top_k: int = 5) -> dict[str, Any]:
        """Compute aggregate and disaggregated retrieval metrics."""
        queries = self.eval_data.get("queries", [])
        answerable_queries = [q for q in queries if q.get("answerability", True)]

        # 1. Aggregate
        aggregate = self._compute_metrics_for_subset(answerable_queries, top_k=top_k)

        # Multi-chunk specific recall
        multi_chunk_queries = [q for q in answerable_queries if q.get("requires_multichunk", False)]
        multi_metrics = self._compute_metrics_for_subset(multi_chunk_queries, top_k=top_k)
        aggregate["multi_chunk_evidence_recall"] = multi_metrics.get("gold_evidence_recall_at_5", 0.0)

        # 2. Per-Language Breakdown
        by_language: dict[str, dict[str, Any]] = {}
        for lang in ["en", "te", "te-Latn", "te-en"]:
            lang_subset = [q for q in answerable_queries if q.get("language") == lang]
            by_language[lang] = self._compute_metrics_for_subset(lang_subset, top_k=top_k)

        # 3. Per-Difficulty Breakdown
        by_difficulty: dict[str, dict[str, Any]] = {}
        for diff in ["easy", "medium", "hard"]:
            diff_subset = [q for q in answerable_queries if q.get("difficulty") == diff]
            by_difficulty[diff] = self._compute_metrics_for_subset(diff_subset, top_k=top_k)

        # 4. Per-Category Breakdown
        categories = sorted({q.get("category", "general") for q in answerable_queries})
        by_category: dict[str, dict[str, Any]] = {}
        for cat in categories:
            cat_subset = [q for q in answerable_queries if q.get("category") == cat]
            by_category[cat] = self._compute_metrics_for_subset(cat_subset, top_k=top_k)

        return {
            "aggregate": aggregate,
            "by_language": by_language,
            "by_difficulty": by_difficulty,
            "by_category": by_category,
        }

    def evaluate_knowledge_graph_ablation(self, top_k: int = 5) -> dict[str, Any]:
        """Evaluate Knowledge Graph contribution (Section 21).

        Ablation Configurations:
        - System A: Hybrid retrieval only (raw query directly to HybridRetriever)
        - System B: Hybrid retrieval + Concept Mapping (query normalized + mapped canonical concept)
        - System C: Hybrid retrieval + Concept Mapping + Graph Context (full pipeline with graph context)
        """
        queries = self.eval_data.get("queries", [])
        answerable_queries = [q for q in queries if q.get("answerability", True)]
        total = len(answerable_queries)

        def eval_system(retrieve_fn: Any) -> dict[str, float]:
            r5_hits = 0
            mrr_scores = []
            cov_scores = []

            for q in answerable_queries:
                gold_chunks = set(q.get("gold_evidence_chunk_ids", []))
                retrieved_cids = retrieve_fn(q["query"])[:top_k]

                # R@5
                hit = 1 if any(cid in retrieved_cids for cid in gold_chunks) else 0
                r5_hits += hit

                # MRR
                rank = 0
                for idx, cid in enumerate(retrieved_cids, start=1):
                    if cid in gold_chunks:
                        rank = idx
                        break
                mrr_scores.append(1.0 / rank if rank > 0 else 0.0)

                # Gold Evidence Recall
                if gold_chunks:
                    cov = len(set(retrieved_cids) & gold_chunks) / len(gold_chunks)
                    cov_scores.append(cov)
                else:
                    cov_scores.append(1.0)

            return {
                "recall_at_5": round(r5_hits / total, 4) if total > 0 else 0.0,
                "mrr": round(sum(mrr_scores) / total, 4) if total > 0 else 0.0,
                "gold_evidence_recall_at_5": round(sum(cov_scores) / total, 4) if total > 0 else 0.0,
            }

        # System A: Hybrid retrieval only
        def retrieve_a(query: str) -> list[str]:
            res = self.hybrid_retriever.retrieve(query=query, top_k=top_k, filters={"subject_id": self.subject_id})
            return [r.chunk_id for r in res]

        # System B: Hybrid + Concept Mapping
        def retrieve_b(query: str) -> list[str]:
            nlp_res = self.nlp_pipeline.analyze(query, subject_id=self.subject_id)
            norm_q = nlp_res.normalization.normalized_text
            candidates = self.hybrid_retriever.retrieve(query=norm_q, top_k=top_k * 2, filters={"subject_id": self.subject_id})
            if nlp_res.concept_mapping.primary_concept:
                cpt_name = nlp_res.concept_mapping.primary_concept.canonical_name
                if cpt_name.lower() not in norm_q.lower():
                    candidates.extend(self.hybrid_retriever.retrieve(query=cpt_name, top_k=top_k, filters={"subject_id": self.subject_id}))
            dedup: dict[str, Any] = {}
            for c in candidates:
                if c.chunk_id not in dedup or c.score > dedup[c.chunk_id].score:
                    dedup[c.chunk_id] = c
            sorted_c = sorted(dedup.values(), key=lambda x: x.score, reverse=True)[:top_k]
            return [c.chunk_id for c in sorted_c]

        # System C: Hybrid + Concept Mapping + Graph Context (Full Pipeline evidence selection)
        def retrieve_c(query: str) -> list[str]:
            resp = self.pipeline.answer(query)
            return [item.chunk_id for item in resp.evidence.items]

        results_a = eval_system(retrieve_a)
        results_b = eval_system(retrieve_b)
        results_c = eval_system(retrieve_c)

        return {
            "system_a_hybrid_only": {
                "description": "Hybrid lexical (BM25) + dense retrieval on raw query text",
                **results_a,
            },
            "system_b_hybrid_plus_concept_mapping": {
                "description": "Hybrid retrieval with Telugu/Romanized normalization and cross-lingual concept mapping",
                **results_b,
            },
            "system_c_hybrid_plus_concept_mapping_plus_graph": {
                "description": "Full proposed pipeline: Concept-aware hybrid retrieval with curriculum graph context and sufficiency gating",
                **results_c,
            },
            "graph_contribution_analysis": (
                "Empirical comparison between System B and System C measures the isolated effect of "
                "knowledge graph context and curriculum graph prerequisite expansion on retrieval recall and MRR."
            ),
        }

    def evaluate_concept_mapping_ablation(self, top_k: int = 5) -> dict[str, Any]:
        """Evaluate Concept Mapping query formulation ablations (Section 22).

        Compare:
        1. Raw query
        2. Normalized query
        3. Normalized + canonical concept expansion
        """
        queries = self.eval_data.get("queries", [])
        answerable_queries = [q for q in queries if q.get("answerability", True)]

        def eval_formulation(get_query_candidates: Any) -> dict[str, Any]:
            cands_list = []
            for q in answerable_queries:
                cids = get_query_candidates(q["query"])
                cands_list.append(cids)

            total = len(answerable_queries)
            r1 = 0
            r3 = 0
            r5 = 0
            mrr_list = []
            cov_list = []

            for q, cids in zip(answerable_queries, cands_list, strict=False):
                gold = set(q.get("gold_evidence_chunk_ids", []))
                r1 += 1 if (cids and cids[0] in gold) else 0
                r3 += 1 if any(c in cids[:3] for c in gold) else 0
                r5 += 1 if any(c in cids[:5] for c in gold) else 0

                rank = 0
                for idx, c in enumerate(cids, start=1):
                    if c in gold:
                        rank = idx
                        break
                mrr_list.append(1.0 / rank if rank > 0 else 0.0)

                if gold:
                    cov_list.append(len(set(cids[:top_k]) & gold) / len(gold))
                else:
                    cov_list.append(1.0)

            return {
                "recall_at_1": round(r1 / total, 4),
                "recall_at_3": round(r3 / total, 4),
                "recall_at_5": round(r5 / total, 4),
                "mrr": round(sum(mrr_list) / total, 4),
                "gold_evidence_recall_at_5": round(sum(cov_list) / total, 4),
            }

        # 1. Raw query
        def raw_fn(q: str) -> list[str]:
            res = self.hybrid_retriever.retrieve(query=q, top_k=top_k, filters={"subject_id": self.subject_id})
            return [r.chunk_id for r in res]

        # 2. Normalized query
        def norm_fn(q: str) -> list[str]:
            nlp_res = self.nlp_pipeline.analyze(q, subject_id=self.subject_id)
            norm_q = nlp_res.normalization.normalized_text
            res = self.hybrid_retriever.retrieve(query=norm_q, top_k=top_k, filters={"subject_id": self.subject_id})
            return [r.chunk_id for r in res]

        # 3. Normalized + Canonical concept expansion
        def norm_plus_concept_fn(q: str) -> list[str]:
            return self._retrieve_for_query(q, top_k=top_k)

        res_raw = eval_formulation(raw_fn)
        res_norm = eval_formulation(norm_fn)
        res_expanded = eval_formulation(norm_plus_concept_fn)

        return {
            "setting_1_raw_query": {
                "description": "Raw unnormalized user query text directly submitted to hybrid retriever",
                **res_raw,
            },
            "setting_2_normalized_query": {
                "description": "Query normalized via TeluguNormalizer / TransliterationNormalizer",
                **res_norm,
            },
            "setting_3_normalized_plus_canonical_expansion": {
                "description": "Normalized query with cross-lingual concept mapping and canonical concept expansion",
                **res_expanded,
            },
        }

    def evaluate_pipeline_and_answerability(self) -> dict[str, Any]:
        """Evaluate full end-to-end pipeline, answerability matrix, and citation integrity."""
        queries = self.eval_data.get("queries", [])
        total = len(queries)

        validity_rates: list[float] = []
        coverage_rates: list[float] = []
        unsupported_rates: list[float] = []

        matrix_counts = {
            "answerable_evidence_sufficient_answered": 0,
            "answerable_evidence_sufficient_incorrect": 0,
            "answerable_evidence_insufficient_refused": 0,
            "out_of_scope_correctly_refused": 0,
            "out_of_scope_hallucinated": 0,
        }

        grounding_status_counts: dict[str, int] = defaultdict(int)

        # Claim-level aggregate counts
        claim_totals = {
            "total_claims": 0,
            "cited_claims": 0,
            "uncited_claims": 0,
            "supported_claims": 0,
            "unsupported_citation_claims": 0,
            "unrelated_evidence_claims": 0,
            "metadata_contradictions": 0,
        }

        for q in queries:
            query_text = q["query"]
            gold_answerable = q.get("answerability", True)
            expected_lang = q.get("language", "en")

            req = RAGRequest(
                query=query_text,
                subject_id=self.subject_id,
                requested_language=expected_lang,
                top_k=5,
            )
            resp = self.pipeline.answer(req)
            status = resp.grounding_report.status
            grounding_status_counts[status] += 1

            if gold_answerable:
                if resp.is_answerable and not resp.evidence.insufficient_evidence:
                    if resp.grounding_report.check.is_grounded:
                        matrix_counts["answerable_evidence_sufficient_answered"] += 1
                    else:
                        matrix_counts["answerable_evidence_sufficient_incorrect"] += 1

                    validity_rates.append(resp.grounding_report.check.citation_validity_rate)
                    coverage_rates.append(resp.grounding_report.check.citation_coverage)
                    unsupported_rates.append(resp.grounding_report.check.unsupported_citation_rate)
                else:
                    matrix_counts["answerable_evidence_insufficient_refused"] += 1
            else:
                # Out-of-scope or adversarial query
                if not resp.is_answerable or resp.status == "INSUFFICIENT_EVIDENCE":
                    matrix_counts["out_of_scope_correctly_refused"] += 1
                else:
                    matrix_counts["out_of_scope_hallucinated"] += 1

            # Perform claim-level propositional audit on non-refusal answers
            if resp.is_answerable and not resp.evidence.insufficient_evidence:
                claim_res = self.claim_auditor.audit_claims(resp.answer, resp.evidence)
                claim_totals["total_claims"] += claim_res.total_claims
                claim_totals["cited_claims"] += claim_res.cited_claims
                claim_totals["uncited_claims"] += claim_res.uncited_claims
                claim_totals["supported_claims"] += claim_res.supported_claims
                claim_totals["unsupported_citation_claims"] += claim_res.unsupported_citation_claims
                claim_totals["unrelated_evidence_claims"] += claim_res.unrelated_evidence_claims
                claim_totals["metadata_contradictions"] += len(claim_res.metadata_contradictions)

        avg_validity = sum(validity_rates) / len(validity_rates) if validity_rates else 1.0
        avg_coverage = sum(coverage_rates) / len(coverage_rates) if coverage_rates else 0.0
        avg_unsupported = sum(unsupported_rates) / len(unsupported_rates) if unsupported_rates else 0.0

        return {
            "total_queries_evaluated": total,
            "answerability_matrix": {
                "answerable": {
                    "evidence_sufficient_and_grounded": matrix_counts["answerable_evidence_sufficient_answered"],
                    "evidence_sufficient_but_grounding_failed": matrix_counts["answerable_evidence_sufficient_incorrect"],
                    "evidence_insufficient_refused": matrix_counts["answerable_evidence_insufficient_refused"],
                },
                "out_of_scope_and_adversarial": {
                    "correctly_refused": matrix_counts["out_of_scope_correctly_refused"],
                    "incorrectly_hallucinated_or_answered": matrix_counts["out_of_scope_hallucinated"],
                },
            },
            "grounding_status_breakdown": dict(grounding_status_counts),
            "citation_integrity_metrics": {
                "citation_validity_rate": round(avg_validity, 4),
                "citation_coverage_rate": round(avg_coverage, 4),
                "unsupported_citation_rate": round(avg_unsupported, 4),
                "scientific_interpretation": (
                    "Citation integrity metrics evaluate citation identifier resolution and sentence-level "
                    "citation attachment within supplied evidence. They do NOT establish real LLM factual accuracy "
                    "or human-perceived semantic correctness."
                ),
            },
            "claim_level_audit": {
                **claim_totals,
                "supported_claim_ratio": (
                    round(claim_totals["supported_claims"] / claim_totals["total_claims"], 4)
                    if claim_totals["total_claims"] > 0
                    else 1.0
                ),
            },
        }

    def evaluate_real_llm_opt_in(self, output_dir: str = "data/evaluation/llm_outputs") -> dict[str, Any]:
        """Execute opt-in evaluation with real external LLM if configured via environment variables."""
        if self.provider_mode != "real":
            return {
                "status": "REAL LLM EVALUATION: NOT RUN",
                "reason": "Real LLM evaluation is opt-in. Specify --provider real to execute.",
            }

        if not isinstance(self.llm_provider, OpenAICompatibleProvider) or not self.llm_provider.is_configured:
            return {
                "status": "REAL LLM EVALUATION: NOT RUN",
                "reason": (
                    "No external OpenAI-compatible credentials configured in environment variables. "
                    "Set LLM_API_KEY, LLM_BASE_URL, and LLM_MODEL to run real model evaluation."
                ),
            }

        # If configured, evaluate sample queries and save sanitized outputs
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        sanitized_file = out_path / "exp005a_sanitized_outputs.json"

        queries = self.eval_data.get("queries", [])
        sample_queries = [q for q in queries if q.get("answerability", True)][:10]

        outputs = []
        for q in sample_queries:
            resp = self.pipeline.answer(q["query"])
            outputs.append({
                "query_id": q.get("query_id"),
                "query": q["query"],
                "language": q.get("language"),
                "answer": resp.answer,
                "citations": [c.model_dump() for c in resp.citations],
                "grounding_status": resp.grounding_report.status,
            })

        with open(sanitized_file, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)

        return {
            "status": "REAL LLM EVALUATION: EXECUTED",
            "provider": "OpenAICompatibleProvider",
            "model": self.llm_provider.model_name,
            "evaluated_queries_count": len(outputs),
            "sanitized_output_path": str(sanitized_file),
        }

    def get_human_evaluation_rubric(self) -> dict[str, Any]:
        """Formal human evaluation schema definition (Section 10)."""
        return {
            "status": "HUMAN EVALUATION: NOT YET EVALUATED",
            "scale": {
                "0": "Poor / incorrect / unsupported / grammatically unacceptable",
                "1": "Partially correct / partially supported / minor translation or terminology flaws",
                "2": "Good / fully correct / strongly grounded / fluent and appropriate academic phrasing",
            },
            "dimensions": {
                "correctness": "Factual accuracy of engineering concepts relative to university signal processing curriculum (0-2).",
                "groundedness": "Extent to which all substantive factual claims are strictly substantiated by the cited retrieved materials (0-2).",
                "relevance": "Directness and specificity of the response addressing the student's question (0-2).",
                "completeness": "Coverage of expected core principles, equations, or properties required for a thorough answer (0-2).",
                "language_appropriateness": "Naturalness, fluency, and proper engineering terminology in the requested language/code-mixed style (0-2).",
            },
            "note": "No human scoring panel has rated model completions in this automated release. Results must strictly reflect NOT YET EVALUATED.",
        }

    def run_experiment(self, output_path: str = "data/evaluation/exp005a_results.json") -> dict[str, Any]:
        """Execute end-to-end EXP-005A evaluation and write versioned results JSON."""
        logger.info("Executing EXP-005A: Research-Hardened RAG Evaluation...")

        leakage_results = self.check_benchmark_leakage()
        retrieval_results = self.evaluate_retrieval(top_k=5)
        kg_ablation = self.evaluate_knowledge_graph_ablation(top_k=5)
        cm_ablation = self.evaluate_concept_mapping_ablation(top_k=5)
        pipeline_and_answerability = self.evaluate_pipeline_and_answerability()
        real_llm_results = self.evaluate_real_llm_opt_in()
        human_eval_schema = self.get_human_evaluation_rubric()

        results = {
            "experiment_id": "EXP-005A",
            "name": "Research-Hardened RAG Evaluation, Real-LLM Validation & Benchmark Expansion",
            "benchmark_dataset": str(self.eval_path),
            "corpus_dataset": str(self.corpus_path),
            "subject_id": self.subject_id,
            "provider_mode": self.provider_mode,
            "llm_provider_class": self.llm_provider.__class__.__name__,
            "scientific_interpretation_principle": (
                "Pipeline validation != Real LLM answer quality != Human-judged correctness. "
                "MockLLMProvider validates grounding mechanics, citation parsing, sufficiency routing, and error boundaries. "
                "It does not establish real LLM factual accuracy or colloquial Telugu generative quality."
            ),
            "benchmark_leakage_audit": leakage_results,
            "retrieval_metrics": retrieval_results,
            "ablations": {
                "knowledge_graph_ablation": kg_ablation,
                "concept_mapping_query_ablation": cm_ablation,
            },
            "pipeline_and_answerability": pipeline_and_answerability,
            "real_llm_evaluation": real_llm_results,
            "human_evaluation": human_eval_schema,
            "limitations": [
                "Automated citation integrity validates source identifier presence and sentence coverage; it does not replace human domain expertise.",
                "Real LLM evaluation requires external API credentials and was executed in offline-mock mode for deterministic CI.",
                "Syntactic sentence splitting is a heuristic proxy for atomic propositional claims.",
                "Multimodal content (plots, diagrams, circuit schematics) is currently evaluated on extracted textual captions only.",
            ],
        }

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"EXP-005A results written to {output_path}")
        return results
