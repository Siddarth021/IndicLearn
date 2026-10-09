# PROJECT_CONTEXT — IndicLearn (single source of truth; keep updated)

_Last updated: 2026-10-06 (Phase 5 completion)_

## Identity
IndicLearn — A Dynamic Personalized Knowledge Map for Engineering Students using Indic Language NLP. Owner: Siddarth. Research-grade prototype, NOT a commercial LMS.

## Objective
Student uploads PDFs/PPTX/notes → system builds an academic concept graph → student asks in Telugu / Romanized Telugu / Telugu-English / English → Indic NLP maps the question to a concept → RAG answers grounded in uploaded material → interactions & quizzes become *evidence* → per-student concept-level knowledge estimates → adaptive quiz + revision recommendations + visual knowledge map.

## Research question (PROPOSED)
Does an Indic/code-mix-aware pipeline (romanization normalization + technical-term layer + concept mapping) map student questions to the correct academic concept more reliably than baselines (no normalization, keyword/BM25), and can evidence-based concept-level knowledge estimates usefully prioritize revision? **Evaluated for retrieval in EXP-002; evaluated for concept/relation extraction in EXP-003; evaluated for concept mapping in EXP-004; evaluated for grounded RAG and citation attribution in EXP-005.**

## Hard rules (from the project brief)
No fabricated papers/datasets/metrics. Mark PROPOSED / NOT YET EVALUATED. Academic graph and student model are SEPARATE layers. Preserve original query text. Weights/formulas configurable (heuristic ≠ validated). A question is evidence, not proof of ignorance. Never say implemented/tested/working without checking. Modular code, no monolith.

## Architecture
See ARCHITECTURE.md, KNOWLEDGE_GRAPH.md, NLP_PIPELINE.md, and RAG_PIPELINE.md. Layers: ingestion → preprocessing → chunking → embeddings/vector store → concepts → academic KG; nlp/* pipeline → concept_mapping → retrieval → rag; student_model (evidence → KnowledgeState) → dynamic student graph; quiz; recommendation; api; frontend. Providers behind interfaces (LLM, Embedding, VectorStore, GraphStore).

## Tech stack
Python 3.11+, Pydantic v2, PyYAML, pypdf, python-pptx, faiss-cpu, sentence-transformers, scikit-learn, networkx, pytest, ruff, mypy. Planned: FastAPI, SQLAlchemy+SQLite, Streamlit/React.

## Implemented components (status)
| Component | Path | Status |
|---|---|---|
| Settings loader (YAML + env override) | config/settings.py | TESTED |
| Structured JSON logging + `log_event` | utils/logging.py | IMPLEMENTED |
| Domain: Document, DocumentChunk, ExtractedPage, DocumentStatus | domain/documents.py | TESTED |
| Domain: Concept, ConceptRelation, ConceptType, RelationType | domain/concepts.py | TESTED |
| Text cleaning (NFC, preserves ZWJ/ZWNJ) | preprocessing/cleaning.py | TESTED |
| Recursive char chunker | chunking/recursive.py | TESTED |
| Loaders TXT/MD/PDF/PPTX | ingestion/loaders.py | TESTED on synthetic files |
| IngestionService (dedupe, failure recording, size limit) | ingestion/service.py | TESTED |
| EmbeddingProvider & domain models | embeddings/base.py | TESTED |
| SentenceTransformer & Mock providers | embeddings/sentence_transformer.py, mock.py | TESTED |
| EmbeddingCache & CachedEmbeddingProvider | embeddings/cache.py | TESTED |
| VectorStore protocol & typed exceptions | retrieval/base.py, exceptions.py | TESTED |
| InMemoryVectorStore (NumPy cosine fallback) | retrieval/in_memory_vector_store.py | TESTED |
| FaissVectorStore (IndexFlatIP, persistence, versioning) | retrieval/faiss_vector_store.py | TESTED |
| BM25Retriever (Indic Unicode tokenization, Okapi) | retrieval/bm25.py | TESTED |
| TFIDFRetriever (scikit-learn baseline) | retrieval/tfidf.py | TESTED |
| DenseRetriever & HybridRetriever (RRF + Linear) | retrieval/dense.py, hybrid.py | TESTED |
| RetrievalPipeline & Query Preprocessor | retrieval/pipeline.py | TESTED |
| IR Metrics (Recall@k, Precision@k, MRR, nDCG@k) | evaluation/retrieval_metrics.py | TESTED |
| EXP-002 Benchmark Dataset & Runner | evaluation/exp002_runner.py | TESTED (Executed) |
| Concept Normalizer (slugify, acronyms, aliases) | concepts/normalizer.py | TESTED |
| GraphStore Protocol | knowledge_graph/base.py | TESTED |
| NetworkXGraphStore (MultiDiGraph, BFS traversals, JSON persistence) | knowledge_graph/networkx_store.py | TESTED |
| Graph Integrity Validation (referential, self-loops, cycles) | knowledge_graph/validation.py | TESTED |
| HeuristicConceptExtractor (baseline rules, grammar, headings) | concepts/extractor.py | TESTED |
| ConceptService (orchestration, graph insertion, chunk linking) | concepts/service.py | TESTED |
| Graph Evaluation Metrics (Precision, Recall, F1) | evaluation/graph_metrics.py | TESTED |
| EXP-003 Manual Gold Standard & Runner | evaluation/exp003_runner.py | TESTED (Executed) |
| NLP Domain Models & Protocols | nlp/models.py, nlp/*/base.py | TESTED |
| HeuristicLanguageDetector | nlp/language_detection/detector.py | TESTED |
| RuleBasedCodeMixDetector (Token spans, CMI metric) | nlp/code_mix/detector.py | TESTED |
| RuleBasedRomanizationNormalizer (Ambiguity preservation) | nlp/romanization/normalizer.py | TESTED |
| UnifiedQueryNormalizer (NFC + Romanization) | nlp/normalization/normalizer.py | TESTED |
| GraphGroundedTermDetector (Canonical + Multilingual aliases) | nlp/terminology/detector.py | TESTED |
| Concept Mapping Baselines A–D & HybridConceptMapper | nlp/concept_mapping/mapper.py | TESTED |
| NLPPipeline (End-to-end, decoupled from learner state) | nlp/pipeline.py | TESTED |
| EXP-004 Benchmark Dataset & Runner | evaluation/exp004_runner.py | TESTED (Executed) |
| RAG Domain Models & Protocols | rag/models.py, llm/base.py | TESTED |
| MockLLMProvider & OpenAICompatibleProvider | llm/mock.py, llm/providers/openai_compatible.py | TESTED |
| EvidenceSelector (Deduplication, RRF scaling, Budget) | rag/evidence_selector.py | TESTED |
| ContextBuilder (Deterministic headers, Graph context) | rag/context_builder.py | TESTED |
| CitationParser & CitationValidator | rag/citation.py | TESTED |
| GroundingVerifier (Validity, Coverage, Unsupported) | rag/grounding.py | TESTED |
| RAGPipeline (End-to-end orchestrator with fallbacks) | rag/pipeline.py | TESTED |
| ClaimExtractor & SentenceClaimExtractor | rag/claims.py | TESTED |
| EntailmentVerifier & HeuristicLexicalOverlapAssessor | rag/claims.py | TESTED |
| ClaimLevelAuditor (Propositional audit & metadata check) | rag/claims.py | TESTED |
| EXP-005 Benchmark Dataset & Runner | evaluation/exp005_runner.py | TESTED (Executed) |
| ScopeAssessor & HeuristicScopeAssessor | rag/scope.py | TESTED |
| EXP-005B Scope Evaluation Dataset & Runner | evaluation/exp005b_runner.py | TESTED (Executed) |
| Student Knowledge Domain Models | student_model/models.py | TESTED |
| EvidenceBasedKnowledgeEstimator | student_model/estimator.py | TESTED |
| InMemoryKnowledgeStateRepository | student_model/repository.py | TESTED |
| KnowledgeStateService | student_model/service.py | TESTED |
| EXP-006 Student Evaluation Dataset & Runner | evaluation/exp006_runner.py | TESTED (Executed) |
| Assessment Domain Models | assessment/models.py | TESTED |
| MultiSignalCandidatePrioritizer | assessment/prioritizer.py | TESTED |
| HeuristicCandidateGenerator | assessment/candidate.py | TESTED |
| DeterministicItemSelector | assessment/selector.py | TESTED |
| AdaptiveAssessmentService & SessionManager | assessment/service.py, session.py | TESTED |
| EXP-007 Assessment Evaluation Dataset & Runner | evaluation/exp007_runner.py | TESTED (Executed) |
| Recommendation Domain Models | recommendation/models.py | TESTED |
| MultiSignalRecommendationPrioritizer | recommendation/prioritizer.py | TESTED |
| PedagogicalActionSelector | recommendation/action_selector.py | TESTED |
| RecommendationExplanationGenerator | recommendation/explanation.py | TESTED |
| HeuristicRecommendationCandidateGenerator | recommendation/candidate.py | TESTED |
| InMemoryRecommendationRepository | recommendation/repository.py | TESTED |
| RecommendationService | recommendation/service.py | TESTED |
| EXP-008 Recommendation Evaluation Dataset & Runner | evaluation/exp008_runner.py | TESTED (Executed) |
| DashboardApplicationService & REST API | api/service.py, api/routes.py, api/app.py | TESTED |
| Interactive 2D Graph & Student Dashboard SPA | frontend/index.html, frontend/app.js, frontend/style.css | TESTED |
| EXP-009 Dashboard Evaluation Dataset & Runner | evaluation/exp009_runner.py | TESTED (Executed) |
| Statistics Suite (Descriptive, Agreement, Significance) | evaluation/statistics/*.py | TESTED |
| Research Report Generator | evaluation/reporting/generator.py | TESTED |
| Pre-Registered Experiment Registry | data/evaluation/experiment_registry.json | TESTED |
| Human Evaluation Infrastructure & Guidelines | data/evaluation/human/* | TESTED (Protocol Ready) |
| EXP-010 Recommendation Evaluation Runner | evaluation/exp010_runner.py | TESTED (Executed) |
| EXP-011 Assessment Evaluation Runner | evaluation/exp011_runner.py | TESTED (Executed) |
| EXP-012 Dashboard Usability Framework | evaluation/exp012_runner.py | TESTED (HUMAN_DATA_REQUIRED) |
| EXP-013 RAG Human Annotation Framework | evaluation/exp013_runner.py | TESTED (HUMAN_DATA_REQUIRED) |
| EXP-014 Multilingual Human Evaluation Framework | evaluation/exp014_runner.py | TESTED (HUMAN_DATA_REQUIRED) |

## Pending
Empirical human participant trials (EXP-012, EXP-013, EXP-014), Controlled Pre/Post Learning Effectiveness RCT (see docs/LEARNING_EFFECTIVENESS_STUDY.md), Production authentication (OAuth2/JWT).

## Current phase / milestone
Phase 9 complete (Research Evaluation & Human Validation Framework). 250 tests passing, ruff clean, EXP-005A, EXP-005B, EXP-006, EXP-007, EXP-008, EXP-009, EXP-010, EXP-011 passing (100% compliance); EXP-012, EXP-013, EXP-014 framework verified (status: NOT_RUN / HUMAN_DATA_REQUIRED, zero fabricated data). Next: Empirical human participant data collection under institutional IRB approval.

## Known limitations / bugs
- PDF loader is text-layer only; no OCR; legacy-encoded Telugu PDF fonts may extract as garbage — UNVERIFIED on real course files (risk R1).
- Real course PDFs for Signals and Systems not yet provided; evaluation ran on curated academic course benchmark.
- Baseline heuristic extractor extracts secondary/sub-level technical terms as concepts, bounding Concept Precision at ~0.68 on the controlled benchmark.
- Heuristic Romanized Telugu normalizer covers core conversational and question particles; rare dialectal slang or colloquial spelling variations require learned transformer transliteration models (e.g. IndicBART).
- Human evaluation of RAG answers and multilingual quality marked HUMAN_DATA_REQUIRED until native Telugu engineering student panels evaluate real system outputs under approved protocols.
- Lexical overlap in ClaimLevelAuditor is a heuristic baseline and does not replace neural Natural Language Inference (NLI).
- Knowledge estimator parameters and assessment prioritization weights are heuristic defaults; they require validation against empirical student learning curves.
- Question bank is currently a curated fixture set; live LLM question generation is decoupled and deferred to future milestones.
- Student authentication is currently parameterized via `student_id`; enterprise auth (OAuth2/JWT) is deferred.
- No real student learning-effectiveness study has yet been conducted; no causal claims regarding learning gains, cognitive load, or retention are supported.

## Design decisions
See DECISIONS.md (DEC-001…DEC-045).

## Model / dataset choices
- Embedding model: `intfloat/multilingual-e5-small` selected as primary multilingual/Indic embedding model; `paraphrase-multilingual-MiniLM-L12-v2` supported as secondary (DEC-004).
- Vector store: `faiss` (IndexFlatIP) with NumPy in-memory fallback (DEC-003).
- Hybrid retrieval: Reciprocal Rank Fusion (RRF, $k=60$) (DEC-009).
- Evaluation datasets:
  - Retrieval: `data/evaluation/signals_and_systems_eval.json`
  - Knowledge Graph: `data/evaluation/concept_graph_gold.json`
  - Concept Mapping: `data/evaluation/concept_mapping_eval.json`
  - RAG Baseline: `data/evaluation/rag_eval.json` (24 queries)
  - RAG Expanded Benchmark: `data/evaluation/rag_eval_expanded.json` (60 queries)
  - Scope Evaluation: `data/evaluation/scope_eval.json` (45 queries)
  - Student Knowledge Evaluation: `data/evaluation/exp006_student_eval.json` (10 scenarios)
  - Controlled Question Bank: `data/evaluation/signals_systems_items.json` (16 items)
  - Adaptive Assessment Evaluation: `data/evaluation/exp007_assessment_eval.json` (10 scenarios)
  - Recommendation Evaluation: `data/evaluation/exp008_recommendation_eval.json` (12 scenarios)
  - Dashboard & Visualization Evaluation: `data/evaluation/exp009_dashboard_eval.json` (15 scenarios)
  - Recommendation Policy Evaluation: `data/evaluation/exp010_recommendation_eval.json` (10 scenarios)
  - Adaptive Assessment Evaluation: `data/evaluation/exp011_assessment_eval.json` (10 scenarios)
  - Dashboard Usability Protocol Template: `data/evaluation/exp012_usability_template.json`
  - RAG Human Annotation Template: `data/evaluation/exp013_annotation_template.json`
  - Multilingual Human Evaluation Template: `data/evaluation/exp014_multilingual_template.json`

## Experiments
- EXP-001: To be measured.
- EXP-002: EXECUTED (Recall@3: Multilingual E5 1.0000, Multi-MiniLM 0.9250, BM25 0.8500, TF-IDF 0.8500; details in EXPERIMENTS.md).
- EXP-003: EXECUTED (Concept Recall: 1.0000, Relation Recall: 0.9091; details in EXPERIMENTS.md).
- EXP-004: EXECUTED (Hybrid Concept Mapping R@1: 1.0000, MRR: 1.0000, Language Accuracy: 96.88%, Script Accuracy: 100.00%, Terminology F1: 1.0000; details in EXPERIMENTS.md).
- EXP-005: EXECUTED (Recall@5: 0.9524, Multi-chunk recall: 1.0000, Insufficient evidence accuracy: 100%, Citation validity rate: 1.0000, Citation coverage: 1.0000, Unsupported citation rate: 0.0000; details in EXPERIMENTS.md).
- EXP-005A: EXECUTED (60 queries; Recall@5: 0.9200, MRR: 0.5677, Gold Evidence Recall@5: 0.9100; KG ablation: System A 0.8800 vs System B 0.9200 vs System C 0.8800; CM ablation: Raw 0.8800 vs Norm 0.9000 vs Canonical 0.9200; Leakage: PASS; Real LLM: NOT RUN; Human: NOT YET EVALUATED; details in EXPERIMENTS.md).
- EXP-005B: EXECUTED (45 queries; Scope Accuracy: 1.0000, Macro F1: 1.0000, False Acceptance Rate on out-of-scope: 0.0000; Graph candidate displacement: eliminated, Recall@5 preserved at 0.9200; details in EXPERIMENTS.md).
- EXP-006: EXECUTED & AUDITED (10 scenarios; 27/27 invariants passed, 100.0%; Monotonicity, Weak evidence restraint, Strong evidence dominance, Conflicting evidence stability, Provenance & Determinism verified; details in EXPERIMENTS.md).
- EXP-007: EXECUTED (10 scenarios; 21/21 invariants passed, 100.0%; Low knowledge prioritization, Uncertainty representation, Prerequisite importance, Coverage, Difficulty adaptation, Repeat avoidance, Language preference, Provenance, Evidence integration, Student isolation; details in EXPERIMENTS.md).
- EXP-008: EXECUTED (12 scenarios; 33/33 invariants passed, 100.0%; High-confidence weakness, Uncertainty distinction, Prerequisite remediation, Downstream blocker, Recent-action damping, Mastered deprioritization, Conflicting diagnostic, Provenance, Missing source safety, Assessment delegation, Student isolation, Determinism; details in EXPERIMENTS.md).
- EXP-009: EXECUTED (15 scenarios; 15/15 passed, 100.0%; Graph rendering, Concept detail, Student overlay, Unknown state, Uncertain state, Recommendations matching, Assessment delegation & submission, State refresh, Source provenance & missing honesty, Student isolation, Empty state, Error handling, Deterministic mapping; details in EXPERIMENTS.md).
- EXP-010: EXECUTED (10 scenarios; 10/10 passed, 100.0%; Top-1 accuracy: 1.0000, Action selection: 1.0000, Reason code: 1.0000, Ranking consistency: 1.0000; details in EXPERIMENTS.md).
- EXP-011: EXECUTED (10 scenarios; 10/10 passed, 100.0%; Difficulty alignment: 1.0000, Weakness targeting: 1.0000, Repeat avoidance: 1.0000, Closed-loop evidence: 1.0000; details in EXPERIMENTS.md).
- EXP-012: FRAMEWORK VERIFIED (5 tasks, 6 Likert dimensions, status: NOT_RUN / HUMAN_DATA_REQUIRED; details in EXPERIMENTS.md).
- EXP-013: FRAMEWORK VERIFIED (7 Likert dimensions, status: HUMAN_DATA_REQUIRED; details in EXPERIMENTS.md).
- EXP-014: FRAMEWORK VERIFIED (Disaggregated EN, TE, te-Latn, te-en tracks, status: HUMAN_DATA_REQUIRED; details in EXPERIMENTS.md).

## Latest test results (2026-10-06)
`pytest`: 250 passed in 4.04s. `ruff check .`: all checks passed.

## Next recommended task
Empirical human evaluation trials under institutional ethical review.




