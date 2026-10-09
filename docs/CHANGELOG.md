## [Phase 9] - 2026-10-06 (Research Evaluation & Human Validation Framework)
### Added
- `indiclearn.evaluation.statistics.descriptive`: Descriptive statistics functions (`compute_mean`, `compute_median`, `compute_variance` with Bessel's $N-1$ correction, `compute_std`, `compute_quartiles`, `compute_iqr`, `compute_descriptive_summary`, `compute_discrete_distribution`).
- `indiclearn.evaluation.statistics.agreement`: Inter-rater agreement functions (`compute_agreement_percentage`, `compute_cohens_kappa`, `compute_weighted_cohens_kappa` with linear/quadratic weights, `compute_fleiss_kappa` for $>2$ raters, Landis & Koch interpretations; strictly returns `status: NOT_AVAILABLE` on missing data).
- `indiclearn.evaluation.statistics.significance`: Inferential and hypothesis testing functions (`compute_confidence_interval_t`, `compute_bootstrap_ci` with reproducible seeds, `compute_cohens_d`, `mann_whitney_u_test`, `wilcoxon_signed_rank_test`, and `holm_bonferroni_correction`).
- `indiclearn.evaluation.reporting.generator`: `ResearchReport` generator enforcing the scientific separation of Observed Results, Interpretation, Scientific Claims, Limitations, and Reproducibility parameters, supporting JSON and formatted Markdown export.
- Pre-Registered Experiment Registry: `data/evaluation/experiment_registry.json` tracking EXP-001 through EXP-014 with status, hypotheses, sample sizes, procedures, seeds, software versions, and limitations.
- Human Evaluation Infrastructure (`data/evaluation/human/`):
  - Documentation: `README.md`, `annotation_guidelines.md`, `consent_template.md`.
  - JSON Schemas: `annotator_schema.json`, `participant_schema.json`.
  - Templates: `rag_annotation_template.json`, `recommendation_annotation_template.json`, `assessment_annotation_template.json`, `usability_questionnaire.json`.
  - Demonstration fixture: `example_annotation.json` (explicitly marked `EXAMPLE / NOT REAL PARTICIPANT DATA`).
- Controlled Experiment EXP-010 (Recommendation Evaluation):
  - Benchmark Dataset: `data/evaluation/exp010_recommendation_eval.json` (10 scenarios).
  - Runner & Script: `src/indiclearn/evaluation/exp010_runner.py`, `scripts/run_exp010.py`.
  - Artifact: `data/evaluation/exp010_results.json` (10/10 scenarios passed, 100.0% pass rate, 100% Top-1 accuracy, action selection, reason code, ranking consistency).
- Controlled Experiment EXP-011 (Adaptive Assessment Evaluation):
  - Benchmark Dataset: `data/evaluation/exp011_assessment_eval.json` (10 scenarios).
  - Runner & Script: `src/indiclearn/evaluation/exp011_runner.py`, `scripts/run_exp011.py`.
  - Artifact: `data/evaluation/exp011_results.json` (10/10 scenarios passed, 100.0% pass rate, 100% difficulty alignment, weakness targeting, repeat avoidance, closed-loop evidence).
- Human Evaluation Protocol EXP-012 (Dashboard Usability Framework):
  - Template & Runner: `data/evaluation/exp012_usability_template.json`, `src/indiclearn/evaluation/exp012_runner.py`, `scripts/run_exp012.py`.
  - Artifact: `data/evaluation/exp012_results.json` (Framework valid, $N=0$ participants, status: `NOT_RUN` / `HUMAN_DATA_REQUIRED`).
- Human Evaluation Protocol EXP-013 (RAG Human Annotation Framework):
  - Template & Runner: `data/evaluation/exp013_annotation_template.json`, `src/indiclearn/evaluation/exp013_runner.py`, `scripts/run_exp013.py`.
  - Artifact: `data/evaluation/exp013_results.json` (7 Likert dimensions, $N=0$ annotations, status: `HUMAN_DATA_REQUIRED`).
- Human Evaluation Protocol EXP-014 (Multilingual Human Evaluation Framework):
  - Template & Runner: `data/evaluation/exp014_multilingual_template.json`, `src/indiclearn/evaluation/exp014_runner.py`, `scripts/run_exp014.py`.
  - Artifact: `data/evaluation/exp014_results.json` (Disaggregated EN, TE, te-Latn, te-en tracks, $N=0$ evaluations, status: `HUMAN_DATA_REQUIRED`).
- Core Documentation:
  - `docs/RESEARCH_EVALUATION.md`: Formal Research Questions (RQ1-RQ5), registry summary, statistical criteria, research limitation disclosures.
  - `docs/HUMAN_EVALUATION_PROTOCOL.md`: Protocol, participant inclusion/exclusion criteria, privacy safeguards, tasks T1-T9.
  - `docs/LEARNING_EFFECTIVENESS_STUDY.md`: Proposed Pre/Post/Delayed RCT study design with exact power analysis ($N=128$ completing, $N=152$ enrolled), explicitly marked `PROPOSED FUTURE RESEARCH STUDY (NOT YET CONDUCTED)`.
- Architectural Decision Records: Added DEC-043, DEC-044, DEC-045 in `docs/DECISIONS.md`.
- 20 comprehensive unit tests in `tests/unit/test_research_evaluation.py` (250 tests passing repo-wide, Ruff clean).

## [Phase 8] - 2026-10-06 (Visual Knowledge Map & Student Learning Dashboard)
### Added
- `indiclearn.api.schemas`: Strongly typed Pydantic data transfer models (`SubjectSummary`, `GraphNodeView`, `GraphEdgeView`, `KnowledgeMapView`, `ConceptDetailView`, `EvidenceItemView`, `StudentDashboardOverview`, `RecommendationCardView`, `ChunkDetailView`, `AssessmentSessionView`, `AssessmentResultView`).
- `indiclearn.api.service`: `DashboardApplicationService` orchestrating the presentation boundary over backend domain services (`GraphStore`, `KnowledgeStateService`, `AdaptiveAssessmentService`, `RecommendationService`), strictly enforcing zero-calculation view decoupling.
- `indiclearn.api.routes`: FastAPI router exposing clean REST endpoints (`/subjects`, `/subjects/{id}/graph`, `/concepts/{id}`, `/students/{id}/dashboard`, `/students/{id}/recommendations`, `/assessment/sessions`, `/chunks/{id}`, `/demo/load`).
- `indiclearn.api.app`: FastAPI application factory `create_app` mounting static frontend assets (`/static`), OpenAPI docs (`/api/docs`), CORS middleware, and application state.
- `frontend/index.html`: Responsive single-page application structure featuring top controls, overview metrics cards, interactive 2D graph container, accessible list table, concept detail drawer, recommendation cards grid, adaptive assessment modal, and source material inspector.
- `frontend/style.css`: Modern dark glassmorphic stylesheet with Inter and Noto Sans Telugu typography, semantic mastery status palettes, responsive grid layout, and accessible focus states.
- `frontend/app.js`: Lightweight vanilla JavaScript client implementing pan/zoom SVG graph controls, procedural topological DAG layout, interactive node selection, closed-loop assessment session execution, and real-time state synchronization.
- Benchmark Dataset & Runner: `data/evaluation/exp009_dashboard_eval.json`, `src/indiclearn/evaluation/exp009_runner.py`, and CLI script `scripts/run_exp009.py`.
- Benchmark Artifact: `data/evaluation/exp009_results.json` (15/15 scenarios passed, 100.0%).
- 10 comprehensive unit tests in `tests/unit/test_api_and_dashboard.py` (230 total tests passing repo-wide, Ruff clean).
- Documentation: Created `docs/STUDENT_DASHBOARD.md` and added ADRs DEC-040, DEC-041, DEC-042 in `docs/DECISIONS.md`.

## [Phase 7B] - 2026-10-06 (Personalized Recommendations & Remediation Planner)
### Added
- `indiclearn.recommendation.models`: `ActionType` (`REVIEW_CONCEPT`, `REVIEW_PREREQUISITE`, `PRACTICE_CONCEPT`, `TAKE_ASSESSMENT`, `REVISIT_SOURCE`), `ReasonCode` (10 categorical reason codes), `RecommendationCandidate`, `Recommendation`, `RecommendationHistoryEntry`.
- `indiclearn.recommendation.prioritizer`: `RecommendationPrioritizer` protocol and `MultiSignalRecommendationPrioritizer` (evaluating weakness, uncertainty, prerequisite deficit, downstream blocker, review need, assessment gap, and recent-action penalty, with mastered damping).
- `indiclearn.recommendation.action_selector`: `ActionSelector` protocol and `PedagogicalActionSelector` (decoupling priority scoring from action assignment and difficulty tier matching).
- `indiclearn.recommendation.explanation`: `RecommendationExplanationGenerator` (adhering to evidence-based communication `evidence -> uncertainty -> recommendation -> reason` without deficit labeling).
- `indiclearn.recommendation.candidate`: `RecommendationCandidateGenerator` protocol and `HeuristicRecommendationCandidateGenerator` (integrating academic graph relationships and student knowledge states).
- `indiclearn.recommendation.repository`: `RecommendationHistoryRepository` protocol and `InMemoryRecommendationRepository` (tracking student recommendation exposure and recent-action damping).
- `indiclearn.recommendation.service`: `RecommendationService` coordinating recommendation planning, explanation formatting, history tracking, and assessment delegation to `AdaptiveAssessmentService`.
- Source Grounding & Missing-Source Safety: Inspects canonical `Concept.source_chunk_ids` (`CHUNK_SS_001`), explicitly sets `source_available=False` and `SOURCE_UNAVAILABLE` when missing without fabricating citations.
- Configuration: Added `recommendation:` section to `configs/config.yaml` and `RecommendationSettings` to `src/indiclearn/config/settings.py`.
- Benchmark Dataset & Runner: `data/evaluation/exp008_recommendation_eval.json`, `src/indiclearn/evaluation/exp008_runner.py`, and CLI script `scripts/run_exp008.py`.
- Benchmark Artifact: `data/evaluation/exp008_results.json` (12/12 scenarios passed, 33/33 invariants passed, 100.0%).
- 12 comprehensive unit tests in `tests/unit/test_recommendation_engine.py` (220 total tests passing repo-wide, Ruff clean).
- Documentation: Created `docs/RECOMMENDATION_ENGINE.md` and added ADRs DEC-037, DEC-038, DEC-039 in `docs/DECISIONS.md`.

## [Phase 7A] - 2026-10-06 (Adaptive Assessment Core)
### Added
- `indiclearn.assessment.models`: `QuestionType`, `ItemDifficulty`, `AssessmentItem`, `AssessmentCandidate`, `AssessmentResponse`, `AssessmentSession`, `AssessmentSelectionExplanation`.
- `indiclearn.assessment.prioritizer`: `CandidatePrioritizer` protocol and `MultiSignalCandidatePrioritizer` (combining weakness, uncertainty, prerequisite graph blocking, curriculum coverage, and recency penalties).
- `indiclearn.assessment.candidate`: `AssessmentCandidateGenerator` protocol and `HeuristicCandidateGenerator`.
- `indiclearn.assessment.selector`: `AssessmentItemSelector` protocol and `DeterministicItemSelector` (difficulty tier matching, language preference with English fallback, session repeat avoidance).
- `indiclearn.assessment.items`: Deterministic grading logic `evaluate_response` for MCQ, True/False, and short-answer questions.
- `indiclearn.assessment.repository`: `AssessmentItemRepository`, `AssessmentSessionRepository` protocols with `InMemoryAssessmentItemRepository` (JSON file loader) and `InMemoryAssessmentSessionRepository`.
- `indiclearn.assessment.session`: `AssessmentSessionManager` for active session tracking and response logging.
- `indiclearn.assessment.service`: `AdaptiveAssessmentService` orchestrating the adaptive assessment loop with closed-loop `EvidenceEvent` emission to `KnowledgeStateService`.
- Controlled Question Bank: `data/evaluation/signals_systems_items.json` (16 academic items grounded with `source_chunk_ids` in course chunks).
- Benchmark Dataset & Runner: `data/evaluation/exp007_assessment_eval.json`, `src/indiclearn/evaluation/exp007_runner.py`, and `scripts/run_exp007.py`.
- Benchmark Artifact: `data/evaluation/exp007_results.json` (10/10 scenarios passed, 21/21 invariants passed, 100.0%).
- 10 comprehensive unit tests in `tests/unit/test_adaptive_assessment.py` (208 total tests passing repo-wide, Ruff clean).
- Documentation: `docs/ADAPTIVE_ASSESSMENT.md` and ADRs DEC-034, DEC-035, DEC-036 in `docs/DECISIONS.md`.

## [Phase 6.1] - 2026-10-06 (Student Model Consistency & Research Hardening Audit)
### Fixed
- `indiclearn.student_model.estimator`: Resolved implementation fallthrough bug in `EvidenceBasedKnowledgeEstimator.compute_mastery_status` where low scores ($S < 0.50$) incorrectly defaulted to `MasteryStatus.DEVELOPING` instead of `MasteryStatus.EMERGING`.
- `indiclearn.student_model.estimator`: Enforced strict evaluative-evidence gating (`strong_count == 0 and medium_count == 0` forces `MasteryStatus.UNKNOWN`) preventing observational interactions (questions asked, views) from leaking into competency statuses (`EMERGING`/`DEVELOPING`).
- `indiclearn.student_model.service`: Refined pedagogical explanations in `explain_concept_state` to provide distinct rationales for foundational remediation ($S < 0.50$) vs nascent passing signals ($S \ge 0.50$) in `EMERGING` status.
- `data/evaluation/exp006_student_eval.json`: Corrected benchmark expectation for `SCEN_002` ($S = 0.0750, C = 0.7438$) to `EMERGING`, `SCEN_003` (pure questions) to `UNKNOWN`, and `SCEN_004` (failed quiz dominance) to `EMERGING`.

### Added
- 5 new targeted audit unit tests in `tests/unit/test_student_model.py` (total 15 student model tests, 198 tests repo-wide):
  - `test_audit_scen002_incorrect_assessment_is_emerging_not_developing`
  - `test_audit_monotonicity_sequences`
  - `test_audit_weak_evidence_cannot_produce_mastery_or_weakness`
  - `test_audit_score_confidence_separation`
  - `test_audit_uncertainty_conflicting_evidence`
- EXP-006 post-audit verification: 10/10 scenarios passed, 27/27 invariants passed (100.0%).
- ADR DEC-033 in `docs/DECISIONS.md`.

## [Phase 6] - 2026-10-06
### Added
- `indiclearn.student_model.models`: `KnowledgeState`, `EvidenceEvent`, `KnowledgeEvidence`, `MasteryStatus`, `EvidenceStrength`, `EvidenceType`, `StudentKnowledgeSummary`, `ConceptStateExplanation`, `PrerequisiteReadiness`.
- `indiclearn.student_model.estimator`: `KnowledgeEstimator` protocol and `EvidenceBasedKnowledgeEstimator` implementation.
- `indiclearn.student_model.repository`: `KnowledgeStateRepository` protocol and `InMemoryKnowledgeStateRepository`.
- `indiclearn.student_model.service`: `KnowledgeStateService` managing state transitions, explanations, summaries, and prerequisite readiness.
- Three-tier evidence hierarchy: Strong (quiz/assessment, weight 1.0) vs Medium (task, 0.4) vs Weak (question asked/views, 0.1).
- Asymptotic confidence saturation and recency decay weighting (30-day half-life).
- Conflicting evidence detection mapping inconsistent assessment results to `MasteryStatus.UNCERTAIN`.
- EXP-006 evaluation benchmark dataset (`data/evaluation/exp006_student_eval.json`), runner (`src/indiclearn/evaluation/exp006_runner.py`), and CLI script (`scripts/run_exp006.py`).
- EXP-006 results artifact (`data/evaluation/exp006_results.json`): 10/10 scenarios passed, 25/25 invariants passed (100%).
- 10 unit tests in `tests/unit/test_student_model.py` covering cold start, weak evidence restraint, strong evidence dominance, conflicting evidence, provenance, student isolation, concept isolation, prerequisite readiness, and determinism.
- Documentation: `docs/KNOWLEDGE_MODEL.md` and ADRs DEC-030, DEC-031, DEC-032 in `docs/DECISIONS.md`.


## [Phase 5.2] - 2026-10-06
### Added
- `indiclearn.rag.scope`: `ScopeAssessor` protocol, `HeuristicScopeAssessor`, and `ScopeDecision` model.
- Generic term down-weighting for broad engineering terminology (`system`, `signal`, `filter`, `transform`, etc.).
- Score margin calculation in `ConceptMappingResult` and `ConceptMapper` (`score_margin < 0.08` flags ambiguity).
- Concept-evidence consistency verification checking if retrieved chunks contain mapped concepts.
- Domain indicator guards for out-of-scope engineering disciplines (digital filter design, control systems, image compression, FFT decimation algorithms).
- Multilingual safe fallback refusal messages for `OUT_OF_SCOPE` and `AMBIGUOUS_SCOPE` across English, Telugu script, Romanized Telugu, and code-mixed.
- Scope evaluation dataset: `data/evaluation/scope_eval.json` (45 queries: 18 in-scope, 18 out-of-scope, 9 ambiguous).
- EXP-005B evaluation runner (`src/indiclearn/evaluation/exp005b_runner.py`) and execution script (`scripts/run_exp005b.py`).
- 10 comprehensive unit tests in `tests/unit/test_rag_scope.py` covering strong/weak in-scope, adversarial out-of-scope, under-specified queries, Telugu queries, concept bait attacks, and evidence consistency.

### Changed
- `indiclearn.rag.pipeline`: Decoupled knowledge graph prerequisite context from the direct retrieval candidate pool. Graph context now enriches the prompt context post-retrieval without candidate displacement.
- `indiclearn.rag.grounding`: Extended `verify_grounding` to accept fallback statuses `OUT_OF_SCOPE` and `AMBIGUOUS_SCOPE`.

### Fixed
- Fixed adversarial false acceptance rate on out-of-scope queries (dropped from 50% in EXP-005A to **0.0000** in EXP-005B).
- Resolved graph context retrieval candidate displacement regression (Recall@5 preserved at **0.9200** instead of dropping to 0.8800).

# CHANGELOG

## 2026-10-06 (Phase 5.1 — RAG Evaluation, Real-LLM Validation & Benchmark Expansion)
- feat: Expanded benchmark dataset `data/evaluation/rag_eval_expanded.json` with 60 curated queries across 12 categories, 4 language varieties (`en`, `te`, `te-Latn`, `te-en`), and 3 difficulty levels with independent gold chunk annotations.
- feat: Claim-level analysis module `src/indiclearn/rag/claims.py` implementing `Claim` model, `ClaimExtractor` protocol, `SentenceClaimExtractor` heuristic baseline, `EntailmentVerifier` protocol, `HeuristicLexicalOverlapAssessor`, and `ClaimLevelAuditor`.
- feat: Advanced citation attack defense modes in `MockLLMProvider` (`multiple_invalid_citations`, `unrelated_citation`, `invalid_page_citation`, `wrong_document_citation`, `citation_omitted`) and comprehensive unit tests in `tests/unit/test_rag_citation_attacks.py`.
- feat: Enhanced `OpenAICompatibleProvider` with environment variable resolution (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`), `from_env()` constructor, and `is_configured` property.
- feat: Experiment runner `Exp005aRunner` in `src/indiclearn/evaluation/exp005a_runner.py` computing disaggregated retrieval metrics, Knowledge Graph contribution ablation, Concept Mapping query formulation ablation, benchmark leakage verification, and answerability matrix.
- feat: Executable CLI script `scripts/run_exp005a.py` with `--provider mock|real` and offline CI guard (`REAL LLM EVALUATION: NOT RUN` when credentials are absent).
- feat: Formal 5-dimension human evaluation schema (0-2 scale on Correctness, Groundedness, Relevance, Completeness, Language Appropriateness) reporting `HUMAN EVALUATION: NOT YET EVALUATED`.
- feat: Versioned evaluation results in `data/evaluation/exp005a_results.json` while strictly preserving historical baseline `data/evaluation/exp005_results.json`.
- test: 12 new unit and integration tests (173 total tests passing repo-wide); Ruff check clean.
- docs: Updated `RAG_PIPELINE.md`, `EXPERIMENTS.md` (EXP-005A empirical results), `DECISIONS.md` (DEC-025 to DEC-027), `ARCHITECTURE.md`, `TODO.md`, `PROJECT_CONTEXT.md`, `HANDOFF.md`, and `HANDOFF_SUMMARY.txt`.

## 2026-10-06 (Phase 5 — RAG Pipeline & Grounded Answer Generation)
- feat: Strongly typed domain models for RAG (`EvidenceItem`, `EvidenceSet`, `ContextDocument`, `Citation`, `CitationSpan`, `GroundingCheck`, `GroundingReport`, `GenerationMetadata`, `RAGRequest`, `RAGResponse`) in `src/indiclearn/rag/models.py`.
- feat: Decoupled `LLMProvider` Protocol, `LLMMessage`, and `LLMResponse` in `src/indiclearn/llm/base.py`.
- feat: Deterministic `MockLLMProvider` simulating grounded answers with source citations, insufficient evidence responses, unsupported claims, and fabricated citation modes (`src/indiclearn/llm/mock.py`).
- feat: Standard library `OpenAICompatibleProvider` for optional real API endpoint execution without third-party vendor SDK lock-in (`src/indiclearn/llm/providers/openai_compatible.py`).
- feat: `EvidenceSelector` with deduplication, RRF score scale adaptation, character budgeting, and sequential `[S1]`, `[S2]` labeling (`src/indiclearn/rag/evidence_selector.py`).
- feat: `ContextBuilder` generating structured academic context and controlled curriculum knowledge graph prerequisites (`src/indiclearn/rag/context_builder.py`).
- feat: Prompt templates, multilingual formatting policies, and standard localized insufficient evidence fallback responses (`src/indiclearn/rag/policies.py`).
- feat: `CitationParser` and `CitationValidator` verifying bracketed source citations against prompt `EvidenceSet` (`src/indiclearn/rag/citation.py`).
- feat: `GroundingVerifier` computing Citation Validity Rate, Sentence Citation Coverage, and Unsupported Citation Rate (`src/indiclearn/rag/grounding.py`).
- feat: End-to-end `RAGPipeline` orchestrating NLP concept mapping, curriculum graph expansion, hybrid retrieval, evidence selection, generation, and citation auditing (`src/indiclearn/rag/pipeline.py`).
- feat: Curated benchmark dataset `data/evaluation/rag_eval.json` with 24 queries across 7 categories (direct factual, concept explanation, relationship/multi-chunk, Telugu script, Romanized Telugu, code-mixed, and deliberate out-of-scope insufficient-evidence queries) with independent gold chunk annotations.
- feat: Experiment runner `Exp005Runner` and executable evaluation script `scripts/run_exp005.py`.
- test: 25 new unit and integration tests (161 total tests passing repo-wide); ruff clean.
- docs: Updated `RAG_PIPELINE.md`, `EXPERIMENTS.md` (EXP-005 empirical results), `DECISIONS.md` (DEC-021 to DEC-024), `ARCHITECTURE.md`, `DATA_MODEL.md`, `TODO.md`, `PROJECT_CONTEXT.md`, `HANDOFF.md`, `HANDOFF_SUMMARY.txt`.

## 2026-10-06 (Phase 4 — Indic NLP & Cross-Lingual Concept Mapping)
- feat: Strongly typed Pydantic models for Indic NLP entities (`src/indiclearn/nlp/models.py`)
- feat: Protocols `LanguageDetector`, `CodeMixDetector`, `TextNormalizer`, `RomanizationNormalizer`, `TechnicalTermDetector`, `ConceptMapper`
- feat: Deterministic, Unicode-aware `HeuristicLanguageDetector` (evaluating script ratios, Romanized functional markers, and engineering vocab)
- feat: `RuleBasedCodeMixDetector` identifying token spans, matrix/embedded languages, and computing Code-Mixing Index (CMI)
- feat: Non-destructive `RuleBasedRomanizationNormalizer` preserving raw student text while generating ranked candidate variants with phonetic standardization and Telugu script transliteration
- feat: `UnifiedQueryNormalizer` combining NFC normalization, whitespace cleaning, and Romanized variation mapping
- feat: `GraphGroundedTermDetector` extracting canonical concepts, textbook aliases, acronyms, and Telugu aliases directly from `GraphStore`
- feat: 4 Concept Mapping Baselines: `ExactConceptMapper` (Baseline A), `LexicalConceptMapper` (Baseline B), `EmbeddingConceptMapper` (Baseline C), `HybridConceptMapper` (Baseline D fusing terminology, lexical, and semantic evidence)
- feat: End-to-end `NLPPipeline` orchestrator strictly decoupled from student knowledge state / mastery scores
- feat: Cross-lingual gold evaluation benchmark `data/evaluation/concept_mapping_eval.json` (32 queries across EN, TE, TE-Latn, TE-EN)
- feat: Experiment runner `Exp004Runner` and script `scripts/run_exp004.py` evaluating EXP-004
- feat: Comprehensive architectural guide `docs/NLP_PIPELINE.md`
- test: 34 new unit and integration tests (136 total passing tests across all phases); ruff clean
- docs: Updated `NLP_PIPELINE.md`, `EXPERIMENTS.md`, `DECISIONS.md` (DEC-017 through DEC-020), `ARCHITECTURE.md`, `DATA_MODEL.md`, `TODO.md`, `PROJECT_CONTEXT.md`, `HANDOFF.md`, `HANDOFF_SUMMARY.txt`

## 2026-10-06 (Phase 3 — Concept Extraction + Academic Knowledge Graph)
- feat: `Concept` and `ConceptRelation` domain models with exact source provenance (`src/indiclearn/domain/concepts.py`)
- feat: Categorization enums `ConceptType` (concept, subconcept, property, method, formula, example), `RelationType`, and `PrerequisiteNature` (`EXPLICIT_PREREQUISITE` vs `INFERRED_PREREQUISITE`)
- feat: `ConceptNormalizer` with acronym expansion, hyphenated title casing, noise phrase rejection, and multilingual alias management (`src/indiclearn/concepts/normalizer.py`)
- feat: Deterministic concept identity generation (`cpt_{subject_id}_{slug}`) and relation ID generation
- feat: `GraphStore` Protocol defining storage-agnostic knowledge graph operations (`src/indiclearn/knowledge_graph/base.py`)
- feat: `NetworkXGraphStore` adapter on `MultiDiGraph` with BFS transitive prerequisite/dependent queries and deterministic JSON persistence (`src/indiclearn/knowledge_graph/networkx_store.py`)
- feat: `validate_graph` integrity validator checking referential integrity, self-loops, duplicate edges, provenance presence, and prerequisite DAG cycle detection (`src/indiclearn/knowledge_graph/validation.py`)
- feat: `HeuristicConceptExtractor` rule-based baseline extracting concepts and relations from headings, definitional grammar, property terms, and domain rules (`src/indiclearn/concepts/extractor.py`)
- feat: `ConceptService` orchestrating extraction, graph insertion, graph validation, and linking `concept_ids` into `DocumentChunk` records (`src/indiclearn/concepts/service.py`)
- feat: Curated Signals and Systems manual gold standard benchmark (`data/evaluation/concept_graph_gold.json`)
- feat: Graph evaluation metrics (`evaluate_concepts`, `evaluate_relations`, `compute_prf1`) in `src/indiclearn/evaluation/graph_metrics.py`
- feat: `Exp003Runner` and benchmark script `scripts/run_exp003.py` executing EXP-003
- feat: Architectural ontology reference in `docs/KNOWLEDGE_GRAPH.md`
- test: 42 new unit & integration tests (102 total passing tests across Phase 1, 2, 3); ruff clean
- docs: Updated `PROJECT_CONTEXT.md`, `DATA_MODEL.md`, `ARCHITECTURE.md`, `DECISIONS.md` (DEC-011 through DEC-016), `EXPERIMENTS.md` (EXP-003 results), `TODO.md`, `HANDOFF.md`, `HANDOFF_SUMMARY.txt`

## 2026-10-06 (Phase 2 — Embeddings + Retrieval + Evaluation)
- feat: `EmbeddingProvider` protocol and `EmbeddingResult` domain types (`src/indiclearn/embeddings/base.py`)
- feat: `SentenceTransformerEmbeddingProvider` with GPU acceleration and asymmetric prefixes (`src/indiclearn/embeddings/sentence_transformer.py`)
- feat: `EmbeddingCache` and `CachedEmbeddingProvider` with version/model-aware SHA-256 keys (`src/indiclearn/embeddings/cache.py`)
- feat: `MockEmbeddingProvider` for deterministic testing (`src/indiclearn/embeddings/mock.py`)
- feat: `VectorStore` protocol, `RetrievalResult`, and typed exceptions (`src/indiclearn/retrieval/base.py`, `exceptions.py`)
- feat: `FaissVectorStore` with IndexFlatIP, metadata sidecars, persistence, and version verification (`src/indiclearn/retrieval/faiss_vector_store.py`)
- feat: `InMemoryVectorStore` NumPy exact cosine fallback (`src/indiclearn/retrieval/in_memory_vector_store.py`)
- feat: `BM25Retriever` with Okapi scoring and Indic Unicode mark-aware tokenization (`src/indiclearn/retrieval/bm25.py`)
- feat: `TFIDFRetriever` baseline with scikit-learn (`src/indiclearn/retrieval/tfidf.py`)
- feat: `DenseRetriever` connecting embeddings and vector storage (`src/indiclearn/retrieval/dense.py`)
- feat: `HybridRetriever` supporting Reciprocal Rank Fusion (RRF) and Min-Max linear normalization (`src/indiclearn/retrieval/hybrid.py`)
- feat: `RetrievalPipeline` orchestrating NFC preprocessing, retrieval, and event logging (`src/indiclearn/retrieval/pipeline.py`)
- feat: IR evaluation metrics (Recall@k, Precision@k, MRR, nDCG@k) and multi-dimensional aggregator (`src/indiclearn/evaluation/retrieval_metrics.py`)
- feat: Curated cross-lingual Signals and Systems evaluation benchmark with 20 queries across 4 linguistic categories (`data/evaluation/signals_and_systems_eval.json`)
- feat: Automated benchmark script `scripts/run_exp002.py` and experiment runner (`src/indiclearn/evaluation/exp002_runner.py`)
- feat: Expanded `Settings` and central YAML configs (`configs/config.yaml`, `configs/models.yaml`)
- test: 60 tests passing (27 new unit/integration tests); ruff clean
- docs: Updated `PROJECT_CONTEXT.md`, `DECISIONS.md` (DEC-003, DEC-004, DEC-009, DEC-010), `EXPERIMENTS.md` (EXP-002 results), `RAG_PIPELINE.md`, `TODO.md`, `HANDOFF.md`

## 2026-10-04 (Phase 1 — Ingestion + Preprocessing + Chunking)
- feat: repo skeleton, packaging, configs (config.yaml/models.yaml/languages.yaml), .env.example
- feat: settings loader with env overrides
- feat: structured JSON logging and pipeline-event helper
- feat: domain models for documents/chunks
- feat: text cleaning (NFC, ZWJ/ZWNJ-safe)
- feat: recursive char chunker with offsets and page/section metadata
- feat: TXT/MD/PDF/PPTX loaders and IngestionService (dedupe, failure recording)
- test: 33 tests (unit + integration) passing; ruff clean
- docs: Phase 0 documents
