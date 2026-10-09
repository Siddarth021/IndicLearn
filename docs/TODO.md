## Phase 9: Research Evaluation & Human Validation Framework — COMPLETE (2026-10-06)
- [x] Pre-registered Experiment Registry (`data/evaluation/experiment_registry.json`) tracking EXP-001 through EXP-014.
- [x] Comprehensive statistical evaluation suite (`indiclearn.evaluation.statistics`):
  - `descriptive.py`: Mean, median, Bessel's sample variance, std, quartiles, IQR, min, max, discrete distributions.
  - `agreement.py`: Raw agreement percentage, unweighted/weighted Cohen's $\kappa$, Fleiss' $\kappa$ with Landis & Koch scales, missing data protection returning `status: NOT_AVAILABLE`.
  - `significance.py`: Bootstrap percentile CIs ($B=1000$), Student's $t$ CIs, Cohen's $d$, Mann-Whitney $U$, Wilcoxon signed-rank, Holm-Bonferroni correction.
- [x] Research report generator (`indiclearn.evaluation.reporting`) strictly separating Observed Results, Interpretation, Scientific Claims, Limitations, and Reproducibility parameters.
- [x] Human evaluation infrastructure (`data/evaluation/human/`):
  - `README.md`, `annotation_guidelines.md`, `annotator_schema.json`, `participant_schema.json`, `consent_template.md`.
  - Templates: `rag_annotation_template.json`, `recommendation_annotation_template.json`, `assessment_annotation_template.json`, `usability_questionnaire.json`.
  - Example annotation: `example_annotation.json` (explicitly marked `EXAMPLE / NOT REAL PARTICIPANT DATA`).
- [x] EXP-010 Controlled Recommendation Evaluation benchmark (`data/evaluation/exp010_recommendation_eval.json`, 10 scenarios):
  - Runner (`src/indiclearn/evaluation/exp010_runner.py`), script (`scripts/run_exp010.py`), results (`data/evaluation/exp010_results.json`).
  - Result: 10/10 scenarios passed (100.0% pass rate, 100% Top-1 accuracy, action selection, reason code, ranking consistency).
- [x] EXP-011 Controlled Adaptive Assessment Evaluation benchmark (`data/evaluation/exp011_assessment_eval.json`, 10 scenarios):
  - Runner (`src/indiclearn/evaluation/exp011_runner.py`), script (`scripts/run_exp011.py`), results (`data/evaluation/exp011_results.json`).
  - Result: 10/10 scenarios passed (100.0% pass rate, 100% difficulty alignment, weakness targeting, repeat avoidance, closed-loop evidence).
- [x] EXP-012 Dashboard Usability Evaluation Framework (`data/evaluation/exp012_usability_template.json`):
  - Runner (`src/indiclearn/evaluation/exp012_runner.py`), script (`scripts/run_exp012.py`), results (`data/evaluation/exp012_results.json`).
  - Result: Framework verified (5 tasks, 6 Likert dimensions), $N=0$ participants, status: `NOT_RUN` (`reason: "HUMAN_DATA_REQUIRED"`).
- [x] EXP-013 RAG Human Annotation Framework (`data/evaluation/exp013_annotation_template.json`):
  - Runner (`src/indiclearn/evaluation/exp013_runner.py`), script (`scripts/run_exp013.py`), results (`data/evaluation/exp013_results.json`).
  - Result: Framework verified (7 Likert dimensions), $N=0$ annotations, status: `HUMAN_DATA_REQUIRED`.
- [x] EXP-014 Multilingual Human Evaluation Framework (`data/evaluation/exp014_multilingual_template.json`):
  - Runner (`src/indiclearn/evaluation/exp014_runner.py`), script (`scripts/run_exp014.py`), results (`data/evaluation/exp014_results.json`).
  - Result: Disaggregated EN, TE, te-Latn, te-en tracks verified, $N=0$ evaluations, status: `HUMAN_DATA_REQUIRED`.
- [x] Zero fabricated human data: Absolutely no invented participants, Likert ratings, or simulated p-values.
- [x] Documented formal protocols:
  - `docs/RESEARCH_EVALUATION.md`: Formal Research Questions (RQ1-RQ5), registry table, statistical criteria.
  - `docs/HUMAN_EVALUATION_PROTOCOL.md`: Human study protocol, inclusion/exclusion criteria, ethics, tasks T1-T9.
  - `docs/LEARNING_EFFECTIVENESS_STUDY.md`: Proposed Pre/Post/Delayed RCT study design with exact power analysis ($N=128$ completing, $N=152$ enrolled).
- [x] Add ADRs DEC-043, DEC-044, DEC-045 in `docs/DECISIONS.md`.
- [x] Add 20 comprehensive unit tests in `tests/unit/test_research_evaluation.py` (250 tests passing repo-wide, Ruff clean).
- [x] Verify zero regression across historical experiments (EXP-005A, EXP-005B, EXP-006, EXP-007, EXP-008, EXP-009).

## Phase 8: Visual Knowledge Map & Student Learning Dashboard — COMPLETE (2026-10-06)
- [x] First frontend audit: Determined frontend did not previously exist; selected self-contained FastAPI + static vanilla SPA architecture.
- [x] Create API schemas (`SubjectSummary`, `GraphNodeView`, `GraphEdgeView`, `KnowledgeMapView`, `ConceptDetailView`, `EvidenceItemView`, `StudentDashboardOverview`, `RecommendationCardView`, `ChunkDetailView`, `AssessmentSessionView`, `AssessmentResultView`).
- [x] Implement `DashboardApplicationService` mediating between HTTP API and backend domain services (strict view decoupling, zero client-side calculation).
- [x] Implement FastAPI REST routes (`/subjects`, `/subjects/{id}/graph`, `/concepts/{id}`, `/students/{id}/dashboard`, `/students/{id}/recommendations`, `/assessment/sessions`, `/chunks/{id}`, `/demo/load`).
- [x] Implement interactive 2D SVG Knowledge Map with procedural topological auto-layout, zoom, pan, directional markers, and mastery rings.
- [x] Implement accessible tabular List View fallback for non-visual, screen-reader, and keyboard navigation.
- [x] Implement Concept Detail Drawer & Evidence Transparency accordion ("Why is this estimate?") with historical events and model weighting notes.
- [x] Implement Personalized Recommendation Cards consuming Phase 7B output with action triggers (`Take Assessment`, `Review Material`, `Map Focus`).
- [x] Implement Adaptive Assessment Modal delegating item selection to Phase 7A `AdaptiveAssessmentService` with multilingual support and immediate reactive state refresh.
- [x] Implement Source Material Modal preserving grounded textbook citations (`CHUNK_SS_001`) with explicit `Source unavailable` handling.
- [x] Implement student privacy and knowledge state isolation boundaries across all endpoints.
- [x] Create EXP-009 evaluation benchmark (`data/evaluation/exp009_dashboard_eval.json`, 15 scenarios).
- [x] Implement EXP-009 runner (`src/indiclearn/evaluation/exp009_runner.py`) and script (`scripts/run_exp009.py`).
- [x] Achieve 100% scenario pass rate in EXP-009 (15/15 scenarios passed).
- [x] Add 10 comprehensive unit tests in `tests/unit/test_api_and_dashboard.py` (230 tests passing repo-wide, Ruff clean).
- [x] Create `docs/STUDENT_DASHBOARD.md` and add ADRs DEC-040, DEC-041, DEC-042 in `docs/DECISIONS.md`.

## Phase 7B: Personalized Recommendations & Remediation Planner — COMPLETE (2026-10-06)
- [x] Create Recommendation domain models (`ActionType`, `ReasonCode`, `RecommendationCandidate`, `Recommendation`, `RecommendationHistoryEntry`).
- [x] Implement `RecommendationPrioritizer` protocol and `MultiSignalRecommendationPrioritizer` evaluating seven signals (weakness, uncertainty, prerequisite deficit, downstream blocker, review need, assessment gap, recent-action penalty) with mastered damping ($0.35$).
- [x] Implement `ActionSelector` protocol and `PedagogicalActionSelector` decoupling prioritization from action assignment across 5 explicit actions (`REVIEW_CONCEPT`, `REVIEW_PREREQUISITE`, `PRACTICE_CONCEPT`, `TAKE_ASSESSMENT`, `REVISIT_SOURCE`).
- [x] Implement `RecommendationExplanationGenerator` adhering to evidence-based communication (`evidence -> uncertainty -> recommendation -> reason`) without deficit labeling.
- [x] Implement `RecommendationCandidateGenerator` protocol and `HeuristicRecommendationCandidateGenerator` integrating knowledge states and canonical graph relationships.
- [x] Implement `RecommendationHistoryRepository` and `InMemoryRecommendationRepository` for recent-action tracking.
- [x] Implement `RecommendationService` coordinating recommendations, explanation formatting, and assessment delegation to Phase 7A `AdaptiveAssessmentService`.
- [x] Implement provenance grounding and missing-source safety (`source_available=False`, `SOURCE_UNAVAILABLE`, zero hallucinated citations).
- [x] Add configuration parameters to `configs/config.yaml` and `src/indiclearn/config/settings.py` under `recommendation:`.
- [x] Create synthetic evaluation benchmark (`data/evaluation/exp008_recommendation_eval.json`, 12 scenarios).
- [x] Implement EXP-008 runner (`src/indiclearn/evaluation/exp008_runner.py`) and script (`scripts/run_exp008.py`).
- [x] Achieve 100% scenario and invariant pass rate in EXP-008 (12/12 scenarios, 33/33 invariants).
- [x] Add 12 comprehensive unit tests in `tests/unit/test_recommendation_engine.py` (220 tests passing repo-wide, Ruff clean).
- [x] Create `docs/RECOMMENDATION_ENGINE.md` and add ADRs DEC-037, DEC-038, DEC-039 in `docs/DECISIONS.md`.

## Phase 7A: Adaptive Assessment Core — COMPLETE (2026-10-06)
- [x] Create Assessment domain models (`AssessmentItem`, `AssessmentCandidate`, `AssessmentResponse`, `AssessmentSession`, `AssessmentSelectionExplanation`).
- [x] Implement `CandidatePrioritizer` protocol and `MultiSignalCandidatePrioritizer` (weakness, uncertainty, prerequisite blocking, coverage, recency).
- [x] Implement `AssessmentCandidateGenerator` protocol and `HeuristicCandidateGenerator` (graph concepts + student states, excluding mastered).
- [x] Implement `AssessmentItemSelector` protocol and `DeterministicItemSelector` (difficulty matching, language preference with fallback, repeat avoidance).
- [x] Implement deterministic response evaluator `evaluate_response` for MCQ, True/False, and short answer.
- [x] Implement `AssessmentItemRepository` and `AssessmentSessionRepository` protocols with in-memory adapters and JSON loaders.
- [x] Implement `AssessmentSessionManager` for active session tracking and response logging.
- [x] Implement `AdaptiveAssessmentService` with closed-loop evidence integration emitting `EvidenceEvent` to `KnowledgeStateService`.
- [x] Create curated controlled question bank (`data/evaluation/signals_systems_items.json`, 16 grounded items).
- [x] Create synthetic evaluation benchmark (`data/evaluation/exp007_assessment_eval.json`, 10 scenarios).
- [x] Implement EXP-007 runner (`src/indiclearn/evaluation/exp007_runner.py`) and script (`scripts/run_exp007.py`).
- [x] Achieve 100% scenario and invariant pass rate in EXP-007 (10/10 scenarios, 21/21 invariants).
- [x] Add 10 comprehensive unit tests in `tests/unit/test_adaptive_assessment.py` (208 tests passing repo-wide, Ruff clean).
- [x] Create `docs/ADAPTIVE_ASSESSMENT.md` and add ADRs DEC-034, DEC-035, DEC-036 in `docs/DECISIONS.md`.

## Phase 6.1: Student Model Consistency & Research Hardening Audit — COMPLETE (2026-10-06)
- [x] Audit `SCEN_002` ($S = 0.0750, C = 0.7438$): Identified fallthrough bug in `estimator.py` assigning `DEVELOPING` for $S < 0.50$; corrected to `EMERGING`.
- [x] Audit all mastery status rules: Formulated definitive, mutually exclusive 5-status partition table.
- [x] Enforce weak evidence isolation: Inquisitive questions/views alone strictly preserve `UNKNOWN` status (`strong_count == 0 and medium_count == 0`).
- [x] Audit monotonicity: Confirmed repeated correct strictly increases score, repeated incorrect strictly decreases score, alternating signals converge to balance.
- [x] Audit score/confidence separation: Verified $S=0.70, C=0.15$ (`UNKNOWN`) vs $S=0.70, C=0.85$ (`DEVELOPING`).
- [x] Audit uncertainty handling: Verified conflicting assessment sequences route to `UNCERTAIN` without flip-flopping.
- [x] Add 5 targeted audit unit tests in `tests/unit/test_student_model.py` (total 15 student model tests, 198 tests passing).
- [x] Re-run EXP-006: 10/10 scenarios passed, 27/27 invariants passed (100.0%).
- [x] Add ADR DEC-033 in `docs/DECISIONS.md` and update all research documentation.

## Phase 6: Student Knowledge Modeling & Evidence-Based Personalization — COMPLETE (2026-10-06)
- [x] Create student model domain entities (`KnowledgeState`, `EvidenceEvent`, `KnowledgeEvidence`, `MasteryStatus`, `EvidenceStrength`, `EvidenceType`).
- [x] Implement `KnowledgeEstimator` protocol and `EvidenceBasedKnowledgeEstimator` with recency decay and explicit uncertainty.
- [x] Enforce research rule: Questions asked are weak evidence, never proof of zero knowledge.
- [x] Implement `KnowledgeStateRepository` protocol and `InMemoryKnowledgeStateRepository` with student isolation.
- [x] Implement `KnowledgeStateService` with summary, weak concepts, explanation, and prerequisite readiness.
- [x] Add configuration parameters to `configs/config.yaml` and `StudentModelSettings`.
- [x] Create synthetic evaluation benchmark (`data/evaluation/exp006_student_eval.json`, 10 scenarios).
- [x] Implement EXP-006 runner (`src/indiclearn/evaluation/exp006_runner.py`) and script (`scripts/run_exp006.py`).
- [x] Achieve 100% scenario and invariant pass rate in EXP-006 (`data/evaluation/exp006_results.json`).
- [x] Add 10 comprehensive unit tests in `tests/unit/test_student_model.py` (193 tests passing).
- [x] Create `docs/KNOWLEDGE_MODEL.md` and add ADRs DEC-030, DEC-031, DEC-032.


## Phase 5.2: RAG Safety, Scope Control & Retrieval Hardening — COMPLETE (2026-10-06)
- [x] Implement `ScopeAssessor` protocol and `HeuristicScopeAssessor` with generic term penalties.
- [x] Add 4 explicit scope states (`IN_SCOPE`, `OUT_OF_SCOPE`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`).
- [x] Implement concept mapping score margin calculation and ambiguity threshold.
- [x] Implement concept-evidence consistency checking in retrieval.
- [x] Add domain indicator guards for adversarial queries (Kaiser, Kalman, DCT, FFT, LMS, PID).
- [x] Decouple knowledge graph prerequisite enrichment from retrieval candidate pool.
- [x] Create hard-negative scope evaluation benchmark (`data/evaluation/scope_eval.json`, 45 queries).
- [x] Implement EXP-005B runner and execute evaluation (`scripts/run_exp005b.py`).
- [x] Achieve 0.0000 False Acceptance Rate on out-of-scope queries (0/18 false acceptances).
- [x] Resolve graph retrieval regression: Recall@5 preserved at 0.9200.
- [x] All 183 unit and integration tests passing, Ruff clean.

# TODO

## Phase 2 — Embeddings + Retrieval + Evaluation (COMPLETED)
- [x] `EmbeddingProvider` / `VectorStore` / `Retriever` / `Reranker` Protocols
- [x] Structured domain types: `EmbeddingResult`, `RetrievalResult`, `IndexVersionMismatchError`
- [x] Indic-aware BM25 baseline retriever (with Unicode combining marks support)
- [x] TF-IDF baseline retriever (scikit-learn)
- [x] `FaissVectorStore` (IndexFlatIP, persistence, sidecar metadata, version verification)
- [x] `InMemoryVectorStore` (NumPy exact cosine similarity fallback)
- [x] Embedding cache (`EmbeddingCache`, `CachedEmbeddingProvider`)
- [x] `DenseRetriever` connecting EmbeddingProvider and VectorStore
- [x] `HybridRetriever` supporting Reciprocal Rank Fusion (RRF) and Min-Max linear normalization
- [x] `RetrievalPipeline` with query preprocessing and structured logging
- [x] Reusable IR evaluation metrics: Recall@k, Precision@k, MRR, nDCG@k
- [x] Synthetic/manual Signals and Systems evaluation benchmark (20 queries, 10 chunks across 4 language formulations)
- [x] Executable experiment runner `scripts/run_exp002.py` and empirical validation

## Phase 3 — Concepts & Academic Knowledge Graph (COMPLETED)
- [x] `Concept` and `ConceptRelation` domain entities with full provenance (`domain/concepts.py`)
- [x] Controlled relation ontology with explicit direction semantics (`RelationType`)
- [x] Deterministic concept ID generation (`cpt_{subject_id}_{slug}`)
- [x] Concept normalizer with acronym resolution, noise rejection, and multilingual aliases (`concepts/normalizer.py`)
- [x] `GraphStore` Protocol decoupling graph operations from engine (`knowledge_graph/base.py`)
- [x] `NetworkXGraphStore` adapter using `MultiDiGraph` with BFS prerequisite/dependent traversals (`knowledge_graph/networkx_store.py`)
- [x] Deterministic JSON persistence preserving typing, attributes, and provenance
- [x] Baseline deterministic extractor `HeuristicConceptExtractor` (headings, definitional grammar, property rules)
- [x] `ConceptService` orchestrating extraction, graph storage, and chunk linking
- [x] Graph integrity validation (`validate_graph`: referential integrity, self-loops, duplicate edges, cycle warnings)
- [x] Curated Signals and Systems manual gold standard (`data/evaluation/concept_graph_gold.json`)
- [x] Executable experiment runner `Exp003Runner` and script `scripts/run_exp003.py` (Recall 1.0000 on concepts, 0.9091 on relations)
- [x] 42 new unit & integration tests (102 total tests passing)
- [x] Comprehensive architectural & ontology guide (`docs/KNOWLEDGE_GRAPH.md`)

## Phase 4 — Indic NLP & Cross-Lingual Concept Mapping (COMPLETED)
- [x] Strongly typed Pydantic domain models (`src/indiclearn/nlp/models.py`)
- [x] Modular protocols: `LanguageDetector`, `CodeMixDetector`, `TextNormalizer`, `TechnicalTermDetector`, `ConceptMapper`
- [x] Heuristic language and script identification (`HeuristicLanguageDetector`, 96.88% language, 100% script accuracy)
- [x] Token-level code-mixing span analyzer and CMI metric (`RuleBasedCodeMixDetector`)
- [x] Non-destructive Romanized Telugu normalization preserving ambiguity and candidate variants (`UnifiedQueryNormalizer`, `RuleBasedRomanizationNormalizer`)
- [x] Graph-grounded technical term detection synchronized with Academic Knowledge Graph (`GraphGroundedTermDetector`)
- [x] 4 Concept Mapping Baselines: Baseline A (Exact), Baseline B (Lexical), Baseline C (Embedding), Baseline D (Hybrid)
- [x] End-to-end `NLPPipeline` orchestrator with student knowledge model decoupling
- [x] Controlled evaluation benchmark `data/evaluation/concept_mapping_eval.json` (32 gold queries across EN, TE, TE-Latn, TE-EN)
- [x] Executable experiment runner `Exp004Runner` and script `scripts/run_exp004.py`
- [x] Comprehensive test suite: 34 new tests (136 total passing tests, ruff clean)
- [x] Comprehensive documentation in `docs/NLP_PIPELINE.md` and `docs/EXPERIMENTS.md`

## Phase 5 — RAG Pipeline & Grounded Answer Generation (COMPLETED)
- [x] Strongly typed RAG domain models (`EvidenceItem`, `EvidenceSet`, `Citation`, `CitationSpan`, `GroundingCheck`, `GroundingReport`, `RAGRequest`, `RAGResponse`, `GenerationMetadata`)
- [x] Decoupled `LLMProvider` protocol with offline `MockLLMProvider` and stdlib `OpenAICompatibleProvider`
- [x] `EvidenceSelector` with deduplication, RRF score scale adaptation, character budgeting, and sequential `[S1]`, `[S2]` labeling
- [x] Controlled curriculum graph expansion (direct prerequisites and related concepts without student state)
- [x] Deterministic `ContextBuilder` preserving full provenance headers
- [x] Strict evidence-only prompt policies and system instructions (`policies.py`)
- [x] `CitationParser` and `CitationValidator` verifying references against provided `EvidenceSet`
- [x] `GroundingVerifier` computing Citation Validity Rate, Sentence Coverage, and Unsupported Citation Rate
- [x] Multilingual insufficient evidence detection and localized fallback messages (`en`, `te`, `te-Latn`, `te-en`)
- [x] Curated RAG evaluation dataset `data/evaluation/rag_eval.json` (24 queries across 7 categories with independent gold chunk annotations)
- [x] Executable experiment runner `Exp005Runner` and script `scripts/run_exp005.py`
- [x] 25 new unit and integration tests (161 total tests passing repo-wide, Ruff clean)
- [x] Comprehensive documentation in `docs/RAG_PIPELINE.md` and `docs/EXPERIMENTS.md`

## Phase 5.1 — RAG Evaluation, Real-LLM Validation & Benchmark Expansion (COMPLETED)
- [x] Expanded benchmark dataset `data/evaluation/rag_eval_expanded.json` (60 queries across 12 categories, 4 languages, 3 difficulties, independent gold annotations)
- [x] Benchmark leakage audit method verifying no verbatim question-to-chunk overlaps (PASS)
- [x] Knowledge Graph contribution ablation (System A vs System B vs System C)
- [x] Concept Mapping query formulation ablation (Raw vs Normalized vs Concept-Expanded)
- [x] Disaggregated retrieval metrics by language (`en`, `te`, `te-Latn`, `te-en`) and difficulty (`easy`, `medium`, `hard`)
- [x] Claim-level propositional extraction interface (`Claim`, `ClaimExtractor`, `SentenceClaimExtractor`)
- [x] Entailment protocol (`EntailmentVerifier`) with heuristic lexical overlap assessor (`HeuristicLexicalOverlapAssessor`)
- [x] Claim-level citation auditor (`ClaimLevelAuditor`) detecting unrelated evidence and metadata mismatches
- [x] Citation attack unit tests (`tests/unit/test_rag_citation_attacks.py`) testing `[S999]`, invalid page numbers, wrong documents, omitted citations, and unrelated evidence
- [x] Real LLM provider validation (`OpenAICompatibleProvider` with `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) with safe offline opt-in guard
- [x] Formal human evaluation rubric definition (0-2 scale across 5 dimensions, reported as `NOT YET EVALUATED`)
- [x] Versioned experiment runner `Exp005aRunner`, script `scripts/run_exp005a.py`, and results `data/evaluation/exp005a_results.json`
- [x] 12 new unit tests (173 total tests passing repo-wide, Ruff clean)

## Next (Phase 6 — Interactive Assessment & Student Knowledge Modeling)
- [ ] Student knowledge state model ($Student \times Concept$)
- [ ] Evidence-based knowledge estimator from quiz and interaction telemetry
- [ ] Diagnostic assessment generation grounded in academic knowledge graph
- [ ] Dynamic personalized knowledge graph view (read-only overlay on canonical graph)
- [ ] Adaptive remediation recommendations

## Open risks
- [ ] R1 Telugu PDF text extraction quality (legacy fonts, scans) — awaiting real course files from user (UNRESOLVED)
- [x] R2 Romanized Telugu ambiguity / spelling variation — addressed via non-destructive candidate normalization (Phase 4)
- [ ] R3 Concept extraction quality & hallucinated relations (needs human validation on larger corpus)
- [x] R4 No labelled Telugu/code-mixed technical data — addressed by creating gold benchmarks `concept_mapping_eval.json` and `rag_eval.json`
- [ ] R5 Student-model heuristic has no ground truth; needs real quiz data (Phase 6/7)
- [x] R6 LLM grounding on Telugu output quality — audited via heuristic citation verification and fallback policies (Phase 5)

## Later
Phases 7–13 per brief. Also: run mypy, persistent dedupe repository.
