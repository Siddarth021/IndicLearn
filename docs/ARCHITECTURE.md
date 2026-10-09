# ARCHITECTURE

## Principles
Clean architecture: pure domain models → services/pipelines → adapters (providers, stores, API). Every external dependency (LLM, embeddings, vector DB, graph store, relational DB) sits behind a `Protocol` so it can be swapped. Config-driven; no hard-coded model names or weights.

## Two mandatory layers
```
ACADEMIC KNOWLEDGE GRAPH (per subject, student-independent)   STUDENT KNOWLEDGE MODEL (per student × concept)
 Concept, ConceptRelation                                     KnowledgeState(score, confidence, evidence_count, last_interaction)
```
The student layer references concept IDs only; it never mutates the academic graph.

## Request flow (target)
```
Question → nlp (lang/script → code-mix → romanization detect → normalize → term detect → intent)
        → concept_mapping (semantic sim + terminology + subject ctx + graph + retrieval) → {concept_id, confidence, matched_terms, supporting_chunks} | AMBIGUOUS
        → retrieval (vector + metadata + concept filter) → rag (context build, grounding policy, LLMProvider) → Answer(+sources, grounded flag)
        → Interaction → Evidence → student_model.estimate_knowledge → KnowledgeState → dynamic graph view
        → quiz / recommendation
```
Original query text is always stored alongside normalized text, candidates, and confidences.

## Interfaces (Protocols)
| Interface | Method sketch | Status | Implementation |
|---|---|---|---|
| `DocumentLoader` | `load(path) -> list[ExtractedPage]` | IMPLEMENTED (Phase 1) | `pypdf`, `python-pptx`, text/md |
| `Chunker` | `chunk_pages(pages, doc_id, subject_id) -> list[DocumentChunk]` | IMPLEMENTED (Phase 1) | `RecursiveCharChunker` |
| `EmbeddingProvider` | `embed_passages(texts)`, `embed_queries(texts)`, `dimension` | IMPLEMENTED (Phase 2) | `SentenceTransformerEmbeddingProvider`, `CachedEmbeddingProvider`, `MockEmbeddingProvider` |
| `VectorStore` | `add(chunks, vecs)`, `search(vec, k, filter)`, `delete(doc_id)`, `save/load` | IMPLEMENTED (Phase 2) | `FaissVectorStore` (IndexFlatIP), `InMemoryVectorStore` |
| `Retriever` | `retrieve(query, top_k, filters) -> list[ScoredChunk]` | IMPLEMENTED (Phase 2) | `BM25Retriever`, `DenseRetriever`, `HybridRetriever` (RRF / Linear) |
| `ConceptExtractor` | `extract(chunks) -> (list[Concept], list[ConceptRelation])` | IMPLEMENTED (Phase 3) | `HeuristicConceptExtractor` |
| `GraphStore` | `add/get/delete_concept`, `add/get/delete_relation`, `neighbors`, `prerequisites`, `save/load` | IMPLEMENTED (Phase 3) | `NetworkXGraphStore` (`networkx.MultiDiGraph`) |
| `LanguageDetector` | `detect(query) -> LanguageDetectionResult` | IMPLEMENTED (Phase 4) | `HeuristicLanguageDetector` |
| `CodeMixDetector` | `analyze(query) -> CodeMixResult` | IMPLEMENTED (Phase 4) | `RuleBasedCodeMixDetector` |
| `TextNormalizer` | `normalize(query, lang) -> NormalizationResult` | IMPLEMENTED (Phase 4) | `UnifiedQueryNormalizer`, `RuleBasedRomanizationNormalizer` |
| `TechnicalTermDetector` | `detect_terms(query, subject_id) -> list[TechnicalTerm]` | IMPLEMENTED (Phase 4) | `GraphGroundedTermDetector` |
| `ConceptMapper` | `map_concept(query, subject_id, top_k) -> ConceptMappingResult` | IMPLEMENTED (Phase 4) | `HybridConceptMapper` (Baselines A–D) |
| `LLMProvider` | `generate(prompt, **opts)`, `generate_messages(messages)` | IMPLEMENTED (Phase 5) | `MockLLMProvider`, `OpenAICompatibleProvider` |
| `EvidenceSelector` | `select_evidence(query, subject_id, results) -> EvidenceSet` | IMPLEMENTED (Phase 5) | `EvidenceSelector` |
| `ContextBuilder` | `build_context(evidence_set, graph_concepts, prereqs) -> str` | IMPLEMENTED (Phase 5) | `ContextBuilder` |
| `CitationValidator` | `validate_citations(answer, evidence_set) -> (all, valid, invalid)` | IMPLEMENTED (Phase 5) | `CitationValidator` |
| `GroundingVerifier` | `verify_grounding(answer, evidence_set) -> GroundingReport` | IMPLEMENTED (Phase 5) | `GroundingVerifier` |
| `ClaimExtractor` | `extract_claims(text) -> list[Claim]` | IMPLEMENTED (Phase 5.1) | `SentenceClaimExtractor` |
| `EntailmentVerifier` | `verify_support(claim, evidence) -> (bool, score, msg)` | IMPLEMENTED (Phase 5.1) | `HeuristicLexicalOverlapAssessor` |
| `ClaimLevelAuditor` | `audit_claims(answer, evidence) -> ClaimAuditResult` | IMPLEMENTED (Phase 5.1) | `ClaimLevelAuditor` |
| `RAGPipeline` | `answer(request) -> RAGResponse` | IMPLEMENTED (Phase 5) | `RAGPipeline` |
| `KnowledgeEstimator` | `update(state, event) -> KnowledgeState` | IMPLEMENTED (Phase 6) | `EvidenceBasedKnowledgeEstimator` |
| `KnowledgeStateRepository` | `get_state`, `save_state`, `get_student_states`, `add_event` | IMPLEMENTED (Phase 6) | `InMemoryKnowledgeStateRepository` |
| `CandidatePrioritizer` | `prioritize_candidate`, `rank_candidates` | IMPLEMENTED (Phase 7A) | `MultiSignalCandidatePrioritizer` |
| `AssessmentCandidateGenerator` | `generate_candidates(student_id, subject_id)` | IMPLEMENTED (Phase 7A) | `HeuristicCandidateGenerator` |
| `AssessmentItemSelector` | `select_item(candidate, available_items)` | IMPLEMENTED (Phase 7A) | `DeterministicItemSelector` |
| `AssessmentItemRepository` | `get_item`, `list_items`, `count` | IMPLEMENTED (Phase 7A) | `InMemoryAssessmentItemRepository` |
| `AssessmentSessionRepository` | `save_session`, `get_session`, `get_student_sessions` | IMPLEMENTED (Phase 7A) | `InMemoryAssessmentSessionRepository` |
| `RecommendationCandidateGenerator` | `generate_candidates(student_id, subject_id)` | IMPLEMENTED (Phase 7B) | `HeuristicRecommendationCandidateGenerator` |
| `RecommendationPrioritizer` | `score_candidate(concept_id, state, ...)` | IMPLEMENTED (Phase 7B) | `MultiSignalRecommendationPrioritizer` |
| `ActionSelector` | `select_action(candidate) -> (action, diff)` | IMPLEMENTED (Phase 7B) | `PedagogicalActionSelector` |
| `RecommendationHistoryRepository` | `record_recommendation`, `get_history`, `get_recent_entries` | IMPLEMENTED (Phase 7B) | `InMemoryRecommendationRepository` |

## Cross-cutting
Config (`config/`), structured logging + pipeline events (`utils/logging.py`), privacy (user isolation, no raw text in logs), failure policy (typed failure results; low confidence → AMBIGUOUS; empty retrieval → no "grounded" claim; LLM down → controlled error).

## Deviation from brief's structure
Added `domain/` package for shared entities (DEC-007).

### 4.5 Scope Assessment & RAG Safety Boundary (Phase 5.2)
The Phase 5.2 hardening introduces `ScopeAssessor` to protect against adversarial keyword injection and generic engineering over-acceptance:
- **Scope States**: `IN_SCOPE`, `OUT_OF_SCOPE`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`.
- **HeuristicScopeAssessor**: Evaluates multi-signal evidence including canonical concept mapping, generic term penalties, concept mapping score margins, domain indicator guards, and retrieval-evidence consistency.
- **Decoupled Graph Context**: Knowledge graph prerequisite information enriches generation prompt context post-retrieval rather than competing in the retrieval candidate pool.

### 4.6 Student Knowledge Model Layer (Phase 6)
Phase 6 establishes the student modeling layer, strictly separated from the canonical academic graph:
- **`KnowledgeState` ($Student \times Concept$)**: Tracks dynamic mastery estimates, confidence, and granular evidence history.
- **Evidence Hierarchy**: Strong assessment evidence (weight 1.0) vs Medium task evidence (0.4) vs Weak interaction evidence (0.1).
- **`EvidenceBasedKnowledgeEstimator`**: Transparent estimator with asymptotic confidence growth, recency decay, and conflicting evidence gating.
- **`KnowledgeStateService`**: Orchestrates state updates, student summaries, weak/uncertain concept identification, provenance explanations, and prerequisite readiness inspection.
- **`KnowledgeStateRepository`**: Protocol with thread-safe `InMemoryKnowledgeStateRepository` guaranteeing student data isolation.

### 4.7 Adaptive Assessment Core Layer (Phase 7A)
Phase 7A implements the closed-loop adaptive assessment subsystem (see `docs/ADAPTIVE_ASSESSMENT.md`):
- **`AssessmentCandidate`**: Concept candidate scored by `MultiSignalCandidatePrioritizer` combining weakness, uncertainty, prerequisite graph importance, coverage need, and recency penalties.
- **`AssessmentItem`**: Curated assessment item preserving source chunk provenance (`source_chunk_ids`) grounding questions in academic material.
- **`DeterministicItemSelector`**: Policy-driven matching on difficulty tiers (EASY, MEDIUM, HARD), language preference (with fallback), and session repeat avoidance.
- **`AdaptiveAssessmentService`**: Orchestrates session state, deterministic response grading, and closed-loop `EvidenceEvent` emission to `KnowledgeStateService`.

### 4.8 Personalized Recommendation & Remediation Planner Layer (Phase 7B)
Phase 7B implements the explainable personalized recommendation subsystem (see `docs/RECOMMENDATION_ENGINE.md`):
- **`RecommendationCandidate`**: Concept scored across seven normalized signals (demonstrated weakness, uncertainty, prerequisite deficit, downstream blocker, review need, assessment gap, and recent-action penalty).
- **`PedagogicalActionSelector`**: Decoupled policy assigning appropriate actions (`REVIEW_CONCEPT`, `REVIEW_PREREQUISITE`, `PRACTICE_CONCEPT`, `TAKE_ASSESSMENT`, `REVISIT_SOURCE`) and difficulty tiers.
- **`RecommendationExplanationGenerator`**: Generates auditable justifications communicating `evidence -> uncertainty -> recommendation -> reason` while strictly avoiding deficit labeling.
- **Provenance Safety**: Explicitly preserves curriculum chunks (`CHUNK_SS_001`), flagging `SOURCE_UNAVAILABLE` when unindexed to prevent hallucinated citations.
- **`RecommendationService`**: Coordinates candidate ranking, history recording for recent-action damping, and assessment delegation to `AdaptiveAssessmentService`.

### 4.9 Visual Knowledge Map & Student Learning Dashboard (Phase 8)
Phase 8 establishes the student-facing presentation and interaction layer (see `docs/STUDENT_DASHBOARD.md`):
- **Strict View-Layer Decoupling**: The presentation layer never calculates scores, mastery statuses, recommendation priorities, or assessment sequences. It is an explainable viewport over backend domain services.
- **REST API Application Layer (`indiclearn.api`)**: FastAPI application backed by `DashboardApplicationService`, providing strongly typed endpoints (`SubjectSummary`, `KnowledgeMapView`, `ConceptDetailView`, `StudentDashboardOverview`, `RecommendationCardView`, `ChunkDetailView`, `AssessmentSessionView`).
- **Interactive 2D Academic Knowledge Map**: Native vector SVG canvas with procedural topological layer layout, directional relation markers (`PREREQUISITE_OF`, `PART_OF`, `RELATED_TO`), pan/zoom controls, and student mastery status rings. Parallel accessible List View for non-visual/keyboard navigation.
- **Concept Detail Drawer & Evidence Transparency**: Surfaces academic prerequisites, model-estimated progress meters, and an audit accordion ("Why is this estimate?") displaying chronological evidence events and weighting rules.
- **Personalized Recommendation Cards**: Directly renders Phase 7B planner output with action triggers (`Take Assessment`, `Review Material`, `Map Focus`).
- **Adaptive Assessment Interface**: Modal dialog delegating item selection to Phase 7A `AdaptiveAssessmentService`, supporting multilingual delivery and real-time state refresh upon response submission.
- **Provenance Grounding & Student Isolation**: Inspects verified textbook chunk citations (`CHUNK_SS_001`), flags `Source unavailable` honestly when unindexed, and strictly isolates student knowledge profiles.

### 4.10 Research Evaluation & Human Validation Framework Layer (Phase 9)
Phase 9 establishes the empirical evaluation architecture, separating software verification, algorithmic benchmarks, and human validation protocols (see `docs/RESEARCH_EVALUATION.md`):
- **Pre-Registered Experiment Registry (`data/evaluation/experiment_registry.json`)**: Machine-readable registry tracking EXP-001 through EXP-014 with hypotheses, sample sizes, random seeds, software versions, and five lifecycle states (`PROPOSED`, `IMPLEMENTED`, `RUN`, `HUMAN_DATA_REQUIRED`, `COMPLETED`).
- **Statistical Suite (`indiclearn.evaluation.statistics`)**:
  - `descriptive.py`: Bessel-corrected variance, IQR, medians, discrete frequency distributions.
  - `agreement.py`: Exact inter-rater agreement (unweighted/weighted Cohen's $\kappa$, Fleiss' $\kappa$ with Landis & Koch interpretation scales); strictly returns `status: NOT_AVAILABLE` on missing data.
  - `significance.py`: Bootstrap percentile CIs ($B=1000$), Student's $t$ CIs, Cohen's $d$, Mann-Whitney $U$, Wilcoxon signed-rank, and Holm-Bonferroni correction for family-wise error control.
- **Reporting Generator (`indiclearn.evaluation.reporting`)**: Structured generator enforcing the scientific separation of Observed Results, Interpretation, Scientific Claims, Limitations, and Reproducibility parameters.
- **Controlled Evaluation Runners**:
  - `EXP-010`: Recommendation policy evaluation (`exp010_runner.py`) across 10 controlled scenarios.
  - `EXP-011`: Adaptive assessment diagnostic policy evaluation (`exp011_runner.py`) across 10 controlled scenarios.
- **Human Evaluation Infrastructure (`data/evaluation/human/`)**:
  - Explicit guidelines (`annotation_guidelines.md`), schemas (`annotator_schema.json`, `participant_schema.json`), and protocol templates (`consent_template.md`).
  - Validation runners (`exp012_runner.py`, `exp013_runner.py`, `exp014_runner.py`) that strictly return `NOT_RUN` / `HUMAN_DATA_REQUIRED` without fabricating participants or ratings.
- **Authentication & Security Production Boundary**:
  - Explicitly documents the transition boundary from pseudonymous research identifiers (`student_id`) to production-grade security:
    $$\text{Authenticated Identity (OAuth2 / OIDC / JWT)} \longrightarrow \text{Authorization Layer (RBAC / ABAC)} \longrightarrow \text{Resolved } \texttt{student\_id} \longrightarrow \texttt{DashboardApplicationService}$$
  - Future production requirements (OAuth2, token introspection, student-resource isolation policies) are cataloged without fake auth mocks.


