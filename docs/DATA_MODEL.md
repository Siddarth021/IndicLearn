# DATA MODEL

Status:
- Document-side entities: **IMPLEMENTED** (`src/indiclearn/domain/documents.py`, Phase 1)
- Academic Knowledge Graph entities: **IMPLEMENTED** (`src/indiclearn/domain/concepts.py`, Phase 3)
- Indic NLP & Concept Mapping entities: **IMPLEMENTED** (`src/indiclearn/nlp/models.py`, Phase 4)
- RAG & Grounded Generation entities: **IMPLEMENTED** (`src/indiclearn/rag/models.py`, Phase 5)
- Student Knowledge Model entities: **IMPLEMENTED** (`src/indiclearn/student_model/models.py`, Phase 6)
- Quiz Engine & Adaptive Testing: **PLANNED** (Phase 7)

| Entity | Implementation Status | Purpose | Key fields |
|---|---|---|---|
| **Document** | IMPLEMENTED | Source uploaded academic file | `document_id`, `subject_id`, `filename`, `file_type`, `sha256`, `size_bytes`, `page_count`, `status`, `error` |
| **DocumentChunk** | IMPLEMENTED | Retrievable slice of academic text | `chunk_id`, `document_id`, `subject_id`, `page_number`, `section`, `text`, `char_start`, `char_end`, `language`, `concept_ids` |
| **Concept** | IMPLEMENTED | Canonical academic concept (objective, subject-scoped) | `concept_id`, `subject_id`, `canonical_name`, `concept_type`, `aliases`, `description`, `source_document_ids`, `source_chunk_ids`, `page_numbers`, `sections`, `confidence`, `metadata` |
| **ConceptRelation** | IMPLEMENTED | Directed, typed curriculum relationship with provenance | `relation_id`, `subject_id`, `source_concept_id`, `target_concept_id`, `relation_type`, `confidence`, `prerequisite_nature`, `source_chunk_ids`, `source_document_ids`, `extraction_method` |
| **LanguageDetectionResult** | IMPLEMENTED | Language and orthographic script identification | `primary_language`, `detected_script`, `confidence`, `script_distribution`, `token_language_counts`, `is_code_mixed` |
| **CodeMixResult** | IMPLEMENTED | Token-level code-mixing breakdown | `original_text`, `spans`, `code_mixing_index`, `matrix_language`, `embedded_languages` |
| **NormalizationResult** | IMPLEMENTED | Non-destructive normalization preserving ambiguity | `original_text`, `canonical_normalized_text`, `candidate_variants`, `is_romanized`, `transliterated_telugu`, `tokens_normalized` |
| **TechnicalTerm** | IMPLEMENTED | Graph-grounded academic engineering term span | `start`, `end`, `matched_text`, `canonical_name`, `concept_id`, `term_type`, `confidence` |
| **ConceptCandidate** | IMPLEMENTED | Ranked concept candidate with provenance and score | `concept_id`, `canonical_name`, `score`, `match_method`, `evidence_spans`, `rank`, `metadata` |
| **ConceptMappingResult** | IMPLEMENTED | Top concept mapping with ambiguity flag | `query`, `subject_id`, `primary_concept`, `candidates`, `is_ambiguous`, `mapping_confidence`, `unmapped_terms` |
| **NLPAnalysisResult** | IMPLEMENTED | Unified query NLP container | `original_text`, `language_detection`, `code_mix`, `normalization`, `technical_terms`, `concept_mapping`, `timestamp` |
| **EvidenceItem** | IMPLEMENTED | Selected academic evidence chunk with prompt ID | `source_id`, `document_id`, `chunk_id`, `text`, `page_number`, `section`, `score`, `retrieval_method`, `concept_ids`, `rank` |
| **EvidenceSet** | IMPLEMENTED | Collection of selected evidence and sufficiency status | `items`, `query`, `subject_id`, `total_candidates`, `insufficient_evidence`, `insufficiency_reason` |
| **Citation** | IMPLEMENTED | Verified citation linking claim to academic source | `source_id`, `document_id`, `chunk_id`, `page_number`, `section`, `text_snippet`, `is_valid`, `validation_error` |
| **CitationSpan** | IMPLEMENTED | Span in generated answer referencing source IDs | `start`, `end`, `raw_text`, `cited_source_ids` |
| **GroundingCheck** | IMPLEMENTED | Quantitative audit metrics of answer grounding | `is_grounded`, `citation_validity_rate`, `citation_coverage`, `unsupported_citation_count`, `unsupported_citation_rate`, `evidence_sufficiency` |
| **GroundingReport** | IMPLEMENTED | Overall answer grounding report and audit status | `check`, `status`, `citations_parsed`, `valid_citations`, `invalid_citations` |
| **GenerationMetadata** | IMPLEMENTED | Telemetry and operational completion metadata | `model_name`, `requested_language`, `generated_language`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `graph_concepts_used` |
| **RAGRequest** | IMPLEMENTED | High-level request container for student question | `query`, `subject_id`, `requested_language`, `top_k`, `max_graph_hops`, `max_graph_concepts`, `min_score_threshold` |
| **RAGResponse** | IMPLEMENTED | Grounded pedagogical answer with full provenance | `query`, `subject_id`, `answer`, `is_answerable`, `status`, `citations`, `evidence`, `concept_ids`, `grounding_report`, `metadata` |
| **EvidenceEvent** | IMPLEMENTED | Atomic interaction or assessment evidence with provenance | `event_id`, `student_id`, `concept_id`, `subject_id`, `event_type`, `strength`, `score`, `timestamp`, `source`, `metadata` |
| **KnowledgeEvidence** | IMPLEMENTED | Granular evidence item preserved in state for audit | `event_id`, `event_type`, `strength`, `score`, `timestamp`, `summary` |
| **KnowledgeState** | IMPLEMENTED | Dynamic mastery estimate ($Student \times Concept$) | `student_id`, `concept_id`, `subject_id`, `knowledge_score`, `confidence`, `evidence_count`, `strong_evidence_count`, `weak_evidence_count`, `mastery_status`, `evidence_history` |
| **StudentKnowledgeSummary** | IMPLEMENTED | Subject-level aggregate mastery summary | `student_id`, `subject_id`, `total_concepts_tracked`, `mastered_count`, `developing_count`, `emerging_count`, `uncertain_count`, `unknown_count`, `mean_knowledge_score`, `mean_confidence` |
| **ConceptStateExplanation** | IMPLEMENTED | Structured explainability report for a state | `student_id`, `concept_id`, `knowledge_score`, `confidence`, `mastery_status`, `reason`, `evidence_summary`, `strong_evidence_count`, `has_conflicting_evidence` |
| **PrerequisiteReadiness** | IMPLEMENTED | Pedagogical readiness analysis without state mutation | `student_id`, `target_concept_id`, `is_ready`, `prerequisite_states`, `unmastered_prerequisites`, `recommendation` |
| **AssessmentItem** | IMPLEMENTED (Phase 7A) | Course-grounded assessment item | `item_id`, `concept_id`, `subject_id`, `question`, `question_type`, `difficulty`, `language`, `options`, `correct_answer`, `explanation`, `source_chunk_ids` |
| **AssessmentCandidate** | IMPLEMENTED (Phase 7A) | Concept evaluated for assessment priority | `concept_id`, `canonical_name`, `subject_id`, `priority_score`, `knowledge_score`, `confidence`, `mastery_status`, `uncertainty`, `weakness`, `prerequisite_relevance`, `coverage_need`, `recency_penalty`, `reasons` |
| **AssessmentResponse** | IMPLEMENTED (Phase 7A) | Student response to an assessment item | `response_id`, `session_id`, `item_id`, `student_id`, `concept_id`, `selected_answer`, `correct`, `feedback`, `answered_at` |
| **AssessmentSession** | IMPLEMENTED (Phase 7A) | Stateful tracking for assessment sequence | `session_id`, `student_id`, `subject_id`, `item_ids`, `current_index`, `responses`, `started_at`, `completed_at`, `is_active` |
| **AssessmentSelectionExplanation** | IMPLEMENTED (Phase 7A) | Full explainability report for selection decision | `candidate`, `item`, `concept_reasons`, `difficulty_reasons`, `item_reasons` |
| **RecommendationCandidate** | IMPLEMENTED (Phase 7B) | Multi-signal scored candidate concept | `concept_id`, `canonical_name`, `knowledge_score`, `confidence`, `mastery_status`, `weakness_score`, `uncertainty_score`, `prerequisite_deficit_score`, `downstream_blocker_score`, `review_need_score`, `assessment_gap_score`, `recent_action_penalty`, `priority_score`, `action_type`, `reason_codes`, `source_chunk_ids`, `source_available` |
| **Recommendation** | IMPLEMENTED (Phase 7B) | Personalized explainable study/practice recommendation | `recommendation_id`, `student_id`, `subject_id`, `concept_id`, `canonical_name`, `action_type`, `priority_score`, `reason_codes`, `explanation`, `knowledge_score`, `confidence`, `mastery_status`, `prerequisite_context`, `source_chunk_ids`, `source_available`, `suggested_difficulty` |
| **RecommendationHistoryEntry** | IMPLEMENTED (Phase 7B) | Audit log entry for recent-action damping | `entry_id`, `student_id`, `subject_id`, `concept_id`, `action_type`, `timestamp`, `metadata` |
| **KnowledgeMapView** | IMPLEMENTED (Phase 8) | Academic graph view with student knowledge overlay | `subject_id`, `student_id`, `nodes`, `edges` |
| **GraphNodeView** | IMPLEMENTED (Phase 8) | Visual node with score, confidence, and mastery status | `id`, `name`, `type`, `aliases`, `knowledge_score`, `confidence`, `mastery_status`, `evidence_count` |
| **GraphEdgeView** | IMPLEMENTED (Phase 8) | Directed visual edge with relation type and label | `id`, `source`, `target`, `relation_type`, `label` |
| **ConceptDetailView** | IMPLEMENTED (Phase 8) | Complete concept drawer data (academic + student + rec) | `concept_id`, `canonical_name`, `prerequisites`, `dependents`, `knowledge_score`, `confidence`, `evidence_history`, `recommended_action` |
| **StudentDashboardOverview** | IMPLEMENTED (Phase 8) | Learning analytics summary for student dashboard | `student_id`, `total_concepts_tracked`, `concepts_with_evidence`, `concepts_assessed`, `mastered_count`, `developing_count`, `emerging_count`, `uncertain_count`, `unknown_count`, `mean_knowledge_score` |
| **RecommendationCardView** | IMPLEMENTED (Phase 8) | Student-facing prioritized recommendation card | `recommendation_id`, `concept_id`, `canonical_name`, `action_type`, `priority_score`, `reason_codes`, `explanation`, `source_available` |
| **ChunkDetailView** | IMPLEMENTED (Phase 8) | Grounded curriculum source chunk with citations | `chunk_id`, `document_id`, `page_number`, `section`, `text`, `language`, `concept_ids` |
| **AssessmentSessionView** | IMPLEMENTED (Phase 8) | Adaptive assessment modal session view | `session_id`, `student_id`, `subject_id`, `is_active`, `current_index`, `total_answered`, `current_item` |
| **AssessmentResultView** | IMPLEMENTED (Phase 8) | Answer submission feedback and updated state | `session_id`, `item_id`, `correct`, `feedback`, `updated_knowledge_score`, `updated_confidence`, `updated_mastery_status`, `next_item` |
| **ExperimentRegistryEntry** | IMPLEMENTED (Phase 9) | Machine-readable experiment pre-registration record | `experiment_id`, `title`, `research_question`, `hypothesis`, `status`, `dataset`, `sample_size`, `metrics`, `procedure`, `seed`, `software_version`, `limitations`, `result_file` |
| **ResearchReport** | IMPLEMENTED (Phase 9) | Auditable scientific reporting entity separating claims | `experiment_id`, `title`, `research_question`, `hypothesis`, `dataset`, `sample_size`, `metrics`, `procedure`, `results`, `interpretation`, `claims`, `limitations`, `reproducibility` |
| **RAGAnnotationRecord** | IMPLEMENTED (Phase 9) | Human evaluation record for RAG generated answers | `sample_id`, `query_id`, `query_text`, `language`, `generated_answer`, `evidence_chunk_ids`, `citations`, `factual_correctness`, `evidence_support`, `citation_correctness`, `pedagogical_usefulness`, `clarity`, `language_quality`, `code_mixed_understanding`, `annotator_id`, `timestamp` |
| **UsabilityResponseRecord** | IMPLEMENTED (Phase 9) | Human usability response record for student dashboard | `session_id`, `participant_id`, `task_outcomes` (completion, time, success, errors for T1-T5), `likert_responses` (Q1-Q6 on 1-5 scale), `open_feedback`, `timestamp` |
| **MultilingualEvaluationRecord** | IMPLEMENTED (Phase 9) | Human multilingual quality record across language tracks | `evaluation_id`, `participant_id`, `language` (`en`, `te`, `te-Latn`, `te-en`), `query_id`, `naturalness_score`, `technical_accuracy_score`, `script_transliteration_clarity`, `code_switch_comfort`, `notes`, `timestamp` |
| **ParticipantProfile** | IMPLEMENTED (Phase 9) | Pseudonymous research participant profile (Zero PII) | `participant_id`, `education_level`, `engineering_branch`, `primary_language`, `telugu_proficiency`, `english_proficiency`, `consent_confirmed`, `created_at` |

### Core Architectural Separation Principle
`Concept` $\ne$ `DocumentChunk` $\ne$ `KnowledgeState` $\ne$ `AssessmentItem` $\ne$ `Recommendation` $\ne$ `DashboardView` $\ne$ `EvaluationArtifact`.
- `Concept` represents canonical subject matter knowledge. It is completely invariant to student progress.
- `KnowledgeState` represents a student's dynamic mastery estimate of a `concept_id`.
- `AssessmentItem` represents an evaluative test question grounded in `source_chunk_ids`.
- `Recommendation` represents a personalized pedagogical action plan derived from evidence without mutating knowledge states.
- `DashboardView` represents strongly-typed presentation projections strictly decoupled from computation.
- `EvaluationArtifact` represents immutable benchmarking results, pre-registered experiment schemas, and empirical human protocols.
- The canonical graph never stores `knowledge_score`, `mastery`, or `student_id`.
- Benchmarks and evaluation registries never fabricate participant identities or empirical Likert distributions.


