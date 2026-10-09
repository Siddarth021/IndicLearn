## Phase 9 Handoff Status — 2026-10-06 (Research Evaluation & Human Validation Framework)
- **Completed**: Phase 9 (Research Evaluation & Human Validation Framework).
- **Test Suite**: 250 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **Benchmarking & Validation Matrix**:
  - `EXP-005A`: R@5 = 0.9200, MRR = 0.5677, Gold Recall@5 = 0.9100 (**PASS**).
  - `EXP-005B`: Scope Accuracy = 1.0000, False Acceptance Rate = 0.0000 (**PASS**).
  - `EXP-006`: 10/10 scenarios, 27/27 invariants (**PASS**).
  - `EXP-007`: 10/10 scenarios, 21/21 invariants (**PASS**).
  - `EXP-008`: 12/12 scenarios, 33/33 invariants (**PASS**).
  - `EXP-009`: 15/15 scenarios (**PASS**).
  - `EXP-010`: 10/10 scenarios passed (100.0% scenario pass rate, 100% Top-1 recommendation accuracy) (**PASS**).
  - `EXP-011`: 10/10 scenarios passed (100.0% scenario pass rate, 100% difficulty alignment & repeat avoidance) (**PASS**).
  - `EXP-012`: Usability questionnaire and 5-task protocol verified; $N=0$ participants; status: `NOT_RUN` (`reason: "HUMAN_DATA_REQUIRED"`).
  - `EXP-013`: RAG 7-dimension annotation framework verified; $N=0$ annotations; status: `HUMAN_DATA_REQUIRED`.
  - `EXP-014`: Multilingual evaluation framework (EN, TE, te-Latn, te-en) verified; $N=0$ evaluations; status: `HUMAN_DATA_REQUIRED`.
- **Key Artifacts Added**:
  - `src/indiclearn/evaluation/statistics/` (`descriptive.py`, `agreement.py`, `significance.py`, `__init__.py`)
  - `src/indiclearn/evaluation/reporting/` (`generator.py`, `__init__.py`)
  - `data/evaluation/human/` (`README.md`, `annotation_guidelines.md`, `annotator_schema.json`, `participant_schema.json`, `consent_template.md`, `rag_annotation_template.json`, `recommendation_annotation_template.json`, `assessment_annotation_template.json`, `usability_questionnaire.json`, `example_annotation.json`)
  - `data/evaluation/experiment_registry.json`
  - Controlled datasets & results: `exp010_recommendation_eval.json`, `exp010_results.json`, `exp011_assessment_eval.json`, `exp011_results.json`, `exp012_usability_template.json`, `exp012_results.json`, `exp013_annotation_template.json`, `exp013_results.json`, `exp014_multilingual_template.json`, `exp014_results.json`.
  - Runners & CLI scripts: `exp010_runner.py`..`exp014_runner.py` and `scripts/run_exp010.py`..`scripts/run_exp014.py`.
  - Unit tests: `tests/unit/test_research_evaluation.py` (20 comprehensive tests).
  - Documentation: `docs/RESEARCH_EVALUATION.md`, `docs/HUMAN_EVALUATION_PROTOCOL.md`, `docs/LEARNING_EFFECTIVENESS_STUDY.md`, and ADRs DEC-043, DEC-044, DEC-045 in `docs/DECISIONS.md`.
- **Zero Fabrication Guarantee**: No synthetic human ratings, participants, Likert scores, or p-values were invented.
- **Recommended Next Milestone**: Empirical human data collection with native Telugu-speaking engineering students under formal IRB/ethical approval.

## Phase 8 Handoff Status — 2026-10-06 (Visual Knowledge Map & Student Learning Dashboard)
- **Completed**: Phase 8 (Visual Knowledge Map & Student Learning Dashboard).
- **Test Suite**: 230 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-009 Results**: 15/15 scenarios passed (100.0% scenario pass rate).
- **Key Artifacts Added**:
  - `src/indiclearn/api/` (`schemas.py`, `service.py`, `routes.py`, `app.py`, `__init__.py`)
  - `frontend/` (`index.html`, `style.css`, `app.js`)
  - `data/evaluation/exp009_dashboard_eval.json` & `data/evaluation/exp009_results.json`
  - `src/indiclearn/evaluation/exp009_runner.py` & `scripts/run_exp009.py`
  - `tests/unit/test_api_and_dashboard.py` (10 comprehensive tests)
  - `docs/STUDENT_DASHBOARD.md` & ADRs DEC-040, DEC-041, DEC-042 in `docs/DECISIONS.md`.
- **Historical Experiments**: EXP-005A, EXP-005B, EXP-006, EXP-007, EXP-008 remain 100% reproducible with unchanged metrics.
- **Next Phase**: Phase 9 (Human Evaluation, Multilingual Usability Pilot & Enterprise Deployment Hardening) is recommended.

## Phase 7B Handoff Status — 2026-10-06 (Personalized Recommendations & Remediation Planner)
- **Completed**: Phase 7B (Personalized Recommendations & Remediation Planner).
- **Test Suite**: 220 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-008 Results**: 12/12 scenarios passed, 33/33 invariants passed (100.0% invariant pass rate).
- **Key Artifacts Added**:
  - `src/indiclearn/recommendation/` (`models.py`, `prioritizer.py`, `action_selector.py`, `explanation.py`, `candidate.py`, `repository.py`, `service.py`, `__init__.py`)
  - `data/evaluation/exp008_recommendation_eval.json` & `data/evaluation/exp008_results.json`
  - `src/indiclearn/evaluation/exp008_runner.py` & `scripts/run_exp008.py`
  - `tests/unit/test_recommendation_engine.py` (12 comprehensive tests)
  - `docs/RECOMMENDATION_ENGINE.md` & ADRs DEC-037, DEC-038, DEC-039 in `docs/DECISIONS.md`.
  - Configuration updates in `configs/config.yaml` and `src/indiclearn/config/settings.py`.
- **Historical Experiments**: EXP-005A, EXP-005B, EXP-006, EXP-007 remain 100% reproducible with unchanged metrics.
- **Next Phase**: Phase 8 (Visual Knowledge Map & Student Learning Dashboard) is cleared to begin.

## Phase 7A Handoff Status — 2026-10-06 (Adaptive Assessment Core)
- **Completed**: Phase 7A (Adaptive Assessment Core).
- **Test Suite**: 208 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-007 Results**: 10/10 scenarios passed, 21/21 invariants passed (100.0% invariant pass rate).
- **Key Artifacts Added**:
  - `src/indiclearn/assessment/` (`models.py`, `prioritizer.py`, `candidate.py`, `selector.py`, `items.py`, `repository.py`, `session.py`, `service.py`, `__init__.py`)
  - `data/evaluation/signals_systems_items.json` (16 curated items with chunk provenance)
  - `data/evaluation/exp007_assessment_eval.json` & `data/evaluation/exp007_results.json`
  - `src/indiclearn/evaluation/exp007_runner.py` & `scripts/run_exp007.py`
  - `tests/unit/test_adaptive_assessment.py`
  - `docs/ADAPTIVE_ASSESSMENT.md` & ADRs DEC-034, DEC-035, DEC-036 in `docs/DECISIONS.md`.
- **Historical Experiments**: EXP-005A, EXP-005B, EXP-006 remain 100% reproducible with unchanged metrics.
- **Next Phase**: Phase 7B (Personalized Recommendations & Practice Planner) is ready to begin.

## Phase 6.1 Handoff Status — 2026-10-06 (Audit & Consistency Hardening)
- **Completed**: Phase 6.1 (Student Model Consistency & Research Hardening Audit).
- **Test Suite**: 198 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-006 Post-Audit Results**: 10/10 scenarios passed, 27/27 invariants passed (100.0% invariant pass rate).
- **Audit Resolutions**:
  - SCEN_002: Fixed code fallthrough bug in `estimator.py` that yielded `DEVELOPING` for $S = 0.0750$; updated to `EMERGING`.
  - SCEN_003: Fixed weak evidence status leak; non-evaluative interactions (asking questions) strictly yield `UNKNOWN`.
  - SCEN_004: Aligned benchmark expectation with authoritative rule ($S = 0.2897, C = 0.3329$ yields `EMERGING`).
- **Historical Experiments**: EXP-005A, EXP-005B remain 100% reproducible with unchanged metrics.
- **Next Phase**: Phase 7 (Adaptive Assessment Generation) is cleared to begin.

## Phase 6 Handoff Status — 2026-10-06
- **Completed**: Phase 6 (Student Knowledge Modeling & Evidence-Based Personalization).
- **Test Suite**: 193 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-006 Results**: 10/10 scenarios passed, 25/25 invariants passed (100.0% invariant pass rate).
- **Historical Experiments**: EXP-005A, EXP-005B remain 100% reproducible with unchanged metrics.
- **Key Artifacts Added**:
  - `src/indiclearn/student_model/` (`models.py`, `estimator.py`, `repository.py`, `service.py`, `__init__.py`)
  - `data/evaluation/exp006_student_eval.json` & `data/evaluation/exp006_results.json`
  - `src/indiclearn/evaluation/exp006_runner.py` & `scripts/run_exp006.py`
  - `tests/unit/test_student_model.py`
  - `docs/KNOWLEDGE_MODEL.md`


## Phase 5.2 Handoff Status — 2026-10-06
- **Completed**: Phase 5.2 (RAG Safety, Scope Control & Retrieval Hardening).
- **Test Suite**: 183 unit and integration tests passing (`python -m pytest -q`).
- **Linter**: Zero Ruff issues (`python -m ruff check .`).
- **EXP-005B Results**: Scope Accuracy = 1.0000, Macro F1 = 1.0000, False Acceptance Rate on out-of-scope = 0.0000 (0/18 false acceptances).
- **Graph Ablation Rerun**: Recall@5 preserved at 0.9200 (regression eliminated).
- **Next Phase**: Phase 6 (Student Knowledge Model) is ready to begin.

# INDICLEARN HANDOFF

## Current Date
2026-10-06

## Current Project Phase
Phase 5 complete (RAG Pipeline & Grounded Answer Generation + EXP-005); Phase 6 (Interactive Assessment & Student Knowledge Modeling) ready to start.

## Current Objective
Phase 6: Interactive Assessment & Student Knowledge Modeling. Estimate per-student concept-level knowledge states ($Student \times Concept$) from quiz and interaction telemetry without mutating the canonical academic knowledge graph.

## What Has Been Completed
- **Phase 0**: Architecture and research documentation.
- **Phase 1**: Ingestion pipeline, text cleaning (NFC + ZWJ/ZWNJ preservation), recursive char chunker with exact offsets, TXT/MD/PDF/PPTX loaders, deduplication, error recording.
- **Phase 2**:
  - `EmbeddingProvider` Protocol and `EmbeddingResult` domain types.
  - `SentenceTransformerEmbeddingProvider` (GPU-accelerated, prefix support) & `MockEmbeddingProvider`.
  - `EmbeddingCache` with model-aware SHA-256 keys.
  - `VectorStore` Protocol, `FaissVectorStore` (IndexFlatIP + sidecar + version checking) & `InMemoryVectorStore` fallback.
  - Lexical baselines: `BM25Retriever` (Indic Unicode tokenization) and `TFIDFRetriever`.
  - `DenseRetriever` and `HybridRetriever` (RRF + Linear MinMax).
  - `RetrievalPipeline` with preprocessing and structured JSON logging.
  - IR Evaluation metrics: Recall@k, Precision@k, MRR, nDCG@k.
  - Curated cross-lingual Signals and Systems benchmark (`data/evaluation/signals_and_systems_eval.json`).
  - Executable experiment runner `scripts/run_exp002.py` and empirical findings in `docs/EXPERIMENTS.md`.
- **Phase 3**:
  - `Concept` and `ConceptRelation` domain entities with exact source provenance (`src/indiclearn/domain/concepts.py`).
  - Controlled relation ontology with explicit direction semantics (`RelationType`).
  - Deterministic concept ID strategy: `cpt_{subject_id}_{slugify(canonical_name)}`.
  - `ConceptNormalizer` with acronym expansion, hyphenated title casing, noise phrase rejection, and multilingual alias management (`src/indiclearn/concepts/normalizer.py`).
  - `GraphStore` Protocol defining storage-agnostic knowledge graph operations (`src/indiclearn/knowledge_graph/base.py`).
  - `NetworkXGraphStore` adapter using `MultiDiGraph` with BFS transitive prerequisite and dependent traversals (`src/indiclearn/knowledge_graph/networkx_store.py`).
  - Deterministic JSON graph persistence preserving IDs, metadata, types, and provenance (`save_to_json` / `load_from_json`).
  - `validate_graph` integrity validator checking referential integrity, self-loops, duplicate edges, provenance presence, and prerequisite DAG cycle detection (`src/indiclearn/knowledge_graph/validation.py`).
  - `HeuristicConceptExtractor` rule-based baseline extracting concepts and relations from headings, definitional grammar, property terms, and domain rules (`src/indiclearn/concepts/extractor.py`).
  - `ConceptService` orchestrating extraction, graph insertion, graph validation, and linking `concept_ids` into `DocumentChunk` records (`src/indiclearn/concepts/service.py`).
  - Curated Signals and Systems manual gold standard benchmark (`data/evaluation/concept_graph_gold.json`).
  - Graph evaluation metrics (`evaluate_concepts`, `evaluate_relations`, `compute_prf1`) in `src/indiclearn/evaluation/graph_metrics.py`.
  - `Exp003Runner` and benchmark script `scripts/run_exp003.py` executing EXP-003.
  - Comprehensive architectural ontology reference in `docs/KNOWLEDGE_GRAPH.md`.
- **Phase 4**:
  - Strongly typed Pydantic models for Indic NLP entities (`src/indiclearn/nlp/models.py`).
  - Protocols: `LanguageDetector`, `CodeMixDetector`, `TextNormalizer`, `RomanizationNormalizer`, `TechnicalTermDetector`, `ConceptMapper`.
  - `HeuristicLanguageDetector` with Unicode character code-points analysis, Romanized Telugu function word lexicon, English common words, and technical term detection (96.88% language, 100% script accuracy).
  - `RuleBasedCodeMixDetector` identifying token spans, language tags, matrix/embedded languages, and Code-Mixing Index (CMI).
  - Non-destructive `RuleBasedRomanizationNormalizer` standardizing Romanized Telugu spelling variants (`enti/yenti/emiti`, `ante/antey`, `cheyyi/cheyi`, `avuthundi/avutundi`), generating transliterated Telugu script representations, and preserving raw text with candidate variant scoring.
  - `UnifiedQueryNormalizer` integrating Unicode NFC cleaning, whitespace collapse, and Romanization normalization.
  - `GraphGroundedTermDetector` matching canonical concepts, textbook aliases, uppercase acronyms (`LTI`, `ROC`), and Telugu script aliases directly from `GraphStore` (Precision: 1.0, Recall: 1.0, F1: 1.0).
  - 4 Concept Mapping Baselines:
    - Baseline A (`ExactConceptMapper`): Exact string matching (Recall@1 = 0.0000 on conversational questions).
    - Baseline B (`LexicalConceptMapper`): Normalized lexical matching (Recall@1 = 1.0000).
    - Baseline C (`EmbeddingConceptMapper`): Dense semantic embedding ranking (Recall@1 = 0.0625).
    - Baseline D (`HybridConceptMapper`): Multi-signal hybrid mapper combining terminology, lexical, and semantic evidence (Recall@1 = 1.0000, MRR = 1.0000).
  - `NLPPipeline` orchestrator strictly decoupled from learner knowledge state / mastery scores.
  - Curated cross-lingual evaluation benchmark `data/evaluation/concept_mapping_eval.json` (32 gold queries across EN, TE, TE-Latn, TE-EN).
  - Executable experiment runner `Exp004Runner` and script `scripts/run_exp004.py`.
  - Comprehensive architectural guide in `docs/NLP_PIPELINE.md`.
- **Phase 5**:
  - Strongly typed Pydantic models for RAG (`EvidenceItem`, `EvidenceSet`, `Citation`, `CitationSpan`, `GroundingCheck`, `GroundingReport`, `RAGRequest`, `RAGResponse`, `GenerationMetadata`).
  - Decoupled `LLMProvider` Protocol, `LLMMessage`, and `LLMResponse`.
  - Deterministic `MockLLMProvider` simulating grounded answers with source citations, insufficient evidence responses, unsupported claims, and fabricated citation modes.
  - Standard library `OpenAICompatibleProvider` for optional external API endpoint execution without third-party vendor SDK lock-in.
  - `EvidenceSelector` with deduplication, RRF score scale adaptation, character budgeting, and sequential `[S1]`, `[S2]` labeling.
  - Controlled curriculum graph expansion providing direct prerequisites and related concept definitions.
  - Deterministic `ContextBuilder` preserving full provenance headers.
  - Prompt policies and system instructions enforcing evidence-only answering (`policies.py`).
  - `CitationParser` and `CitationValidator` verifying references against provided `EvidenceSet`.
  - `GroundingVerifier` computing Citation Validity Rate, Sentence Citation Coverage, and Unsupported Citation Rate.
  - Multilingual insufficient evidence detection and localized fallback messages (`en`, `te`, `te-Latn`, `te-en`).
  - Curated RAG evaluation dataset `data/evaluation/rag_eval.json` (24 queries across 7 categories with independent gold chunk annotations).
  - Executable experiment runner `Exp005Runner` and script `scripts/run_exp005.py`.
  - **Phase 5.1**:
  - Expanded benchmark dataset `data/evaluation/rag_eval_expanded.json` (60 queries across 12 categories, 4 languages, 3 difficulties, independent gold annotations).
  - Automated benchmark leakage audit method verifying 0 verbatim question-to-chunk overlaps (PASS).
  - Claim-level propositional extraction interface (`Claim`, `ClaimExtractor`, `SentenceClaimExtractor`).
  - Entailment protocol (`EntailmentVerifier`) with heuristic lexical overlap assessor (`HeuristicLexicalOverlapAssessor`).
  - Propositional claim-level citation auditor (`ClaimLevelAuditor`) checking invalid citations, uncited claims, metadata contradictions, and unrelated evidence.
  - Citation attack unit tests (`tests/unit/test_rag_citation_attacks.py`) verifying safety boundary defense against `[S999]`, invalid page numbers, wrong documents, omitted citations, and unrelated evidence chunks.
  - Enhanced `OpenAICompatibleProvider` with environment variable resolution (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) and safe offline opt-in guard (`--provider real`).
  - Formal human evaluation rubric definition (0-2 scale across 5 dimensions, reported strictly as `NOT YET EVALUATED`).
  - Knowledge Graph contribution ablation (System A 0.8800 vs System B 0.9200 vs System C 0.8800) and Concept Mapping query formulation ablation (Raw 0.8800 vs Norm 0.9000 vs Canonical 0.9200).
  - Executable experiment runner `Exp005aRunner`, script `scripts/run_exp005a.py`, and versioned results `data/evaluation/exp005a_results.json` while strictly preserving historical baseline `data/evaluation/exp005_results.json`.

## What Is Currently Working
- `python -m pytest -q` → **173 passed** in ~2.8s.
- `python -m ruff check .` → **All checks passed!**
- `python scripts/run_exp002.py` → reproduces EXP-002 retrieval benchmark.
- `python scripts/run_exp003.py` → reproduces EXP-003 graph extraction benchmark.
- `python scripts/run_exp004.py` → reproduces EXP-004 Indic NLP & concept mapping benchmark.
- `python scripts/run_exp005.py` → reproduces historical EXP-005 RAG pipeline benchmark.
- `python scripts/run_exp005a.py` → reproduces EXP-005A hardened RAG evaluation, disaggregated metrics, and ablations.

## What Is Broken / Limitations
- `l3cube-pune/indic-sentence-bert-nli` checkpoint cannot be loaded under `transformers 5.8.1` / `torch < 2.6` due to CVE-2025-32434 check (legacy `.bin` without `safetensors`). Model marked `BLOCKED / NOT YET EVALUATED`.
- Telugu PDF extraction on real legacy-encoded course PDFs is unverified (waiting for user-provided real course PDFs).
- Human evaluation of RAG answers marked NOT YET EVALUATED until native Telugu engineering student panels evaluate real LLM outputs.
- Lexical overlap in ClaimLevelAuditor is a heuristic baseline and does not replace neural Natural Language Inference (NLI).

## What Was Being Implemented When Handoff Occurred
Phase 5.1 research hardening complete; ready for Phase 6 (Interactive Assessment & Student Knowledge Modeling).

## Exact Next Steps
1. Design `KnowledgeState` domain model ($Student \times Concept$) with score, confidence, evidence count, and last interaction timestamp.
2. Implement evidence-based knowledge estimator updating state from quiz results and Q/A interactions without mutating the canonical academic knowledge graph.
3. Implement diagnostic quiz generator grounded in academic knowledge graph concepts.
4. Implement dynamic personalized knowledge graph view (read-only overlay on canonical curriculum graph).
5. Implement adaptive prerequisite-aware remediation recommendation engine.

## Important Files
- `src/indiclearn/rag/pipeline.py`, `models.py`, `policies.py`
- `src/indiclearn/rag/claims.py`, `citation.py`, `grounding.py`
- `src/indiclearn/evaluation/exp005a_runner.py`, `scripts/run_exp005a.py`
- `data/evaluation/rag_eval_expanded.json`, `data/evaluation/exp005a_results.json`
- `docs/RAG_PIPELINE.md`, `docs/EXPERIMENTS.md`, `docs/DECISIONS.md`

- `src/indiclearn/llm/base.py`, `mock.py`, `providers/openai_compatible.py`
- `src/indiclearn/evaluation/exp005_runner.py`, `scripts/run_exp005.py`
- `data/evaluation/rag_eval.json`, `exp005_results.json`
- `docs/RAG_PIPELINE.md`, `docs/DECISIONS.md`, `docs/EXPERIMENTS.md`, `docs/PROJECT_CONTEXT.md`

## Architecture Summary
Decoupled Clean Architecture:
- Academic Knowledge Graph models the subject curriculum immutably.
- NLP pipeline translates raw, code-mixed, Romanized student queries into canonical `concept_id` without inferring student mastery or ignorance.
- RAG pipeline answers exclusively from retrieved evidence with full provenance chain (`answer` $\rightarrow$ `citation` $\rightarrow$ `evidence item` $\rightarrow$ `chunk` $\rightarrow$ `document` $\rightarrow$ `page/section`).
- Insufficient evidence triggers honest, transparent multilingual fallback responses rather than forced hallucinations.

## Database State
Knowledge graph persists to JSON format. Vector store indexes persist to `data/indices/`. Relational tables planned for Phase 6.

## Environment
Python 3.11.9, torch 2.5.1+cu121, transformers 5.8.1, sentence-transformers 6.1.0, faiss-cpu 1.15.1, networkx 3.6.1, scikit-learn 1.9.0, pydantic 2.13.2, pytest 9.1.0, ruff 0.16.10.
