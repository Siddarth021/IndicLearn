# RAG PIPELINE & GROUNDED ANSWER GENERATION

**Status**: Retrieval layer (Phase 2), Knowledge Graph (Phase 3), Indic NLP (Phase 4), and End-to-End Grounded Generation (Phase 5) are **IMPLEMENTED & VERIFIED**.

---

## 1. Executive Summary & Design Principles

IndicLearn's Phase 5 RAG pipeline establishes a **provenance-preserving, evaluation-ready, multilingual academic question-answering system** for engineering students.

The fundamental architectural principle governing this pipeline is:
> **The model must answer exclusively from retrieved academic course evidence, never from unsupported model knowledge or ungrounded parametric memory.**

### Core Tenets
1. **Provenance Chain Integrity**: Every answer is systematically traceable from the generated sentence to the cited source ID ($S_i$), the retrieved `EvidenceItem`, the underlying `DocumentChunk`, the source `Document`, and its page/section number.
2. **Concept-Aware Dual Retrieval**: Retrieval does not rely solely on raw or normalized query text. Canonical concepts mapped in Phase 4 are used to bridge multilingual student expressions (Telugu script, Romanized Telugu, code-mixed) to English-medium academic textbooks.
3. **Controlled Curriculum Graph Expansion**: The academic knowledge graph provides direct prerequisites and related concept definitions without ever polluting prompts with student mastery, weakness, or ignorance scores.
4. **Explicit Insufficient Evidence Handling**: When course material does not contain the answer, or when a student query falls outside the curriculum graph, the system explicitly returns an insufficient evidence fallback rather than forcing a hallucinated answer.
5. **Decoupled LLM Provider Interface**: The pipeline depends on an abstract `LLMProvider` protocol. Deterministic testing runs offline with `MockLLMProvider`, while real deployments can leverage `OpenAICompatibleProvider` without external SDK hard dependencies.
6. **Heuristic Grounding Verification**: Every generated answer undergoes an automated citation and claim audit that measures Citation Validity Rate, Citation Coverage, and Unsupported Citation Rate.

---

## 2. End-to-End Pipeline Architecture

```text
                          Student Query
                                ↓
                 Phase 4: Indic NLP Pipeline
             (Normalization, Tech Term Extraction)
                                ↓
                 Phase 4: Concept Mapping
              (Canonical Academic Concept IDs)
                                ↓
            Phase 3: Controlled Curriculum Context
           (Prerequisites & Related Graph Concepts)
                                ↓
              Phase 2: Hybrid Retrieval Engine
         (BM25 Lexical + Dense Semantic + RRF Fusion)
                                ↓
                     Evidence Selection
         (Deduplication, Score Scaling, Token Budget)
               [Sufficiency Gate: Insufficient?]
                     /                    \
                   YES                     NO
                   /                        \
    Return Multilingual Fallback      Deterministic Context Builder
           (No LLM Call)             ([S1], [S2] Provenance Headers)
                                                    ↓
                                          Decoupled LLM Provider
                                         (Mock or Real API Provider)
                                                    ↓
                                             Generated Answer
                                                    ↓
                                         Citation Parser & Validator
                                       (Extract [S_i], Match Evidence)
                                                    ↓
                                            Grounding Verifier
                                      (Validity, Coverage, Status)
                                                    ↓
                                             Final RAGResponse
```

---

## 3. Strongly Typed Domain Models (`indiclearn.rag.models`)

All RAG components communicate using strictly typed Pydantic models:

- **`EvidenceItem`**:
  - `source_id: str`: Prompt identifier (e.g. `'S1'`, `'S2'`).
  - `document_id: str`: Unique document identifier.
  - `chunk_id: str`: Deterministic chunk hash.
  - `text: str`: Academic text snippet.
  - `page_number: int | None`, `section: str | None`.
  - `score: float`: Retrieval fusion score.
  - `retrieval_method: str`: `'hybrid'`, `'bm25'`, or `'dense'`.
  - `concept_ids: list[str]`: Associated curriculum concept IDs.
  - `rank: int`: Assigned ordinal rank.

- **`EvidenceSet`**:
  - `items: list[EvidenceItem]`
  - `query: str`, `subject_id: str`
  - `total_candidates: int`
  - `insufficient_evidence: bool`
  - `insufficiency_reason: str | None`
  - Helper methods: `get_by_source_id(sid)`, `get_by_chunk_id(cid)`.

- **`Citation` & `CitationSpan`**:
  - `source_id: str`, `document_id: str`, `chunk_id: str`, `page_number: int | None`, `section: str | None`.
  - `is_valid: bool`: True if mapped to an item in `EvidenceSet`.
  - `validation_error: str | None`: Error description if fabricated.
  - `CitationSpan`: Character `start`/`end` offsets in answer text.

- **`GroundingCheck` & `GroundingReport`**:
  - `is_grounded: bool`
  - `citation_validity_rate: float`: $\frac{\text{valid citations}}{\text{total citations}} \in [0.0, 1.0]$.
  - `citation_coverage: float`: $\frac{\text{cited substantive sentences}}{\text{total substantive sentences}} \in [0.0, 1.0]$.
  - `unsupported_citation_rate: float`: $\frac{\text{invalid citations}}{\text{total citations}} \in [0.0, 1.0]$.
  - `status: str`: `'GROUNDED'`, `'PARTIALLY_GROUNDED'`, `'UNSUPPORTED'`, or `'INSUFFICIENT_EVIDENCE'`.

- **`RAGRequest` & `RAGResponse`**:
  - `RAGRequest`: `query`, `subject_id`, `requested_language`, `top_k`, `max_graph_hops`, `max_graph_concepts`, `min_score_threshold`.
  - `RAGResponse`: `query`, `subject_id`, `answer`, `is_answerable`, `status`, `citations`, `evidence`, `concept_ids`, `grounding_report`, `metadata`.

---

## 4. Evidence Selection & Deduplication (`EvidenceSelector`)

The evidence selection stage bridges raw retrieval candidates and prompt assembly:

1. **Deduplication**: When multi-query expansion (e.g. normalized query + canonical concept name) returns overlapping chunks, `EvidenceSelector` collapses them by `chunk_id`, preserving the highest score.
2. **Score Scale Adaptation**: Retrieval methods have radically different score distributions:
   - BM25: Unbounded positive floats ($1.0 - 20.0+$).
   - Dense Cosine: Range $[0.0, 1.0]$.
   - Reciprocal Rank Fusion (RRF): $\sum \frac{w}{60 + r} \approx [0.005, 0.020]$.
   If candidate scores indicate RRF scale ($\max < 0.05$), the selector scales the effective threshold proportionally ($0.15 \times 0.05 = 0.0075$), preventing valid RRF candidates from being discarded.
3. **Character Budgeting**: Context size is capped by `max_total_chars` (default 4000 chars) to prevent context stuffing and token overflow.
4. **Deterministic Labeling**: Selected candidates are numbered sequentially as `[S1]`, `[S2]`, `[S3]`, ...

---

## 5. Controlled Graph Context Expansion

The academic Knowledge Graph (`NetworkXGraphStore`) enriches retrieval when concepts are mapped:
- **Direct Prerequisites**: Extracted via `graph_store.get_prerequisites(concept_id)`.
- **Related Curriculum Concepts**: Parent/child concepts within the same subject.
- **Controlled Scope**:
  - `max_graph_hops = 1` (direct prerequisites only).
  - `max_graph_concepts = 3`.
- **Strict Decoupling**: Absolutely no learner mastery, weakness, or ignorance scores are exposed to this layer.

---

## 6. Prompt Construction & Deterministic Formatting (`ContextBuilder`)

Prompts are constructed deterministically to eliminate ambiguity:

```text
=== ACADEMIC CURRICULUM CONTEXT ===
Subject: signals_and_systems
- Concept: Convolution (concept) [Aliases: కన్వల్యూషన్, convolution integral] | Prerequisites: LTI System
===================================

=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: signals_notes.pdf | Page 1 | Section: LTI Systems and Convolution | Chunk: chunk_001
Related Concepts: convolution, lti_systems
Convolution is a mathematical operation combining input signal x(t) and impulse response h(t)...
===================================

Student Question: Explain convolution and its relationship with LTI systems.
```

### System Instruction Policy (`RAG_SYSTEM_PROMPT`)
- Answer ONLY using supplied academic evidence.
- Every factual claim must reference the exact source identifier in square brackets (`[S1]` or `[S1, S2]`).
- Never invent facts absent from the context.
- Never invent unprovided source identifiers (e.g. `[S999]`).
- If evidence is insufficient, state that the available material is insufficient.

---

## 7. Decoupled LLM Provider Interface (`indiclearn.llm`)

- **`LLMProvider` Protocol**:
  - `generate(prompt: str, **kwargs) -> LLMResponse`
  - `generate_messages(messages: list[LLMMessage], **kwargs) -> LLMResponse`
- **`MockLLMProvider`**:
  - Supports deterministic testing modes: `'grounded'`, `'insufficient_evidence'`, `'unsupported'`, `'incorrect_citation'`, `'missing_citation'`.
  - Works 100% offline without external dependencies or API keys.
- **`OpenAICompatibleProvider`**:
  - Built using standard library `urllib.request` to avoid third-party vendor lock-in.
  - Reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from environment without hardcoding.

---

## 8. Citation Verification & Grounding Audit (`GroundingVerifier`)

Post-generation, the answer text is parsed and audited:
1. **Citation Parsing**: Regex extraction of all `[S_i]` and multi-citations `[S1, S2]`.
2. **Attribution Verification**: Each extracted ID is checked against the prompt's `EvidenceSet`.
3. **Sentence-Level Coverage**: The answer is segmented into sentences, and each substantive claim is checked for valid citations.
4. **Status Assignment**:
   - `GROUNDED`: 100% of citations valid, coverage $\ge 40\%$, no unsupported citations.
   - `PARTIALLY_GROUNDED`: All citations valid, but coverage $< 40\%$.
   - `UNSUPPORTED`: Any citation is invalid or references an unsupplied source.
   - `INSUFFICIENT_EVIDENCE`: Correctly returned fallback response when retrieval was insufficient.

---

## 9. Multilingual Fallback Policy

When retrieval yields zero valid chunks or the query falls outside curriculum scope, the pipeline responds in the student's language:

| Language | Code | Standard Fallback Message |
| :--- | :--- | :--- |
| **English** | `en` | *"I could not find sufficient information in the uploaded academic materials to answer this question reliably."* |
| **Telugu Script** | `te` | *"అందించిన విద్యా విషయాలలో ఈ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి తగినంత సమాచారం దొరకలేదు."* |
| **Romanized Telugu** | `te-Latn` | *"Upload chesina academic material lo e prasnaku saripoyina information ledu."* |
| **Code-Mixed** | `te-en` | *"Uploaded academic material lo deeniki sambhandhinchina sufficient information ledu."* |

---

## 10. Phase 5.1 Research Hardening & Scientific Interpretation Rule

Phase 5.1 hardens the evaluation methodology by enforcing a strict scientific interpretation boundary:

```text
Pipeline validation
        ≠
Real LLM answer quality
        ≠
Human-judged correctness
```

1. **MockLLMProvider Validation**:
   - Validates citation parsing regex, citation ID cross-referencing, grounding verification thresholds, insufficient-evidence routing, and adversarial attack detection.
   - **Does NOT** establish factual truth, Telugu grammatical naturalness, real-world semantic faithfulness, or parametric hallucination rates of a frontier LLM.
2. **Real LLM Evaluation**:
   - Evaluates whether real OpenAI-compatible models (e.g. `gpt-4o-mini`, local Ollama, vLLM) follow citation syntax and generate grounded summaries from retrieved context.
   - Supported via `OpenAICompatibleProvider` (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`).
   - Completely opt-in (`--provider real`). Unconfigured runs safely report `REAL LLM EVALUATION: NOT RUN`.
3. **Human-Judged Correctness**:
   - Formal evaluation on 5 core dimensions: Correctness (0-2), Groundedness (0-2), Relevance (0-2), Completeness (0-2), and Language Appropriateness (0-2).
   - If no human rating panel has scored completions, the system strictly reports `HUMAN EVALUATION: NOT YET EVALUATED`. No synthetic human ratings are fabricated.

---

## 11. Claim-Level Propositional Analysis (`indiclearn.rag.claims`)

To progress beyond raw surface-level string matching, Phase 5.1 introduces a modular claim-level architecture:

```text
Generated Answer
       ↓
Claim Extraction (SentenceClaimExtractor)
       ↓
Claim → Citation Resolution
       ↓
Evidence Retrieval & Verification
       ↓
Support Assessment (EntailmentVerifier Protocol)
```

- **`Claim`**: Strongly typed data structure representing atomic propositions with cited source IDs and support statuses (`SUPPORTED`, `UNSUPPORTED_CITATION`, `UNRELATED_EVIDENCE`, `UNCITED`).
- **`EntailmentVerifier` Protocol**: Defines the abstract contract for semantic entailment verification.
- **`HeuristicLexicalOverlapAssessor`**: Baseline term-overlap assessor checking content words (excluding stopwords).
  > [!NOTE]
  > Heuristic lexical overlap is **not** a neural Natural Language Inference (NLI) model. It serves as an auditable baseline for topical relevance and prevents citing evidence for completely unrelated claims.
- **`ClaimLevelAuditor`**: Audits answers for invalid citation tags, uncited substantive statements, metadata contradictions (e.g. claiming wrong page or wrong document), and unrelated evidence chunks.

---

## 12. Citation Attack Resilience & Safety Boundaries

The safety boundary is validated through unit test attacks (`tests/unit/test_rag_citation_attacks.py`) simulating adversarial generation:
- **`[S999]` Fabricated Identifier**: Detected and rejected (`unsupported_citation_count > 0`, status `UNSUPPORTED`).
- **Multiple Invalid Citations (`[S998, S999]`)**: Fully captured with 0.0 validity rate.
- **Citation Omitted**: Substantive assertions without citations yield 0.0 coverage.
- **Citation Attached to Unsupported Sentence**: Yields partial coverage below threshold.
- **Citation to Unrelated Evidence**: Flagged by `ClaimLevelAuditor` as `UNRELATED_EVIDENCE`.
- **Invalid Page Number & Wrong Document**: Detected via metadata audit comparing claimed page/doc against `EvidenceItem`.

---

---

## 14. Curriculum Scope Control, Ambiguity Gating & Retrieval Hardening (Phase 5.2)

### The Core Problem Revealed by EXP-005A
EXP-005A demonstrated two critical architectural vulnerabilities:
1. **Adversarial False Acceptance**: 5/10 out-of-scope and adversarial queries were mistakenly answered because broad engineering keywords (e.g. `"system"`, `"signal"`, `"filter"`, `"Fourier"`, `"algorithm"`) triggered high BM25 scores or partial concept overlaps with unrelated chunks.
2. **Graph Candidate Displacement**: Expanding the retrieval candidate pool with knowledge graph prerequisite neighbors reduced Recall@5 from **0.9200** to **0.8800**, as prerequisite definitions competed with and displaced direct local evidence chunks.

### Core Principle: A Lexical Match is NOT Curriculum Grounding
Academic scope membership ($q \in \text{Signals \& Systems}$) cannot be established merely by the presence of shared technical words. IndicLearn enforces a strict distinction between:
- `IN_SCOPE`: Strong curriculum grounding with verified evidence chunks.
- `OUT_OF_SCOPE`: Query references external academic domains (e.g. digital filter design, state-space control, image compression, FFT decimation algorithms) or unrelated engineering topics.
- `AMBIGUOUS`: Under-specified queries containing only broad generic engineering terms without disciplinary qualifiers (e.g. "Explain filters.", "What is a system?", "filters ni explain cheyyi").
- `INSUFFICIENT_EVIDENCE`: Query belongs to the subject, but uploaded materials lack sufficient supporting chunks.

### Multi-Signal Scope Evidence Hierarchy
`HeuristicScopeAssessor` assesses curriculum scope using a deterministic 6-signal hierarchy:
- **Signal A — Canonical Concept Mapping**: Checks if the query maps confidently to a curriculum concept via exact canonical or validated alias match.
- **Signal B — Technical Terminology & Generic Penalty**: Distinguishes subject-specific terminology from broad generic words (`system`, `signal`, `filter`, `transform`, `algorithm`, `response`, `frequency`, `function`). Generic terms contribute minimally to scope confidence.
- **Signal C — Concept Mapping Margin**: Evaluates ambiguity using `score_margin = top_score - second_score`. A margin $< 0.08$ on moderate scores indicates competing candidate ambiguity.
- **Signal D — Evidence Retrieval**: Requires direct retrieval candidates to exceed a minimum evidence threshold (`score >= 0.007`).
- **Signal E — Concept-Evidence Consistency**: Verifies that retrieved top-5 chunks actually mention the mapped canonical concept or contain its concept ID. If concept mapping asserts concept $X$ but retrieval contains zero mentions of $X$, the query is not safely grounded.
- **Signal F — Domain Guards & Semantic Alignment**: Employs explicit out-of-scope domain guards (`digital_filter_design`, `control_systems`, `image_and_multidimensional`, `dsp_fft_algorithms`). If an adversarial prompt injects curriculum terms (e.g. "Use convolution to design a JPEG compression algorithm"), the domain guard halts generation and returns `OUT_OF_SCOPE`.

### Decoupled Post-Retrieval Graph Enrichment Architecture
To resolve the candidate displacement regression, Phase 5.2 restructures graph context integration:
```text
Query → Concept Mapping → Hybrid Retrieval → Evidence Selection → Graph Context → LLM Context
```
The academic knowledge graph strictly enriches the **generation prompt context** (providing prerequisite definitions and conceptual relationships for pedagogical explanations) without competing in or displacing direct chunks from the **retrieval candidate pool**.

---

## 15. Verification & Quality Gates

The entire RAG system is verified by:
- 183 automated unit and integration tests (`python -m pytest -q`).
- Zero Ruff lint errors (`python -m ruff check .`).
- Historical EXP-005 benchmark (`scripts/run_exp005.py`).
- Research-hardened EXP-005A benchmark (`scripts/run_exp005a.py`).
- RAG Safety & Scope Hardening EXP-005B benchmark (`scripts/run_exp005b.py`).

