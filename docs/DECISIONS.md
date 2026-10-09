# DECISIONS

## DEC-001 Domain model library — 2026-10-04
Problem: typed, validated entities. Options: dataclasses, Pydantic v2, attrs. Chosen: Pydantic v2 (also needed for FastAPI later). Trade-off: extra dependency. Replacement: models are plain data; swap is mechanical.

## DEC-002 Chunking baseline — 2026-10-04
Problem: how to chunk. Options: fixed, recursive char, token-aware, semantic. Chosen: recursive char (size 800/overlap 120, PROPOSED heuristics) with exact offsets. Reason: simplest reproducible baseline; Telugu has no whitespace-tokenization issue for splitting at sentence/space. Trade-off: char≠token. Future: token-aware chunker behind `Chunker` Protocol; tune via EXP-003.

## DEC-003 Vector store — DECIDED (2026-10-06, Phase 2)
Problem: local vector indexing with persistence, metadata filtering, and swappability.
Options: FAISS, Chroma, Qdrant, InMemory (NumPy).
Chosen: FAISS (`FaissVectorStore`) as primary local store, with `InMemoryVectorStore` (NumPy) as pure-Python fallback, both implementing `VectorStore` Protocol.
Design:
- `IndexFlatIP` on unit-normalized vectors provides exact cosine similarity.
- Sidecar metadata dictionary stores document, page, character offsets, language, and concept IDs.
- Metadata filtering (`subject_id`, `document_id`, `language`, `concept_id`) performed on retrieved candidate pool.
- Persistence: writes `.faiss` binary and `.meta.json` sidecar.
- Versioning: validates `embedding_model` and `embedding_dimension` on index load, raising `IndexVersionMismatchError` if incompatible.

## DEC-004 Embedding model — DECIDED (2026-10-06, Phase 2 via EXP-002)
Problem: embedding model capable of cross-lingual semantic alignment between Telugu/Romanized/code-mixed student queries and English academic text.
Evaluated in EXP-002:
1. `intfloat/multilingual-e5-small` (384-dim, 470 MB): **Selected Primary Model**. Achieved 1.0000 Recall@3 across all four linguistic formulations (EN, TE, Romanized TE, Code-mixed) and 0.9750 MRR. Uses asymmetric prefixes (`passage: ` / `query: `).
2. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 470 MB): **Supported Alternative**. Symmetric embeddings, 0.9250 Recall@3, 0.9375 MRR.
3. `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 90 MB): English baseline. Degraded to 0.5000 Recall@3 on Telugu script queries.
4. `l3cube-pune/indic-sentence-bert-nli` (768-dim, 440 MB): **BLOCKED / NOT YET EVALUATED**. Legacy `.bin` checkpoint lacks `safetensors`; blocked by `transformers 5.8.1` under `torch < 2.6` for CVE-2025-32434.

## DEC-005 Preserve ZWJ/ZWNJ in cleaning — 2026-10-04
Problem: invisible chars. Chosen: strip only NUL/BOM/soft-hyphen; keep U+200C/200D (meaningful in Indic conjuncts). NFC default (configurable).

## DEC-006 Academic graph vs student model separation — 2026-10-04 (mandated)
Student layer holds only concept_id references; canonical graph immutable w.r.t. student data.

## DEC-007 Added `domain/` package — 2026-10-04
Shared entities needed by many packages; avoids circular imports.

## DEC-008 PDF/PPTX extraction libraries — 2026-10-04
pypdf (pure Python, baseline) and python-pptx. Trade-off: weaker layout/Telugu-font handling than e.g. PyMuPDF/pdfplumber. Future: compare extractors on real Telugu PDFs; OCR for scans.

## DEC-009 Score Normalization & Hybrid Fusion Strategy — 2026-10-06 (Phase 2)
Problem: raw BM25 scores (unbounded positive) and dense cosine similarity scores ([-1, 1]) cannot be naively summed.
Chosen:
- Primary default: **Reciprocal Rank Fusion (RRF)** ($k=60$). RRF is rank-based, parameter-free, and invariant to disparate score distributions and outlier spikes.
- Secondary supported option: **Min-Max score normalization** followed by weighted linear summation ($\alpha \cdot s'_{lex} + (1-\alpha) \cdot s'_{dense}$).
Verified in EXP-002: RRF achieved MRR 0.9625 and robust tie-breaking.

## DEC-010 Indic Word Tokenization for Lexical Retrieval — 2026-10-06 (Phase 2)
Problem: Python standard `re.findall(r"\w+")` splits Telugu words at viramas (`\u0C4D`) and vowel signs because Unicode categories `Mn` (nonspacing mark) and `Mc` (spacing combining mark) are not included in `\w`.
Chosen: Custom tokenizer in `indiclearn.retrieval.bm25` using `unicodedata.category` that groups letters (`L*`), marks (`M*`), numbers (`N*`), connectors (`Pc`), and conjunct joiners (`ZWJ`/`ZWNJ`). Preserves compound Telugu script tokens intact.

## DEC-011 Graph Store Backend: NetworkX MultiDiGraph — 2026-10-06 (Phase 3)
Problem: selecting knowledge graph storage engine for Phase 3 prototype.
Options considered:
1. Neo4j (property graph database, external daemon).
2. RDF / SPARQL (rdflib, semantic triple store).
3. SQLite ad-hoc relational tables.
4. NetworkX in-memory `MultiDiGraph`.
Chosen: **NetworkX `MultiDiGraph`** wrapped behind the `GraphStore` Protocol (`NetworkXGraphStore`).
Rationale: NetworkX provides built-in multi-edge directed graph algorithms (cycle detection, BFS/DFS traversal, pathfinding) with zero infrastructure overhead. Wrapping all operations in `GraphStore` guarantees that higher-level services (`ConceptService`, RAG, evaluation runners) do not depend on NetworkX API directly, facilitating future migration to Neo4j or RDF without refactoring.

## DEC-012 Deterministic Concept Identity Strategy — 2026-10-06 (Phase 3)
Problem: preventing concept explosion and non-deterministic IDs (e.g. `CONVOLUTION_001`, `CONVOLUTION_002`) across chunks.
Chosen: Deterministic formula `cpt_{subject_id}_{slugify(canonical_name)}`.
Rationale: Uniquely and deterministically binds concept identity to subject domain and canonicalized name. Merges multiple occurrences across documents and chunks, aggregating source provenance while preserving distinct academic concepts (e.g. `Fourier Transform` and `Fourier Series` remain distinct; `LTI` and `Linear Time-Invariant System` merge to `cpt_signals_and_systems_lti_system`).

## DEC-013 Controlled Relation Ontology & Direction Semantics — 2026-10-06 (Phase 3)
Problem: unbounded or ambiguously directed graph edges.
Chosen: Controlled vocabulary of 8 typed relations with strict directionality:
- `PREREQUISITE_OF`: $A \rightarrow B$ ($A$ is required to understand $B$). Incoming predecessors are prerequisites.
- `HAS_PROPERTY`: $Host \rightarrow Property$ (e.g. LTI System $\rightarrow$ Linearity).
- `USED_IN`: $Tool \rightarrow Context$ (e.g. Convolution $\rightarrow$ LTI System).
- `PART_OF`: $Component \rightarrow Whole$ (e.g. Nyquist Rate $\rightarrow$ Sampling Theorem).
- `SUBCONCEPT_OF`: $Specific \rightarrow General$.
- `DERIVED_FROM`: $Target \rightarrow Source$.
- `CONTRASTS_WITH`: Bidirectional distinction.
- `RELATED_TO`: General conceptual association.
Prerequisite nature is explicitly flagged as `EXPLICIT_PREREQUISITE` vs `INFERRED_PREREQUISITE`.

## DEC-014 Dual-Level Source Provenance Strategy — 2026-10-06 (Phase 3)
Problem: explaining why concepts and relations exist in the graph without hallucination.
Chosen: Mandatory provenance arrays on both `Concept` and `ConceptRelation`:
- `Concept`: `source_document_ids`, `source_chunk_ids`, `page_numbers`, `sections`.
- `ConceptRelation`: `source_chunk_ids`, `source_document_ids`, `extraction_method`, `confidence`.
Graph validation checks provenance presence, warning or erroring on orphaned knowledge items.

## DEC-015 Baseline Heuristic Concept Extractor First — 2026-10-06 (Phase 3)
Problem: whether to rely immediately on LLMs for knowledge graph extraction.
Chosen: Implement a deterministic, reproducible heuristic extractor (`HeuristicConceptExtractor`) first.
Rationale: Establishing an empirical baseline on Markdown heading hierarchies, definitional grammar patterns, property term patterns, and domain term rules provides a rigorous benchmark for EXP-003. LLM-assisted extraction can be benchmarked against this baseline in future phases without architectural changes.

## DEC-016 Deterministic JSON Graph Persistence Format — 2026-10-06 (Phase 3)
Problem: serialization format for saving and loading academic knowledge graphs.
Options: GraphML, GEXF, JSON, Pickle.
Chosen: Human-readable JSON with sorted node and edge lists.
Rationale: GraphML and GEXF introduce XML bloat and lose Pydantic typed fields without complex attribute mappings. Pickle is uninspectable and unsafe. Sorted JSON preserves exact typing, provenance lists, and deterministic diffs in version control.

## DEC-017 Heuristic Multi-Signal Language & Script Detector — 2026-10-06 (Phase 4)
Problem: how to reliably classify short, conversational engineering queries in English, Telugu script, Romanized Telugu, and code-mixed formats without heavyweight external dependencies.
Chosen: Implement a Unicode-aware heuristic detector analyzing script character distributions, Romanized Telugu functional markers (`ante`, `enti`, `cheyyi`, `lo`, `ki`), and academic engineering terminology.
Rationale: Short conversational queries like `"convolution ante enti?"` contain English technical words (`convolution`) that mislead generic language identifiers into predicting pure English. Balancing functional particle frequencies with technical term detection correctly identifies Romanized Telugu and code-mixed patterns.

## DEC-018 Non-Destructive Ambiguity Preservation in Normalization — 2026-10-06 (Phase 4)
Problem: Romanized Telugu spelling varies widely (`enti`, `yenti`, `emiti`, `yemiti`), with potential ambiguities across words.
Chosen: The student's raw query is preserved immutably in `original_text`. The normalizer outputs a canonical normalized string alongside a ranked list of candidate variants (`NormalizationCandidate`) with explicit confidence scores and optional back-transliteration into Telugu script.
Rationale: Prevents unrecoverable loss of student intent. Downstream retrieval and concept mapping can inspect alternative candidates if the canonical hypothesis is low-confidence.

## DEC-019 Graph-Grounded Academic Terminology Detection — 2026-10-06 (Phase 4)
Problem: how to extract academic technical terms across languages without maintaining an independent, drift-prone vocabulary.
Chosen: Ground `GraphGroundedTermDetector` directly in the Academic Knowledge Graph (`GraphStore` / `Concept`).
Rationale: Canonical concept names, textbook aliases, acronyms (`LTI`, `ROC`, `CTFT`), and multilingual aliases are already maintained in the curriculum graph. Grounding terminology extraction directly in the graph ensures complete synchronization and zero maintenance divergence.

## DEC-020 Multi-Signal Hybrid Concept Mapping with Strict Learner Decoupling — 2026-10-06 (Phase 4)
Problem: mapping student queries across languages to canonical concept IDs while maintaining scientific modularity and avoiding premature learner state assumptions.
Chosen: Implement four evaluated baselines (A: Exact, B: Lexical, C: Embedding, D: Hybrid fusing terminology, lexical, and semantic evidence). Strictly forbid inferring student mastery, ignorance, or weakness from query topics.
Rationale: Evaluates the individual contribution of each representation layer. Mandating that `"convolution ante enti?"` only indicates query intent about convolution—not student weakness—protects learner modeling integrity until Phase 7.

## DEC-021 Decoupled LLM Provider Interface with Offline Deterministic Mock — 2026-10-06 (Phase 5)
Problem: how to build and evaluate the RAG generation pipeline without forcing hard dependencies on external proprietary APIs, incurring API costs in CI/CD, or blocking offline development.
Chosen: Abstract language model generation behind a minimal `LLMProvider` Protocol (`generate`, `generate_messages`). Implement `MockLLMProvider` simulating deterministic test modes (`grounded`, `insufficient_evidence`, `unsupported`, `incorrect_citation`, `missing_citation`), and an optional `OpenAICompatibleProvider` using standard library `urllib` for external endpoint deployment.
Rationale: Enables complete, fast, reproducible offline testing of citation parsing, evidence selection, and grounding auditing without external network access or vendor SDK lock-in.

## DEC-022 Sequential Citation Identifiers ([S1], [S2]) with Strict Provenance — 2026-10-06 (Phase 5)
Problem: how to instruct LLMs to cite retrieved course material without hallucinating arbitrary page numbers or invented textbook titles.
Chosen: Number retrieved and selected evidence chunks sequentially as `[S1]`, `[S2]`, `[S3]`, etc. In the context prompt, each chunk header specifies `[SOURCE i] (ID: Si)` along with document ID, page number, section, chunk ID, and related concepts. Instruct the model to cite claims strictly using `[Si]`.
Rationale: Constrains the citation action space for the LLM to known tokens. Allows regex post-processing to reliably extract citations and verify that every referenced source exists in the supplied `EvidenceSet`.

## DEC-023 Heuristic Grounding and Citation Verification Engine — 2026-10-06 (Phase 5)
Problem: auditing generated answers for hallucination and unsupported claims without claiming an infallible "truth detector".
Chosen: Implement `GroundingVerifier` calculating three objective, mathematically defined metrics: Citation Validity Rate (fraction of citations mapped to supplied evidence), Citation Coverage (fraction of substantive sentences with valid citations), and Unsupported Citation Rate (fraction of invalid or unsupplied citations). Assign status `GROUNDED`, `PARTIALLY_GROUNDED`, `UNSUPPORTED`, or `INSUFFICIENT_EVIDENCE`.
Rationale: Grounding is treated as empirical provenance verification rather than speculative semantic entailment. Clearly labeling the checker as heuristic avoids overclaiming factual correctness while catching fabricated source IDs and ungrounded statements.

## DEC-024 Mandatory Multilingual Insufficient Evidence Fallback Policy — 2026-10-06 (Phase 5)
Problem: how to prevent the model from hallucinating plausible-sounding answers when the student asks about topics absent from uploaded course material or outside the syllabus curriculum graph.
Chosen: If candidate retrieval yields zero chunks above threshold, or if the student's question fails curriculum graph concept grounding and lacks strong lexical evidence, the pipeline halts generation and returns a standardized insufficient evidence message in the student's requested language (`en`, `te`, `te-Latn`, `te-en`).
Rationale: Retrieval failure must never be masked by ungrounded parametric model generation. A student asking about Kalman filtering or FIR Kaiser filters in an introductory Signals & Systems course receives an honest, transparent statement of material absence rather than an ungrounded hallucination.

## DEC-025 Scientific Interpretation Boundary & Three-Tier Evidence Model — 2026-10-06 (Phase 5.1)
Problem: confusing software pipeline verification with real-world LLM factual generation quality or human correctness.
Chosen: Formally establish a three-tier evidence hierarchy in all documentation and evaluation outputs:
1. Pipeline validation (MockLLMProvider testing citation parsing, provenance tracking, and error handling mechanics).
2. Real LLM generation quality (OpenAI-compatible models evaluated on citation following and context synthesis).
3. Human-judged correctness (formal 5-dimension rubric on a 0-2 scale, reported strictly as NOT YET EVALUATED until native annotator panels score outputs).
Rationale: Prevents misleading research claims (e.g. claiming "hallucination eliminated" when only citation syntax was verified). Enforces scientific rigor across all publications and benchmarks.

## DEC-026 Claim-Level Propositional Auditing with Entailment Protocol — 2026-10-06 (Phase 5.1)
Problem: coarse document-level or chunk-level citation checking misses granular factual drift, citation to unrelated evidence, or metadata contradictions.
Chosen: Implement `ClaimExtractor` protocol with `SentenceClaimExtractor` baseline, `EntailmentVerifier` protocol with `HeuristicLexicalOverlapAssessor`, and `ClaimLevelAuditor`. The system breaks answer text into propositional statements, audits each against cited sources, and verifies that cited evidence shares substantive non-stopword terminology with the claim.
Rationale: Provides a modular pathway from heuristic lexical overlap to future neural cross-encoder NLI models without breaking pipeline interfaces. Catches adversarial attacks like citing DSP convolution notes to support cake recipes or claiming non-existent page numbers.

## DEC-027 Opt-In Real LLM Evaluation with Offline Test Suite Isolation — 2026-10-06 (Phase 5.1)
Problem: enabling evaluation of real frontier language models without requiring API keys, incurring external costs, or violating offline test suite isolation.
Chosen: Real LLM evaluation is strictly opt-in (`--provider real`). All unit and integration tests run 100% offline using `MockLLMProvider`. If external credentials (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`) are missing, the runner cleanly outputs `REAL LLM EVALUATION: NOT RUN` without failing CI or generating fabricated metrics.
Rationale: Balances practical deployment verification with deterministic, reproducible, air-gapped CI test execution.




## DEC-028 Multi-Signal Scope Assessment & Generic-Term Penalties — 2026-10-06 (Phase 5.2)
Problem: Broad engineering terms (e.g. "filter", "signal", "system", "transform") cause false acceptance of out-of-scope technical queries (e.g. Kalman filters, JPEG DCT, Cooley-Tukey FFT) when matched to curriculum concepts.
Chosen: Implement `ScopeAssessor` protocol with `HeuristicScopeAssessor` evaluating a 6-signal hierarchy: (1) Canonical concept mapping, (2) Subject-specific vs generic terminology down-weighting (`GENERIC_TERMS`), (3) Concept score margin (`score_margin < 0.08` flags ambiguity), (4) Evidence retrieval score thresholding, (5) Concept-evidence consistency checking in retrieved chunks, and (6) Adversarial domain indicator guards (`OUT_OF_SCOPE_DOMAINS`). Introduce explicit `AMBIGUOUS` state for under-specified queries ("Explain filters.") and safe localized refusals.
Rationale: A keyword match is not curriculum grounding. In engineering education, refusing an out-of-scope query or asking for clarification is safer than hallucinating an ungrounded or out-of-syllabus answer.

## DEC-029 Decoupled Post-Retrieval Curriculum Graph Context Enrichment — 2026-10-06 (Phase 5.2)
Problem: Expanding the retrieval candidate pool with knowledge graph prerequisite neighbors displaced direct evidence chunks in EXP-005A, reducing Recall@5 from 0.9200 to 0.8800.
Chosen: Decouple direct evidence retrieval from curriculum explanation context. Direct evidence retrieval runs strictly via Hybrid BM25 + dense search on the student query and canonical concept. After evidence selection determines the top chunks, knowledge graph prerequisites are retrieved separately and attached as supplementary pedagogical context in the prompt builder, without competing for slots in the evidence candidate pool.
Rationale: Direct evidence and curriculum prerequisites serve fundamentally different purposes: direct evidence answers the specific question, while prerequisites enrich the conceptual explanation. Separating them preserves peak retrieval recall (0.9200) while maintaining pedagogical graph context.

## DEC-030 Separation of Student Knowledge State from Academic Graph — 2026-10-06 (Phase 6)
Problem: whether student mastery, weakness, or ignorance should be stored directly on nodes/edges of the canonical Academic Knowledge Graph.
Chosen: Maintain strict architectural separation. The Academic Knowledge Graph (`Concept`, `ConceptRelation`) is canonical, objective, and completely invariant to student actions. The Student Knowledge Model (`KnowledgeState`, `EvidenceEvent`) stores private, dynamic mastery estimates for each $Student \times Concept$ pair. The academic graph is never mutated with student IDs, mastery scores, or ignorance flags.
Rationale: Subject matter knowledge is shared across all students; learner mastery is private, non-stationary, and uncertain. Blurring these layers causes graph pollution, concurrency bottlenecks, and prevents evaluating different knowledge-tracing estimators over the same canonical curriculum.

## DEC-031 Treating Questions as Weak Evidence Rather than Proof of Ignorance — 2026-10-06 (Phase 6)
Problem: how student interaction queries (e.g. asking "What is convolution?" in the RAG pipeline) should influence estimated concept mastery.
Chosen: Categorize student interactions into a strict evidence hierarchy: Strong (quiz/assessment results, weight = 1.00), Medium (evaluated tasks, weight = 0.40), and Weak (questions asked, views, explanations, weight = 0.10). An inquisitive question contributes minimal pull toward score adjustment and adds bounded confidence (\Delta C = 0.02).
Rationale: Asking a question indicates curiosity, active engagement, or exam revision. In engineering education, penalizing questions as proof of ignorance creates perverse incentives, discouraging students from asking questions. Observational interaction evidence must never dominate validated assessment evidence.

## DEC-032 Interpretable Evidence-Based Estimator with Explicit Uncertainty — 2026-10-06 (Phase 6)
Problem: selecting a knowledge-tracing estimator that is scientifically defensible, reproducible, and explainable before extensive real-student calibration data is available.
Chosen: Implement `KnowledgeEstimator` protocol with `EvidenceBasedKnowledgeEstimator`. Separates estimated knowledge score ($S \in [0.0, 1.0]$) from confidence ($C \in [0.0, 1.0]$). Utilizes recency decay (exponential decay with 30-day half-life), weighted accumulation over preserved evidence history, and conflicting evidence detection (variance in recent strong assessments) that maps contradictory signals to `MasteryStatus.UNCERTAIN` rather than oscillating between 0 and 1.
Rationale: Deep neural knowledge tracing models are black boxes that require thousands of real student interaction logs to train without overfitting. A transparent, evidence-based estimator provides clear explainability ("Why is this student considered developing?") and full provenance while establishing a clean protocol for future BKT or neural drop-in replacements.

## DEC-033 Strict Evaluative-Evidence Gating for Mastery Status Categorization — 2026-10-06 (Phase 6.1 Audit)
Problem: SCEN_002 produced `DEVELOPING` for a student with monotonic failure ($S = 0.0750, C = 0.7438$) due to an implementation fallthrough, and observational interactions (questions asked) could prematurely graduate students from `UNKNOWN` to `EMERGING`.
Chosen: Formulate an authoritative, mutually exclusive priority partition for `MasteryStatus` in `EvidenceBasedKnowledgeEstimator`:
1. Rule 1 (`UNKNOWN`): Insufficient evidence ($C < 0.25$) OR lack of any evaluative assessment (`strong_count == 0 and medium_count == 0`). Non-evaluative interactions (asking questions, viewing concepts) strictly leave status as `UNKNOWN`.
2. Rule 2 (`UNCERTAIN`): High assessment variance ($\ge 0.15$ in last 5 strong events) with $C \ge 0.40$ and intermediate score ($0.40 \le S \le 0.65$).
3. Rule 3 (`MASTERED`): Demonstrated high score ($S \ge 0.80$) with high confidence ($C \ge 0.60$).
4. Rule 4 (`DEVELOPING`): Demonstrated competency ($S \ge 0.50, C \ge 0.40$) approaching standard.
5. Rule 5 (`EMERGING`): Assessed low performance ($S < 0.50$ with $C \ge 0.25$) requiring foundational remediation, or early passing signal ($S \ge 0.50$) with nascent confidence ($C < 0.40$).
Rationale: Ensures semantic consistency across mathematical definitions, source code, and benchmarks. Evaluated failure is not `DEVELOPING`, and non-evaluative inquisitiveness does not fabricate competency signals.

## DEC-034 Multi-Signal Interpretable Concept Prioritization Policy — 2026-10-06 (Phase 7A)
Problem: Selecting what concept to assess next by simply picking the lowest knowledge score fails to account for estimation uncertainty, curriculum coverage, prerequisite dependencies, or repetitive fatigue.
Chosen: Implement `CandidatePrioritizer` protocol with `MultiSignalCandidatePrioritizer` evaluating five normalized, configurable signals: (1) Weakness ($1 - S$ for evaluated concepts), (2) Uncertainty ($1 - C$, or 1.0 for conflicting assessments), (3) Prerequisite importance (counting unmastered dependent concepts in the canonical graph), (4) Curriculum coverage need (asymptotic boost for unassessed concepts), and (5) Recency penalty (decaying penalty for concepts assessed within $T_{\text{window}}$). Generates transparent, human-readable rationales for every scored concept.
Rationale: Educational assessment serves dual purposes: diagnosing uncertain knowledge states and remediating demonstrated weaknesses. An interpretable, multi-signal policy explicitly surfaces why a concept is chosen without relying on opaque black-box heuristics or premature cognitive claims.

## DEC-035 Policy-Driven Deterministic Assessment Item Selection & Difficulty Matching — 2026-10-06 (Phase 7A)
Problem: Dynamically selecting an assessment item without introducing non-deterministic randomness, hallucinated questions, or language/difficulty mismatches.
Chosen: Implement `DeterministicItemSelector` adhering to three explicit policies: (1) Difficulty tier matching ($S < 0.40 \implies \text{EASY}$, $0.40 \le S < 0.75 \implies \text{MEDIUM}$, $S \ge 0.75 \implies \text{HARD}$), (2) Language preference with English fallback, and (3) Session repeat avoidance (prioritizing unseen items). Deterministic tie-breaking is enforced by `item_id`. Every selected item preserves source chunk provenance (`source_chunk_ids`).
Rationale: Decoupling question selection from generation enables rigorous algorithmic verification on a controlled question bank before introducing generative LLMs. Preserving chunk provenance ensures all assessment items remain grounded in the student's uploaded academic material.

## DEC-036 Closed-Loop Evidence Integration via Student Knowledge Model — 2026-10-06 (Phase 7A)
Problem: How student responses to assessment items should update the system state without violating architectural boundaries or coupling the assessment layer to knowledge tracing mathematics.
Chosen: The assessment engine does NOT mutate knowledge scores directly. Upon evaluating a student's answer deterministically, `AdaptiveAssessmentService` emits an `EvidenceEvent` (`QUIZ_CORRECT` or `QUIZ_INCORRECT` with `strength=STRONG`) to `KnowledgeStateService.record_event`. The Student Knowledge Model updates knowledge score and confidence via configurable evidence-weighted updates, confidence updates, and exponential recency decay. The next assessment item is then selected adaptively based on the updated state.
Rationale: Enforces clean unidirectional dependency: Assessment produces evidence events; the Student Model computes knowledge states. Keeps both subsystems modular, testable, and independently extensible.

## DEC-037 Multi-Signal Recommendation Prioritization with Graph Dependency Awareness — 2026-10-06 (Phase 7B)
Problem: Generating personalized study and remediation recommendations solely by sorting knowledge scores in ascending order overlooks estimation uncertainty, prerequisite bottlenecks in the curriculum, and repetitive fatigue.
Chosen: Implement `MultiSignalRecommendationPrioritizer` computing candidate priority over seven interpretable, normalized signals: (1) Demonstrated weakness ($W$, scaled by confidence and zero for unassessed concepts), (2) Uncertainty ($U$, max for conflicting evidence, moderate for cold-start), (3) Prerequisite deficit ($P_{\text{prereq}}$, unmastered upstream prerequisites), (4) Downstream blocker ($D_{\text{block}}$, unmastered concepts blocking downstream curriculum), (5) Review need ($R_{\text{need}}$, exponential decay since last interaction), (6) Assessment gap ($A_{\text{gap}}$, lack of strong evaluation), and (7) Recent-action penalty ($P_{\text{recent}}$, exponential damping for recently presented concepts). Mastered concepts receive a damping factor ($\delta = 0.35$).
Rationale: A remediation recommender must prioritize evidence-backed failure and curriculum bottlenecks over arbitrary unassessed concepts, while preserving candidate diversity and explainability.

## DEC-038 Decoupled Pedagogical Action Selection and Controlled Action Taxonomy — 2026-10-06 (Phase 7B)
Problem: Conflating recommendation priority (urgency of attention) with pedagogical activity (what specific study action to take) leads to rigid, inappropriate recommendations (e.g. recommending a hard practice quiz to a student lacking foundational prerequisites).
Chosen: Formulate an explicit 5-action taxonomy (`REVIEW_CONCEPT`, `REVIEW_PREREQUISITE`, `PRACTICE_CONCEPT`, `TAKE_ASSESSMENT`, `REVISIT_SOURCE`) and decouple action assignment into `PedagogicalActionSelector`. Action assignment follows a strict priority cascade: (1) Prerequisite deficit $\implies$ `REVIEW_PREREQUISITE`, (2) Conflicting evidence $\implies$ `TAKE_ASSESSMENT` (diagnostic), (3) Mastered competency $\implies$ `REVISIT_SOURCE` (spaced review), (4) Demonstrated weakness / developing competency $\implies$ `PRACTICE_CONCEPT` with difficulty matching, (5) Cold-start $\implies$ `REVIEW_CONCEPT`. When recommending `TAKE_ASSESSMENT`, the service delegates item selection directly to `AdaptiveAssessmentService` without duplicate logic.
Rationale: Clear separation of concerns enables independent tuning of priority heuristics and pedagogical decision rules, ensuring safe scaffolding before practice.

## DEC-039 Non-Deficit Evidence-Grounded Communication and Provenance Safety — 2026-10-06 (Phase 7B)
Problem: Recommender systems in education frequently issue blunt, demotivating deficit labels ("You are weak in Convolution"), or hallucinate textbook citations when source material is absent.
Chosen: Implement `RecommendationExplanationGenerator` adhering to the communicative template `evidence -> uncertainty -> recommendation -> reason`. Explanations explicitly cite observed evidence, score, confidence, and grounded source chunks (`CHUNK_SS_001`). For concepts lacking source material in the curriculum, the recommender explicitly marks `source_available = False`, assigns `ReasonCode.SOURCE_UNAVAILABLE`, and discloses that source material is unindexed, strictly prohibiting hallucinated citations.
Rationale: Research-grade educational technology must build learner trust through transparent, non-judgmental evidence explanations and uncompromising provenance integrity.
## DEC-040 Unified Self-Contained FastAPI + Static SPA Presentation Architecture — 2026-10-06 (Phase 8)
Problem: Providing an accessible, maintainable student learning dashboard and visual knowledge map without introducing fragile multi-process build pipelines, Node.js runtime locks, or heavyweight SPA dependencies.
Chosen: Implement a thin, decoupled FastAPI application layer (`indiclearn.api`) serving typed REST endpoints backed by `DashboardApplicationService`, alongside a responsive, dependency-free vanilla HTML/CSS/JavaScript single-page application served directly from FastAPI static file mounting.
Rationale: A self-contained Python-served frontend allows spinning up the entire educational platform (backend domain services, REST API, interactive 2D graph, and assessment modal) with a single command (`python -m uvicorn indiclearn.api.app:app`). It maintains absolute maintainability, instant startup, zero node build friction, and complete cross-platform portability.

## DEC-041 Interactive 2D SVG Graph Visualization with Topological Auto-Layout — 2026-10-06 (Phase 8)
Problem: Visualizing the academic concept graph and student knowledge overlay requires zoom, pan, directional prerequisite edges, and status rings without introducing bloated 3D canvas libraries (e.g. Three.js/D3) or sacrificing accessibility.
Chosen: Implement a native 2D SVG graph canvas with procedural topological layer-based auto-layout based on prerequisite DAG depth and relaxed ranks. Nodes render with status rings (`MASTERED`, `DEVELOPING`, `EMERGING`, `UNCERTAIN`, `UNKNOWN`), labels, and score badges; edges render with directional markers (`PREREQUISITE_OF`, `PART_OF`, `RELATED_TO`). An accessible tabular List View fallback is provided for non-visual, screen-reader, or keyboard navigation.
Rationale: Native SVG provides crisp vector rendering at any scale, negligible memory footprint, native CSS styling, and straightforward DOM event handling. The parallel table view guarantees accessibility compliance for diverse learners.

## DEC-042 Non-Deficit Evidence Transparency and Strict View-Layer Decoupling — 2026-10-06 (Phase 8)
Problem: Dashboards frequently act as a second intelligence layer by recalculating mastery thresholds, predicting student success, or presenting heuristic model scores as absolute psychological facts ("You know this 70%").
Chosen: Enforce strict view-layer decoupling: the frontend performs ZERO scoring, mastery classification, recommendation ranking, or item selection. All metrics, action cards, and assessment sequences originate from backend domain services. All UI copy communicates "Estimated knowledge" and "Evidence confidence", paired with a collapsible "Why this estimate?" evidence transparency panel detailing historical assessment and interaction events.
Rationale: Protects student agency and epistemological integrity. Demonstrating evidence provenance and model estimation limits prevents false psychological certainty and fosters metacognitive awareness.

## DEC-043 Pre-Registered Experiment Registry and Execution Lifecycle — 2026-10-06 (Phase 9)
Problem: Research systems often suffer from informal benchmarking, post-hoc metric alteration, unrecorded parameter tweaking, or ambiguous execution status where proposed ideas are conflated with validated trials.
Chosen: Implement a formal machine-readable Experiment Registry (`data/evaluation/experiment_registry.json`) tracking experiments EXP-001 through EXP-014. Each entry requires a pre-registered research question, hypothesis, explicit metric targets, dataset path, sample size, random seed, software version, and five-state lifecycle: `PROPOSED`, `IMPLEMENTED`, `RUN`, `HUMAN_DATA_REQUIRED`, or `COMPLETED`. An experiment is strictly barred from `COMPLETED` unless genuine empirical data exists.
Rationale: Guarantees reproducible, open-science scientific methodology. Prevents research drift, retrospective benchmark tampering, and silent specification changes.

## DEC-044 Statistical and Agreement Metric Suite with Missing-Data Protection — 2026-10-06 (Phase 9)
Problem: Computational educational tools frequently report uncorrected p-values, misleading point estimates without confidence intervals, or compute inter-rater agreement even when human data is absent or trivial.
Chosen: Implement an isolated, dependency-light statistical suite (`indiclearn.evaluation.statistics`) supporting descriptive statistics (Bessel-corrected variance, IQR, medians), exact inter-rater agreement (unweighted/weighted Cohen's kappa, Fleiss' kappa with Landis & Koch scales), inferential testing (bootstrap percentile CIs, Student's t CIs, Cohen's d, Mann-Whitney U, Wilcoxon signed-rank), and multiple-hypothesis corrections (Holm-Bonferroni). When evaluation data is absent ($N=0$), the system strictly returns `status: NOT_AVAILABLE` rather than computing degenerate zeros or default values.
Rationale: Prevents p-hacking and statistical fabrication. Enforces mathematical rigor and honest reporting of data sufficiency.

## DEC-045 Four-Tier Epistemic Evaluation Taxonomy and Anti-Fabrication Safeguards — 2026-10-06 (Phase 9)
Problem: Educational AI papers frequently conflate software correctness, retrieval accuracy, or synthetic benchmark performance with real human usability and cognitive learning gains.
Chosen: Establish a strict four-tier epistemic evaluation hierarchy:
  1. Software Verification (unit/integration tests, invariant checks)
  2. Controlled Algorithmic Benchmarks (EXP-005A, EXP-005B, EXP-006, EXP-007, EXP-008, EXP-009, EXP-010, EXP-011)
  3. Human Usability & Linguistic Evaluation (EXP-012, EXP-013, EXP-014 — returned as `NOT_RUN` / `HUMAN_DATA_REQUIRED` pending human recruitment)
  4. Controlled Learning-Effectiveness Trials (Pre/Post/Delayed-retention RCT protocol documented in `docs/LEARNING_EFFECTIVENESS_STUDY.md` as future work)
The framework strictly prohibits fabricating synthetic student participants, fake Likert ratings, simulated human p-values, or claiming educational effectiveness without empirical human trials.
Rationale: Upholds scientific ethics, institutional compliance, and epistemological honesty in educational NLP research.


