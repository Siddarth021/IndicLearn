# IndicLearn Studio

**A multilingual study studio for students.** Upload your notes → read them in the app → ask a tutor bot in
**English, తెలుగు, हिन्दी or संस्कृतम्** → take quizzes → watch a **personal mind map** grow from your questions and quiz results.
Everything runs **locally – no hosted LLM or external API** is needed.

| | |
| :-- | :-- |
| Upload & read | PDF, PPTX, DOCX, Markdown, TXT. Structure-aware parsing (headings from font sizes / slide titles / styles). |
| Default knowledge map | Built *only* from the document: topic hierarchy, prerequisite flow, related links. Same for every student. |
| Personal map | The default map + your mastery colours, progress rings and "doubt" nodes created by questions the notes could not answer. |
| Tutor bot | Grounded, extractive answers with page citations; replies in the language you write in; refuses to guess. |
| Quizzes | Cloze, "which definition", true/false – generated from your own notes; every answer updates the map. |
| Learn from the map | Hover/click for definitions, examples, prerequisites; Radial / Tree / Flow layouts; guided tour along the learning path. |

## Run it

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m indiclearn.studio      # http://127.0.0.1:8000
# or: run.bat (Windows) · ./run.sh (Linux/macOS) · docker compose up --build
```

Configuration is by environment variables (see `.env.example`). Optional **local** models (never a hosted API):

```bash
pip install -r requirements-models.txt          # sentence-transformers / transformers / torch
INDICLEARN_EMBEDDING_BACKEND=sentence_transformers   # dense multilingual retrieval (multilingual-e5-small)
INDICLEARN_TRANSLATOR_BACKEND=nllb                   # translate quoted passages into the chat language
INDICLEARN_LLM_BACKEND=ollama                        # a local Ollama daemon rephrases the grounded answer
```

Tests: `PYTHONPATH=src pytest` (271 tests: 252 research-pipeline + 19 studio).
The previous dark dashboard is kept in `legacy_frontend_dark/`; the original research pipeline (`src/indiclearn/{rag,nlp,...}`) is untouched.
Read **docs/STUDIO_GUIDE** (the project explainer) for how every part works.

---

# Research track (original prototype)


**A Dynamic Personalized Knowledge Map for Engineering Students using Indic Language NLP** — Research-Grade Prototype.

Initial scope: Telugu (native script, Romanized, Telugu–English code-mixed) + English; academic subject demo: Signals and Systems.

---

## Status Summary

```text
250 unit and integration tests passing
Ruff clean (zero linter warnings)
```

| Phase | Description | Status |
| :--- | :--- | :---: |
| **Phase 1** | Document Ingestion, Normalization & Recursive Chunking | **TESTED** |
| **Phase 2** | Multilingual Embeddings, Vector Stores & Hybrid Retrieval (EXP-002) | **TESTED** |
| **Phase 3** | Academic Knowledge Graph Construction & Validation (EXP-003) | **TESTED** |
| **Phase 4** | Indic NLP (Language/Code-Mix Detection, Romanization, Concept Mapping) (EXP-004) | **TESTED** |
| **Phase 5** | Scope-Controlled RAG & Grounded Answer Generation (EXP-005) | **TESTED** |
| **Phase 5.1** | RAG Research Hardening & Graph-Enrichment Ablations (EXP-005A) | **TESTED** |
| **Phase 5.2** | Scope Control & Adversarial Query Hardening (EXP-005B) | **TESTED** |
| **Phase 6** | Student Knowledge Modeling & Evidence-Weighted Updates (EXP-006) | **TESTED** |
| **Phase 6.1** | Student Model Consistency & Epistemic Audit | **TESTED** |
| **Phase 7A** | Adaptive Assessment Core & Question Selection (EXP-007) | **TESTED** |
| **Phase 7B** | Personalized Recommendations & Remediation Planner (EXP-008) | **TESTED** |
| **Phase 8** | Visual Knowledge Map & Student Learning Dashboard (EXP-009) | **TESTED** |
| **Phase 9** | Research Evaluation & Human Validation Framework (EXP-010 to EXP-014) | **TESTED** |

---

## Controlled Evaluation & Validation Matrix

| Experiment | Focus Area | Status | Verified Metric Summary |
| :--- | :--- | :---: | :--- |
| **EXP-005A** | RAG Multilingual Retrieval & Ablations | **PASS** | R@5 = 0.9200, MRR = 0.5677, Gold Evidence R@5 = 0.9100 |
| **EXP-005B** | Scope Control & Adversarial Hard Negatives | **PASS** | Scope Accuracy = 1.0000, False Acceptance Rate = 0.0000 |
| **EXP-006** | Student Knowledge Model Invariants | **PASS** | 10/10 scenarios passed, 27/27 invariants (100.0%) |
| **EXP-007** | Adaptive Assessment Policy Invariants | **PASS** | 10/10 scenarios passed, 21/21 invariants (100.0%) |
| **EXP-008** | Recommendation & Remediation Invariants | **PASS** | 12/12 scenarios passed, 33/33 invariants (100.0%) |
| **EXP-009** | Presentation & Dashboard Invariants | **PASS** | 15/15 scenarios passed (100.0%) |
| **EXP-010** | Recommendation Policy & Rule Consistency | **PASS** | 10/10 scenarios passed (100.0%), Top-1 Accuracy = 1.0000 |
| **EXP-011** | Adaptive Assessment Diagnostic Policy | **PASS** | 10/10 scenarios passed (100.0%), Repeat Avoidance = 1.0000 |
| **EXP-012** | Dashboard Human Usability Framework | **NOT_RUN** | Framework verified; $N=0$; `HUMAN_DATA_REQUIRED` |
| **EXP-013** | RAG Human Annotation Framework | **NOT_RUN** | Framework verified (7 dims); $N=0$; `HUMAN_DATA_REQUIRED` |
| **EXP-014** | Multilingual Human Evaluation Framework | **NOT_RUN** | Framework verified (4 tracks); $N=0$; `HUMAN_DATA_REQUIRED` |

*Scientific Policy*: Zero fabricated participant data. Experiments EXP-012 through EXP-014 are strictly marked `HUMAN_DATA_REQUIRED` / `NOT_RUN` pending human-subject recruitment under institutional ethical review.

---

## Setup & Execution

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env
```

### 2. Testing and Code Quality
```bash
python -m pytest -q
python -m ruff check .
```

### 3. Running Controlled Benchmarks
```bash
# Core Benchmarks (Phases 5-8)
python scripts/run_exp005a.py
python scripts/run_exp005b.py
python scripts/run_exp006.py
python scripts/run_exp007.py
python scripts/run_exp008.py
python scripts/run_exp009.py

# Phase 9 Evaluation Framework (EXP-010 to EXP-014)
python scripts/run_exp010.py
python scripts/run_exp011.py
python scripts/run_exp012.py
python scripts/run_exp013.py
python scripts/run_exp014.py
```

### 4. Running the Dashboard Application
```bash
python -m uvicorn indiclearn.api.app:app --reload --port 8000
```
Open `http://localhost:8000` to interact with the student learning dashboard and visual knowledge map.

---

## Epistemic Hierarchy & Research Governance

IndicLearn strictly distinguishes four levels of evaluation:
1. **Software Verification**: Automated test suites validating unit logic, data structures, and invariants.
2. **Controlled Algorithmic Benchmarks**: Deterministic evaluation of retrieval, extraction, scope control, and policy selection against curated gold standards.
3. **Human Usability & Linguistic Evaluation**: Protocols, guidelines, and schemas for empirical studies with native speakers (`docs/HUMAN_EVALUATION_PROTOCOL.md`).
4. **Controlled Learning-Effectiveness Trials**: Pre/post/delayed-retention randomized controlled trial protocol for future educational research (`docs/LEARNING_EFFECTIVENESS_STUDY.md`).

**No learning-effectiveness claims are made** without completing empirical human-subject trials.

---

## Documentation Index

- `docs/PROJECT_CONTEXT.md`: Single source of truth for project status, architecture, and constraints.
- `docs/RESEARCH_EVALUATION.md`: Formal research questions, registry summary, and statistical criteria.
- `docs/HUMAN_EVALUATION_PROTOCOL.md`: Protocol, participant inclusion/exclusion, and tasks T1-T9.
- `docs/LEARNING_EFFECTIVENESS_STUDY.md`: Pre/post RCT trial design with exact statistical power analysis ($N=128$ completing, $N=152$ enrolled).
- `docs/DECISIONS.md`: Architectural Decision Records (DEC-001 through DEC-045).
- `docs/EXPERIMENTS.md`: Full empirical reports for EXP-001 through EXP-014.
- `docs/ARCHITECTURE.md`: Subsystem layer descriptions.
- `docs/HANDOFF.md`: Comprehensive phase-by-phase handoff history.
