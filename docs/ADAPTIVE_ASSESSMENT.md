# ADAPTIVE ASSESSMENT CORE (PHASE 7A)

**Status**: Phase 7A Adaptive Assessment Core is **IMPLEMENTED & VERIFIED**.

---

## 1. Executive Summary & Design Principles

IndicLearn's Phase 7A establishes an **interpretable, multi-signal, evidence-driven Adaptive Assessment Core**. It dynamically answers the foundational pedagogical question:

> *"Given the student's current knowledge state and the academic knowledge graph, what concept should be assessed next, and what assessment item should be presented?"*

The system specifically avoids simplistic heuristics such as `"lowest knowledge score = always test that concept"`. Instead, it evaluates multiple orthogonal signals—weakness, uncertainty, prerequisite importance, curriculum coverage, and recency—to plan diagnostic and formative evaluations.

### Core Architectural Invariants

1. **Strict Architectural Tripartite Separation**:
   - **Academic Knowledge Graph** (`Concept`, `ConceptRelation`, `RelationType`): Objective, canonical subject matter structure. Read-only; never mutated with student IDs, scores, or assessment counts.
   - **Student Knowledge Model** (`KnowledgeState`, `EvidenceEvent`, `MasteryStatus`): Dynamic per-student knowledge estimates ($Student \times Concept$). Read-only during candidate prioritization; updated strictly via `EvidenceEvent` emission.
   - **Adaptive Assessment Layer** (`AssessmentCandidate`, `AssessmentItem`, `AdaptiveAssessmentService`): Plans candidate concepts, matches items, manages sessions, and produces evidence.

2. **Closed-Loop Evidence Integration**:
   - When a student answers an assessment item, the assessment system does **NOT** directly mutate the student's knowledge score.
   - It deterministically evaluates the response, packages the outcome into an `EvidenceEvent` (`QUIZ_CORRECT` or `QUIZ_INCORRECT` with `strength=STRONG`), and submits it to `KnowledgeStateService.record_event`.
   - The Student Knowledge Model remains the sole authority for belief updates.

3. **Academic Grounding & Provenance**:
   - Every assessment item preserves `source_chunk_ids` linking directly to the student's uploaded academic course material.
   - Assessment is grounded in curricular text, not unsupported model knowledge.

4. **Transparent Explainability**:
   - Every candidate prioritization and item selection returns explicit, human-readable rationales answering:
     - *Why was this concept selected?*
     - *Why this difficulty tier?*
     - *Why this specific question?*

5. **No Uncalibrated Claims**:
   - Weights, decay rates, and difficulty tiers are explicitly designated as **heuristic research defaults**.
   - No psychometric Item Response Theory (IRT) calibration, cognitive predictive validity, or learning gain is claimed without real-world educational trials.

---

## 2. Adaptive Feedback Loop Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                 Academic Knowledge Graph                     │
│  - Concepts, Relations, Prerequisites (Canonical Domain)   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Read-Only Prerequisite Traversal)
                               ▼
┌──────────────────────────────┴──────────────────────────────┐
│                  Student Knowledge Model                    │
│  - KnowledgeState (Score, Confidence, History, Status)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Read-Only Knowledge States)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                Candidate Generation Layer                   │
│  - Filter eligible concepts (preserves UNKNOWN/UNCERTAIN)   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│             Multi-Signal Prioritizer Layer                  │
│  - Weakness + Uncertainty + Prerequisite + Coverage - Rec   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Prioritized AssessmentCandidates)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 Item Selection Layer                        │
│  - Difficulty Matching + Language + Repeat Avoidance        │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Selected AssessmentItem)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                  Assessment Session                         │
│  - Presentation -> Student Response -> Deterministic Grader │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           EvidenceEvent (QUIZ_CORRECT / INCORRECT)          │
│  - Strength: STRONG, Score: 1.0 / 0.0, Source: Provenance   │
└──────────────────────────────┬──────────────────────────────┘
                               │ (Emitted Event)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│         KnowledgeStateService.record_event()                │
│  - Recency Decay + Evidence Accumulation -> Updated State   │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Candidate Prioritization Policy

The prioritization score in $[0.0, 1.0]$ is computed deterministically by [`MultiSignalCandidatePrioritizer`](file:///d:/Indic_NLP/src/indiclearn/assessment/prioritizer.py):

$$\text{Priority} = \text{clamp}\left(w_w \cdot \text{Weakness} + w_u \cdot \text{Uncertainty} + w_p \cdot \text{Prereq} + w_c \cdot \text{Coverage} - w_r \cdot \text{RecencyPenalty}, 0.0, 1.0\right)$$

### Heuristic Default Weights (`configs/config.yaml`)

| Parameter | Weight | Description |
| :--- | :--- | :--- |
| $w_{\text{weakness}}$ | **0.35** | Prioritizes concepts where the student demonstrates validated difficulty ($S < 0.50$). |
| $w_{\text{uncertainty}}$ | **0.25** | Prioritizes concepts with low confidence ($C < 0.30$) or conflicting outcomes (`UNCERTAIN`). |
| $w_{\text{prerequisite}}$ | **0.20** | Boosts foundational concepts that block active, unmastered dependent concepts. |
| $w_{\text{coverage}}$ | **0.12** | Boosts curriculum concepts that have never or rarely been assessed ($N = 0$). |
| $w_{\text{recency}}$ | **0.08** | Penalizes concepts assessed very recently to avoid immediate repetitive fatigue. |

### Signal Formulations

1. **Weakness Signal**:
   - If evaluative evidence exists ($C \ge 0.25$ or strong assessments $> 0$):
     $$\text{Weakness} = 1.0 - S$$
   - If cold start / no evaluative evidence ($C < 0.25$):
     $$\text{Weakness} = 0.50 \quad (\text{Neutral prior baseline; not falsely marked weak})$$

2. **Uncertainty Signal**:
   - If `mastery_status == UNCERTAIN` or conflicting flags set:
     $$\text{Uncertainty} = 1.0$$
   - Else:
     $$\text{Uncertainty} = 1.0 - C$$

3. **Prerequisite Graph Importance**:
   - Queries canonical graph dependents: $\text{dependents} = \text{GraphStore.get\_dependents}(c)$.
   - Counts dependent concepts whose student state is not yet `MASTERED`:
     $$\text{Prereq} = \min(1.0, 0.40 \cdot \text{unmastered\_dependents})$$

4. **Curriculum Coverage Need**:
   - If total assessments $N = 0$: $\text{Coverage} = 1.0$.
   - If $N > 0$: $\text{Coverage} = \frac{1.0}{1.0 + 0.5 \cdot N}$.

5. **Recency Penalty**:
   - For time elapsed $\Delta t_{\text{min}} = (\text{now} - \text{last\_assessed\_at})$:
     $$\text{RecencyPenalty} = \max\left(0.0, 1.0 - \frac{\Delta t_{\text{min}}}{T_{\text{window}}}\right) \quad (\text{Default } T_{\text{window}} = 60\text{ min})$$

---

## 4. Item Selection & Adaptation Policies

Implemented in [`DeterministicItemSelector`](file:///d:/Indic_NLP/src/indiclearn/assessment/selector.py):

### 1. Difficulty Tier Matching Policy
- **`EASY`**: Selected when $S < 0.40$, $C < 0.30$, or status is `EMERGING`. Focuses on foundational recall and definitional comprehension.
- **`MEDIUM`**: Selected when $0.40 \le S < 0.75$ or status is `DEVELOPING`. Focuses on standard application and property analysis.
- **`HARD`**: Selected when $S \ge 0.75$ and $C \ge 0.50$ (approaching or at `MASTERED`). Focuses on boundary conditions, proofs, and multi-concept synthesis.
- *Fallback*: If target difficulty is unavailable, adjacent tiers (`MEDIUM` $\leftrightarrow$ `EASY` / `HARD`) are utilized cleanly.

### 2. Language Preference Policy
- Supports `"en"` (English), `"te"` (Telugu script), `"te-Latn"` (Romanized Telugu), and `"te-en"` (Code-mixed).
- Exact match on requested language variety is prioritized (+30 bonus).
- If unavailable, gracefully falls back to the canonical English course item (+10 fallback score).

### 3. Repeat Avoidance Policy
- Checks `seen_item_ids` within the active assessment session and recent student exposure history.
- Fresh unseen items receive a +50 selection bonus.
- If all items for a concept have been presented, the selector safely cycles through the least recently presented item.

---

## 5. Domain Models & Protocols

Implemented in [`src/indiclearn/assessment/models.py`](file:///d:/Indic_NLP/src/indiclearn/assessment/models.py):

- **`QuestionType`**: `MCQ`, `TRUE_FALSE`, `SHORT_ANSWER`.
- **`ItemDifficulty`**: `EASY`, `MEDIUM`, `HARD`.
- **`AssessmentItem`**: Item ID, concept ID, subject ID, question prompt, options, correct answer, explanation, and `source_chunk_ids`.
- **`AssessmentCandidate`**: Concept ID, priority score, knowledge score, confidence, mastery status, normalized signals, and explainability reasons.
- **`AssessmentResponse`**: Session ID, item ID, student ID, student's selected answer, deterministic correctness boolean, and feedback.
- **`AssessmentSession`**: Session ID, student ID, presented item IDs, response log, active flag, and timestamps.
- **`AssessmentSelectionExplanation`**: Full explainability capsule combining candidate reasons, difficulty rationale, and item selection justifications.

---

## 6. Controlled Question Bank

Stored in [`data/evaluation/signals_systems_items.json`](file:///d:/Indic_NLP/data/evaluation/signals_systems_items.json):
- 16 curated, academic items for Signals and Systems.
- Covers canonical concepts: Convolution, LTI Systems, Linearity, Time-Invariance, CTFT, Laplace Transform, Sampling Theorem, Energy/Power Signals.
- Preserves 100% provenance tracing to chunks (`CHUNK_SS_001` through `CHUNK_SS_010`) in `data/evaluation/signals_and_systems_eval.json`.

---

## 7. Verification & Benchmark (EXP-007)

Automated evaluation runner in [`src/indiclearn/evaluation/exp007_runner.py`](file:///d:/Indic_NLP/src/indiclearn/evaluation/exp007_runner.py):
- **10 Controlled Scenarios**:
  1. Low Knowledge Concept Prioritized over Mastered ($P_A = 0.4191 > P_B = 0.2611$)
  2. Explicit Uncertainty Representation ($U_A = 0.90 \ge 0.70$ vs $W_B = 0.82 \ge 0.60$)
  3. Prerequisite Graph Importance Boost (LTI boosted by unmastered dependent Convolution)
  4. Coverage Advantage over Recently Assessed Concept ($P_{\text{unassessed}} > P_{\text{recent}}$)
  5. Policy-Driven Difficulty Tier Matching (EASY, MEDIUM, HARD matching student states)
  6. Repeat Avoidance (alternating items on subsequent selections)
  7. Language Preference (Telugu requested $\to$ Telugu item selected)
  8. Provenance Integrity (100% items cite valid course chunks)
  9. Closed-Loop Evidence Integration (`QUIZ_CORRECT` updates Student Model via `KnowledgeStateService`)
  10. Strict Multi-Student Isolation (Alice's session leaves Bob at cold start)
- **Results**:
  - **10 / 10 Scenarios Passed (100.0%)**
  - **21 / 21 Invariants Passed (100.0%)**

---

## 8. Scientific Limitations & Boundaries

1. **No Real Student Learning Claims**: EXP-007 is a controlled synthetic evaluation validating software and algorithmic invariants. It does **NOT** measure student grade improvement, retention, or learning efficiency.
2. **No Psychometric Calibration**: Item difficulties (`EASY`, `MEDIUM`, `HARD`) and concept prioritization weights are heuristic engineering defaults. They are not calibrated using Item Response Theory (IRT) or Rasch models.
3. **Deterministic Fixture Bank**: Phase 7A utilizes a curated, deterministic item repository. It does not employ an LLM for live question generation.
4. **No Semantic Grading**: Short answer and free-form responses are evaluated using normalized string matching. Neural NLP semantic grading is intentionally excluded to maintain deterministic verification.
