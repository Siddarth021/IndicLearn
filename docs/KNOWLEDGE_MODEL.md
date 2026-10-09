# STUDENT KNOWLEDGE MODELING & EVIDENCE-BASED PERSONALIZATION

**Status**: Phase 6 Student Knowledge Model is **IMPLEMENTED & VERIFIED**.

---

## 1. Executive Summary & Design Principles

IndicLearn's Phase 6 establishes an **interpretable, evidence-based, scientifically principled Student Knowledge Model**. It tracks student understanding at the concept level to enable personalized pedagogical guidance.

The fundamental architectural principle governing this module is:
> **The canonical Academic Knowledge Graph describes the academic subject domain. The Student Knowledge Model describes the student's estimated understanding regarding concepts ($Student \times Concept$). These two layers MUST remain strictly separated.**

### Core Tenets

1. **Separation of Subject vs Learner**:
   - The canonical academic graph (`Concept`, `ConceptRelation`) is objective, curricular, and completely invariant to individual student actions.
   - The student model (`KnowledgeState`, `EvidenceEvent`) stores private, dynamic mastery estimates for a student-concept pair.
   - The academic graph is never mutated with student IDs, mastery scores, or ignorance flags.

2. **A Question is NOT Proof of Ignorance**:
   - Asking a question (e.g. *"What is convolution?"*) demonstrates curiosity, engagement, or active study. It is **weak, observational interaction evidence**, NEVER conclusive evidence that the student has zero knowledge of the concept.
   - Weak evidence contributes minimally to score updates and yields strictly bounded confidence increases ($\Delta C = 0.02$).

3. **Explicit Uncertainty Representation**:
   - Knowledge score and confidence are separate dimensions.
   - $S = 0.70, C = 0.15$ indicates an early estimate supported by minimal evidence.
   - $S = 0.70, C = 0.85$ indicates substantial assessment evidence supporting the estimate.
   - Uncertainty is never collapsed into a single scalar.

4. **Transparent, Interpretable Estimation**:
   - Initial model avoids opaque black-box neural knowledge tracing.
   - Uses an evidence-based estimator with configurable weights, recency decay, and provenance logging.
   - All parameters are labeled `PROPOSED / HEURISTIC` rather than claiming calibrated cognitive probabilities.

5. **Prerequisite-Aware Context Without Blind Propagation**:
   - The model inspects canonical prerequisite edges to analyze readiness.
   - It **never** blindly infers: *"Weak in A $\rightarrow$ Weak in all downstream descendants"* without explicit evidence.

---

## 2. Evidence Hierarchy & Taxonomy

IndicLearn categorizes all incoming student signals into a three-tier evidence hierarchy:

```text
                     EVIDENCE HIERARCHY
┌─────────────────────────────────────────────────────────────┐
│ 1. STRONG EVIDENCE (Weight = 1.00, Delta C = 0.20)          │
│    - QUIZ_CORRECT, QUIZ_INCORRECT                           │
│    - ASSESSMENT_RESULT (Validated score in [0.0, 1.0])       │
│    - SELF_ASSESSMENT (Explicit student evaluation)          │
├─────────────────────────────────────────────────────────────┤
│ 2. MEDIUM EVIDENCE (Weight = 0.40, Delta C = 0.08)          │
│    - SUCCESSFUL_TASK, UNSUCCESSFUL_TASK                     │
│    - Evaluated application of concepts                      │
├─────────────────────────────────────────────────────────────┤
│ 3. WEAK EVIDENCE (Weight = 0.10, Delta C = 0.02)            │
│    - QUESTION_ASKED (RAG queries, curiosity interactions)   │
│    - CONCEPT_VIEWED (Reading, UI clicks)                    │
│    - EXPLANATION_REQUESTED (Seeking clarification)          │
└─────────────────────────────────────────────────────────────┘
```

### Why Weak Evidence Must Not Dominate
In engineering education, students frequently ask foundational questions before exams as quick refresher checks. If asking a question caused the system to immediately mark the student as failing that concept, the student would be discouraged from asking questions. Conversely, asking 10 questions without assessment must not grant mastery.

---

## 3. Data Model Architecture

The domain entities are implemented in [`src/indiclearn/student_model/models.py`](file:///d:/Indic_NLP/src/indiclearn/student_model/models.py):

### `KnowledgeState`
Represents the estimated state for a single $Student \times Concept$:
- `student_id`: Unique identifier for the learner.
- `concept_id`: Canonical concept identifier from the academic knowledge graph.
- `subject_id`: Scoping subject ID (default: `"signals_and_systems"`).
- `knowledge_score`: Estimated understanding in $[0.0, 1.0]$ (heuristic, uncalibrated).
- `confidence`: Estimated certainty in $[0.0, 1.0]$ based on evidence volume and recency.
- `evidence_count`: Total events applied.
- `strong_evidence_count`: Validated assessment count.
- `medium_evidence_count`: Evaluated task count.
- `weak_evidence_count`: Observational interaction count.
- `mastery_status`: Categorical status (`UNKNOWN`, `EMERGING`, `DEVELOPING`, `MASTERED`, `UNCERTAIN`).
- `evidence_history`: Chronological list of `KnowledgeEvidence` summaries preserving full provenance.
- `last_assessed_at`: Timestamp of most recent assessment.
- `last_interaction_at`: Timestamp of most recent interaction.

### `MasteryStatus` (Authoritative Source of Truth)

Mastery status is computed deterministically by [`EvidenceBasedKnowledgeEstimator.compute_mastery_status`](file:///d:/Indic_NLP/src/indiclearn/student_model/estimator.py) using an explicit priority cascade:

| Status | Semantic Meaning | Actual Implementation Condition |
| :--- | :--- | :--- |
| **`UNKNOWN`** | Insufficient or non-evaluative evidence | `(strong_count == 0 and medium_count == 0)` OR `confidence < 0.25` |
| **`UNCERTAIN`** | Conflicting assessment signals | `has_conflicting_evidence` (variance in last 5 strong scores $\ge 0.15$) AND `confidence >= 0.40` AND `0.40 <= score <= 0.65` |
| **`EMERGING`** | Nascent/early competency: assessed low score ($S < 0.50$) OR early passing signal ($S \ge 0.50$) with low confidence ($C < 0.40$) | Has evaluative evidence (`strong_count > 0` or `medium_count > 0`), NOT `UNCERTAIN`, and either `(score < 0.50 and confidence >= 0.25)` OR `(score >= 0.50 and confidence < 0.40)` |
| **`DEVELOPING`** | Demonstrated developing competency approaching standard | Has evaluative evidence, NOT `UNCERTAIN`, and either `(0.50 <= score < 0.80 and confidence >= 0.40)` OR `(score >= 0.80 and 0.40 <= confidence < 0.60)` |
| **`MASTERED`** | Verified mastery meeting both score and confidence standards | Has evaluative evidence, NOT `UNCERTAIN`, `score >= 0.80`, and `confidence >= 0.60` |

#### Key Invariants & Audit Resolutions (Phase 6.1)
- **SCEN_002 Resolution (Monotonic Incorrect Assessment)**:
  A student with repeated failures ($S = 0.0750, C = 0.7438$) receives **`EMERGING`**, representing assessed low performance requiring foundational remediation. A previous fallthrough bug in `estimator.py` that assigned `DEVELOPING` to $S < 0.50$ was corrected, and the benchmark expectation aligned.
- **Weak Evidence Restraint (`QUESTION_ASKED`, `CONCEPT_VIEWED`)**:
  Observational interactions do not constitute evaluative proof of understanding or ignorance. Without strong/medium assessments (`strong_count == 0 and medium_count == 0`), status remains strictly **`UNKNOWN`** regardless of interaction volume.
- **Score vs Confidence Separation**:
  A score of $S = 0.70$ with $C = 0.15$ yields `UNKNOWN` (insufficient evidence), whereas $S = 0.70$ with $C = 0.85$ yields `DEVELOPING` (verified solid progress). Score and confidence are never conflated.

---

## 4. Estimator Mechanics (`EvidenceBasedKnowledgeEstimator`)

Implemented in [`src/indiclearn/student_model/estimator.py`](file:///d:/Indic_NLP/src/indiclearn/student_model/estimator.py):

### 1. Cold Start Initialization
- Prior Knowledge Score: $S_0 = 0.50$
- Prior Confidence: $C_0 = 0.10$
- Prior Weight: $w_{prior} = 1.00$
- Initial Status: `MasteryStatus.UNKNOWN`

### 2. Recency Weighting
Exponential time decay models the fading validity of older evidence:
$$\text{decay}_i = \exp\left(-\frac{\ln(2)}{\text{half\_life\_days}} \cdot \Delta t_{days}\right)$$
Default half-life is set to 30.0 days (`configs/config.yaml`).

### 3. Knowledge Score Accumulation
Score is updated as the recency-weighted average of prior belief and all historical events:
$$S = \frac{w_{prior} \cdot S_{0} + \sum_{i=1}^N (w_i \cdot \text{decay}_i) \cdot s_i}{w_{prior} + \sum_{i=1}^N (w_i \cdot \text{decay}_i)}$$
- Monotonic increase under repeated correct answers ($s_i = 1.0$).
- Monotonic decrease under repeated incorrect answers ($s_i = 0.0$).
- Stable convergence under mixed signals.

### 4. Asymptotic Confidence Growth
Confidence increases monotonically with evidence, strictly bounded in $[0.10, 1.00)$:
$$C = 1.0 - (1.0 - C_0) \cdot \prod_{i=1}^N (1.0 - \Delta_i \cdot \text{decay}_i)$$
Where $\Delta_i \in \{0.20, 0.08, 0.02\}$.

### 5. Conflicting Evidence Detection
The estimator inspects the variance across the last 5 strong assessment events:
$$\text{Var}(s_i) = \frac{1}{M} \sum_{k=1}^M (s_k - \bar{s})^2$$
If $\text{Var}(s_i) \ge 0.15$ and confidence $\ge 0.40$ with score in $[0.40, 0.65]$, the state is flagged as conflicting and assigned `MasteryStatus.UNCERTAIN`.

---

## 5. Service & Persistence Architecture

Implemented in [`src/indiclearn/student_model/service.py`](file:///d:/Indic_NLP/src/indiclearn/student_model/service.py) and [`src/indiclearn/student_model/repository.py`](file:///d:/Indic_NLP/src/indiclearn/student_model/repository.py):

```text
┌─────────────────────────────────────────────────────────┐
│               KnowledgeStateService                     │
│  - record_event(EvidenceEvent)                          │
│  - record_interaction(student, concept, type, score)    │
│  - get_concept_state(student, concept)                  │
│  - get_student_summary(student)                         │
│  - get_weak_concepts(student)                           │
│  - get_uncertain_concepts(student)                      │
│  - explain_concept_state(student, concept)              │
│  - get_prerequisite_readiness(student, target_concept)  │
└───────────────┬─────────────────────────┬───────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────────────┐ ┌───────────────────────┐
│  KnowledgeStateRepository     │ │  KnowledgeEstimator   │
│  (InMemory / Future DB)       │ │  (Evidence-Weighted)  │
└───────────────────────────────┘ └───────────────────────┘
                ▲
                │ (Contextual Prerequisite Queries Only)
┌───────────────┴───────────────┐
│  Academic GraphStore          │
│  (Canonical Domain Graph)     │
└───────────────────────────────┘
```

### Complete Provenance & Explainability
Calling `service.explain_concept_state(student_id, concept_id)` returns a structured explanation answering:
> *"Why does IndicLearn believe this student is in this state?"*
It breaks down assessment count, interaction count, score trajectory, and conflicting evidence flags.

---

## 6. Prerequisite-Aware Readiness Inspection

The service queries direct prerequisites from the canonical graph via `graph_store.get_prerequisites(target_concept_id)`.
- Inspects whether each prerequisite concept has established understanding ($S \ge 0.50$ and status $\in \{\text{DEVELOPING}, \text{MASTERED}\}$).
- Returns `PrerequisiteReadiness` with specific unmastered prerequisites and recommendations.
- **Strict Invariant**: Does **NOT** alter the target concept's state. Readiness is pedagogical advisory context only.

---

## 7. Verification & Empirical Benchmark (EXP-006)

The student knowledge model is verified by:
- 15 comprehensive unit tests in [`tests/unit/test_student_model.py`](file:///d:/Indic_NLP/tests/unit/test_student_model.py) covering cold start, monotonicity sequences, weak evidence restraint, score/confidence separation, and conflicting evidence handling.
- EXP-006 benchmark (`scripts/run_exp006.py`, `data/evaluation/exp006_results.json`):
  - 10 controlled scenarios.
  - 27 mathematical invariants checked.
  - **100% Scenario Pass Rate** (10 / 10).
  - **100% Invariant Pass Rate** (27 / 27).

---

## 8. Scientific Limitations & Boundaries

1. **Synthetic Validation**: EXP-006 is evaluated on synthetic scenarios designed to test mathematical invariants (monotonicity, stability, bounds). It does **NOT** represent empirical student learning data.
2. **Heuristic Parameters**: Weights ($1.0, 0.4, 0.1$), confidence deltas ($0.20, 0.08, 0.02$), and decay half-life (30 days) are proposed heuristic defaults. They are not psychometrically calibrated.
3. **No Cognitive Predictive Claims**: IndicLearn does not claim to predict exam grades or human cognitive retention until validated against real educational trials.
