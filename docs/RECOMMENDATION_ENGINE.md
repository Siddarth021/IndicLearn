# Personalized Recommendation & Remediation Planner (Phase 7B)

## 1. Overview & Objective

The **Personalized Recommendation & Remediation Planner** subsystem answers the central pedagogical planning question:

> *"Given the student's current evidence-based knowledge state and the canonical academic knowledge graph, what should the student study, revise, or practice next, and why?"*

### Core Research Principle: Evidence $\to$ Uncertainty $\to$ Recommendation $\to$ Reason

The system adheres strictly to an evidence-grounded communicative policy. It never issues blunt, unsubstantiated deficit declarations such as:
> *"You are weak in Convolution, so study Convolution."*

Instead, it transparently communicates:
> *"Your recent assessment evidence suggests lower demonstrated performance on Convolution (score: 0.25, confidence: 0.85). Reviewing the source material [CHUNK_SS_001] before another assessment may be useful to reinforce foundational understanding."*

### Explicit Scientific Disclaimers
1. **No Cognitive Mind-Reading**: The system models observable assessment and interaction evidence. It does NOT claim to measure true cognitive state or latent brain capabilities.
2. **Proposed Initial Heuristics**: All prioritizer weights, decay functions, and thresholds are explicitly labeled as **proposed initial heuristics**. They are not claimed to be psychometrically calibrated or mathematically optimal.
3. **Controlled Empirical Metrics**: System performance is validated against software engineering invariants, deterministic reproducibility, provenance integrity, and student isolation (EXP-008), not empirical student exam gains.

---

## 2. Architectural Separation

The recommendation engine strictly preserves the system's core architectural boundaries:

```text
Academic Knowledge Graph (Canonical, Immutable)
       │ (concepts, relations, prerequisites)
       ▼
Student Knowledge Model (Student × Concept dynamic states)
       │ (knowledge_score, confidence, mastery_status, evidence_history)
       ▼
Recommendation Candidate Generation (HeuristicRecommendationCandidateGenerator)
       │
       ▼
Multi-Signal Prioritization (MultiSignalRecommendationPrioritizer)
       │
       ▼
Pedagogical Action Selection (PedagogicalActionSelector)
       │
       ▼
Explanation Formatting & Source Grounding (RecommendationExplanationGenerator)
       │
       ▼
Personalized Recommendation List
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
Study / Revision Action              TAKE_ASSESSMENT Action
(RAG chunk retrieval)                           │
                                                ▼
                                    Adaptive Assessment Service (Phase 7A)
                                                │ (deterministic item selection)
                                                ▼
                                        AssessmentResponse
                                                │
                                                ▼
                                          EvidenceEvent
                                                │
                                                ▼
                                    KnowledgeStateService (Phase 6)
                                                │ (evidence-weighted update + recency decay)
                                                ▼
                                      Updated Knowledge State
```

### Invariants Maintained:
- **No Direct Mutation**: The recommendation engine **never** mutates knowledge scores, confidence values, or mastery statuses.
- **Closed Loop via Evidence**: When a recommended `TAKE_ASSESSMENT` action is completed, it emits an `EvidenceEvent` to `KnowledgeStateService`, updating learner state through the standard evidence-weighted knowledge estimation layer.
- **Assessment Delegation**: When recommending an assessment, the service delegates item selection to `AdaptiveAssessmentService` (Phase 7A), preventing duplicate question bank indexing or grading logic.

---

## 3. Action Taxonomy

The system defines five explicit, supported pedagogical actions (`ActionType`):

| Action Type | Description | Pedagogical Trigger |
|:---|:---|:---|
| `REVIEW_CONCEPT` | Initial study or refresher on curriculum text | Cold-start concept ($C < 0.30$) or low-confidence uncertainty |
| `REVIEW_PREREQUISITE` | Foundational study of upstream prerequisite concepts | Concept has unmastered direct prerequisites in the academic graph |
| `PRACTICE_CONCEPT` | Active problem solving / exercise practice | Demonstrated weakness ($S < 0.40, C \ge 0.40$) or developing skill |
| `TAKE_ASSESSMENT` | Diagnostic quiz to clarify mastery | Conflicting evidence (`MasteryStatus.UNCERTAIN`) or assessment gap |
| `REVISIT_SOURCE` | Periodic spaced retrieval of mastered material | Mastered concept ($S \ge 0.80, C \ge 0.60$) with elevated review need |

---

## 4. Structured Reason Codes

Recommendations expose auditable categorical reason codes (`ReasonCode`) alongside natural-language explanations:

- `LOW_KNOWLEDGE`: Evaluated score indicates difficulty supported by strong assessment evidence.
- `LOW_CONFIDENCE`: Evidence volume is nascent; reflects prior belief rather than confirmed deficit.
- `UNCERTAIN`: Conflicting assessment outcomes detected; diagnostic clarification needed.
- `PREREQUISITE_DEFICIT`: Foundational prerequisites in academic graph lack demonstrated mastery.
- `DOWNSTREAM_BLOCKER`: Non-mastered concept is a prerequisite for one or more downstream curriculum topics.
- `NEGLECTED_REVIEW`: Significant time elapsed since last interaction or assessment.
- `ASSESSMENT_GAP`: Concept has zero or few validated assessment events.
- `MASTERED_SPACED_REVIEW`: Mastered concept eligible for periodic retention reinforcement.
- `SOURCE_AVAILABLE`: Associated curriculum chunks successfully identified and preserved.
- `SOURCE_UNAVAILABLE`: Concept lacks associated source material in the uploaded curriculum (explicitly flagged without hallucination).

---

## 5. Multi-Signal Prioritization Formulation

The priority of a concept candidate is computed across seven normalized dimensions in $[0.0, 1.0]$:

$$\text{Raw Priority} = w_w \cdot W + w_u \cdot U + w_p \cdot P_{\text{prereq}} + w_d \cdot D_{\text{block}} + w_r \cdot R_{\text{need}} + w_a \cdot A_{\text{gap}} - w_{\text{recent}} \cdot P_{\text{recent}}$$

If the concept is `MASTERED`, the raw priority is multiplied by a damping factor:
$$\text{Priority} = \begin{cases} \text{clamp}(\text{Raw Priority} \times \delta_{\text{mastered}}, 0.0, 1.0) & \text{if } \text{status} = \text{MASTERED} \\ \text{clamp}(\text{Raw Priority}, 0.0, 1.0) & \text{otherwise} \end{cases}$$

### Component Signal Definitions:

1. **Demonstrated Weakness ($W$)**:
   $$W = \begin{cases} (1.0 - S) \cdot (0.5 + 0.5 \cdot C) & \text{if } C \ge 0.30 \text{ or } \text{strong\_count} > 0 \\ 0.0 & \text{if cold-start / unassessed} \end{cases}$$
   *Crucial Property*: Cold-start concepts without assessment evidence receive $W = 0.0$, preventing unearned weakness labeling.

2. **Uncertainty Signal ($U$)**:
   $$U = \begin{cases} 1.0 & \text{if } \text{status} = \text{UNCERTAIN} \\ 0.50 \cdot (1.0 - C) & \text{if } \text{evidence\_count} == 0 \\ 1.0 - C & \text{otherwise} \end{cases}$$

3. **Prerequisite Deficit ($P_{\text{prereq}}$)**:
   $$P_{\text{prereq}} = \min\left(1.0, \frac{N_{\text{unmastered\_prerequisites}}}{2.0}\right)$$

4. **Downstream Blocker ($D_{\text{block}}$)**:
   $$D_{\text{block}} = \begin{cases} 0.0 & \text{if } \text{status} = \text{MASTERED} \\ \min\left(1.0, \frac{N_{\text{blocked\_dependents}}}{3.0}\right) & \text{otherwise} \end{cases}$$

5. **Review Need ($R_{\text{need}}$)**:
   $$R_{\text{need}} = \begin{cases} 0.0 & \text{if } \text{evidence\_count} == 0 \text{ (unintroduced)} \\ 0.50 \cdot (1 - e^{-\Delta t / 72}) & \text{if } \text{status} = \text{MASTERED} \text{ (spaced review)} \\ 1 - e^{-\Delta t / 72} & \text{otherwise} \end{cases}$$

6. **Assessment Gap ($A_{\text{gap}}$)**:
   $$A_{\text{gap}} = \begin{cases} 0.80 & \text{if } \text{strong\_count} == 0 \text{ and } \text{weak\_count} > 0 \\ 0.30 & \text{if cold-start} \\ 0.40 & \text{if } \text{strong\_count} == 1 \\ 0.0 & \text{if } \text{strong\_count} \ge 2 \end{cases}$$

7. **Recent Action Penalty ($P_{\text{recent}}$)**:
   $$P_{\text{recent}} = e^{-\Delta t_{\text{action}} / 30.0}$$

### Configurable Heuristic Default Weights:

| Parameter | Proposed Default | Scientific Status | Rationale |
|:---|:---:|:---:|:---|
| `weakness` ($w_w$) | 0.35 | Heuristic | Primary driver for targeted remediation |
| `uncertainty` ($w_u$) | 0.15 | Heuristic | Balances diagnostic exploration against remediation |
| `prerequisite_deficit` ($w_p$) | 0.15 | Heuristic | Prioritizes foundational unblocking |
| `downstream_blocker` ($w_d$) | 0.15 | Heuristic | Prioritizes high-degree foundational concepts |
| `review_need` ($w_r$) | 0.10 | Heuristic | Encourages spaced retention check |
| `assessment_gap` ($w_a$) | 0.10 | Heuristic | Surfaces unmeasured topics |
| `recent_action_penalty` ($w_{\text{recent}}$) | 0.25 | Heuristic | Prevents repetitive single-concept recommendations |
| `mastered_damping_factor` ($\delta_{\text{mastered}}$) | 0.35 | Heuristic | Keeps mastered topics subordinate to active needs |

---

## 6. Action Selection & Difficulty Policy

Action selection is decoupled from priority scoring:

```text
Candidate State
       │
       ├─ Unmastered Prerequisites > 0? ──► Action: REVIEW_PREREQUISITE
       │
       ├─ Status == UNCERTAIN? ───────────► Action: TAKE_ASSESSMENT (MEDIUM)
       │
       ├─ Status == MASTERED? ────────────► Action: REVISIT_SOURCE
       │
       ├─ Confidence >= 0.40?
       │     ├─ Score < 0.40 ─────────────► Action: PRACTICE_CONCEPT (EASY)
       │     ├─ 0.40 <= Score < 0.75 ─────► Action: PRACTICE_CONCEPT (MEDIUM)
       │     └─ Score >= 0.75 ────────────► Action: PRACTICE_CONCEPT (HARD)
       │
       └─ Cold-Start / Nascent ───────────► Action: REVIEW_CONCEPT
```

---

## 7. Source Grounding & Provenance Safety

Every recommendation inspects the canonical `Concept.source_chunk_ids`:
- **Grounded Concepts**: `source_available = True`, retains exact chunk IDs (e.g. `["CHUNK_SS_001", "CHUNK_SS_004"]`), assigns `ReasonCode.SOURCE_AVAILABLE`, and embeds citation references in explanations.
- **Unindexed Concepts**: If a concept has no source chunks, the system sets `source_available = False`, assigns `ReasonCode.SOURCE_UNAVAILABLE`, and emits: *"Source material for this concept is not currently indexed in the academic curriculum."* The system **never** fabricates fake citations or page numbers.

---

## 8. Experimental Verification: EXP-008

EXP-008 verifies 12 controlled scenarios and 33 software invariants with 100% compliance:

| Scenario | Focus | Invariants Evaluated | Result |
|:---|:---|:---:|:---:|
| `SCEN_001` | High-confidence weakness triggers practice remediation | 3 | PASS |
| `SCEN_002` | Low-confidence uncertainty distinguished from weakness | 3 | PASS |
| `SCEN_003` | Prerequisite deficit triggers `REVIEW_PREREQUISITE` | 3 | PASS |
| `SCEN_004` | Downstream blocker score scales with curriculum impact | 1 | PASS |
| `SCEN_005` | Recent-action penalty damps repeated concept recommendations | 2 | PASS |
| `SCEN_006` | Mastered concepts deprioritized below unresolved topics | 3 | PASS |
| `SCEN_007` | Conflicting evidence (`UNCERTAIN`) triggers diagnostic action | 2 | PASS |
| `SCEN_008` | Review recommendations ground in curriculum chunk provenance | 3 | PASS |
| `SCEN_009` | Unindexed concepts explicitly flag `SOURCE_UNAVAILABLE` | 4 | PASS |
| `SCEN_010` | `TAKE_ASSESSMENT` delegates to Adaptive Assessment Service | 3 | PASS |
| `SCEN_011` | Strict multi-student recommendation isolation | 3 | PASS |
| `SCEN_012` | Deterministic ranking, priority scores, and explanations | 3 | PASS |

**Overall EXP-008 Compliance**: **33 / 33 Invariants Passed (100.0%)**.
