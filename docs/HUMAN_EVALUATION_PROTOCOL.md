# Human Evaluation Protocol for IndicLearn (Phase 9)

**Document Version:** 1.0  
**Phase:** Phase 9 — Human Validation Framework  
**Project:** A Dynamic Personalized Knowledge Map for Engineering Students using Indic Language NLP  

---

> [!CAUTION]
> **Ethical Review Notice & Regulatory Disclaimer:**  
> This protocol describes the methodological instruments, participant tasks, and annotation procedures prepared for future research evaluation. **This software repository does not possess institutional ethical approval.** Before administering this protocol to real students, the research team must obtain formal Institutional Review Board (IRB) or human-subject ethical clearance.

---

## 1. Study Objectives

This human evaluation protocol evaluates:
1. **Interface Usability (EXP-012):** Can engineering students find concepts, understand latent knowledge estimates, and interpret recommendations transparently?
2. **Attribution & RAG Quality (EXP-013):** How do bilingual domain experts rate factual correctness, grounding, and citation accuracy of generated answers?
3. **Multilingual Comprehension (EXP-014):** How do engineering students evaluate pedagogical clarity across English, Telugu native script, Romanized Telugu, and code-mixed formulations?

---

## 2. Participant Eligibility Criteria

### 2.1 Inclusion Criteria
To ensure evaluation validity, recruited participants must satisfy:
1. **Academic Enrollment:** Currently enrolled in or graduated within 24 months from an accredited Bachelor of Technology (B.Tech) or Bachelor of Engineering (B.E.) program in Computer Science & Engineering (CSE), Electronics & Communication Engineering (ECE), Electrical & Electronics Engineering (EEE), or Information Technology (IT).
2. **Curriculum Exposure:** Must have completed or be currently enrolled in introductory coursework in *Signals and Systems*, *Computer Networks*, or closely related mathematical engineering subjects.
3. **Language Background:** Self-reported working fluency in academic English and bilingual familiarity with Telugu (able to read conversational/academic Telugu in native script, Latin script transliteration, or code-mixed conversation).
4. **Consent:** Provision of informed consent via pseudonymous digital signature.

### 2.2 Exclusion Criteria
Candidates are excluded if:
1. They are directly affiliated with the design, development, or code authoring of the IndicLearn system (to prevent developer bias).
2. They have no prior exposure to computer science or engineering coursework.
3. They are unable or unwilling to complete tasks without automated external translation aids.

---

## 3. Pseudonymization and Privacy Safeguards

The protocol strictly forbids collecting Personally Identifiable Information (PII):
- **Forbidden Fields:** Legal name, university roll number, email address, phone number, Aadhaar number, IP address, and residential address.
- **Participant ID (`participant_id`):** Alphanumeric pseudonyms (e.g., `PARTICIPANT-7C81`) assigned at intake.
- **Annotator ID (`annotator_id`):** Coded identifiers (e.g., `ANN-01`, `ANN-02`) used in double-blind annotation sets.
- **Data Persistence:** Interaction logs, survey responses, and time-stamped metrics link exclusively to pseudonymous IDs.

---

## 4. Evaluator Tasks Protocol (T1 – T9)

Participants perform 9 structured operational tasks in a standardized environment:

| Task ID | Task Title | Operational Instruction | Target Success Criteria | Max Time |
| :--- | :--- | :--- | :--- | :--- |
| **T1** | English Query | Ask an academic question regarding Convolution in English. | System returns answer with citations; user reviews citations. | 60s |
| **T2** | Telugu Script Query | Ask a question regarding LTI systems in Telugu native script. | Query accepted; Telugu explanation with citations inspected. | 90s |
| **T3** | Romanized Telugu Query | Ask a question using Romanized Telugu (e.g., `"Convolution ante enti?"`). | Query normalized; relevant academic concepts retrieved. | 90s |
| **T4** | Code-Mixed Query | Ask a code-mixed question with English nouns and Telugu postpositions. | Semantic intent correctly parsed; concept identified. | 90s |
| **T5** | Inspect Concept Detail | Locate and click on the concept node for *Nyquist Rate*. | Concept detail drawer opens with description and dependencies. | 45s |
| **T6** | Interpret Knowledge State | Explain what the displayed *Estimated knowledge* and *Evidence confidence* signify. | Student correctly recognizes probabilistic estimate vs absolute cognition. | 60s |
| **T7** | Follow Recommendation | Inspect the top recommendation card and view its rationale explanation. | Student articulates why the action was recommended. | 60s |
| **T8** | Complete Assessment Item | Answer an adaptively presented multiple-choice question. | Deterministic feedback displayed; knowledge state refreshes. | 90s |
| **T9** | Inspect Source Evidence | Click the evidence drawer to trace a concept back to the source textbook chunk. | Student identifies chapter chunk supporting the concept. | 60s |

---

## 5. RAG Expert Annotation Protocol (EXP-013)

Independent expert annotators review generated answers side-by-side with retrieved textbook evidence chunks.

### 5.1 Rating Dimensions (1 to 5 Ordinal Scale)
1. **Factual Correctness:** 1 (Completely incorrect) to 5 (Fully factually correct).
2. **Evidence Support (Grounding):** 1 (Completely unsupported hallucination) to 5 (Fully supported by provided chunk).
3. **Citation Correctness:** 1 (Citations point to irrelevant/fabricated sources) to 5 (Citations perfectly identify supporting passages).
4. **Pedagogical Usefulness:** 1 (Counter-productive / misleading for learners) to 5 (Exceptionally instructive and pedagogically sound).
5. **Clarity & Structure:** 1 (Incoherent / confusing syntax) to 5 (Exemplary clarity and organization).
6. **Language Quality:** 1 (Machine translation artifacts / broken grammar) to 5 (Authentic, natural phrasing).
7. **Code-Mixed Comprehension:** 1 (Total semantic breakdown on code-mixing) to 5 (Flawless capture of technical English nouns and Telugu morphosyntax).

### 5.2 Mandatory Annotation Guidelines
- **Rule A (Plausibility $\ne$ Grounding):** If retrieved chunks do not contain evidence for a statement, rate Evidence Support $\le 2$, even if the statement is factually true in real-world engineering.
- **Rule B (Fluency $\ne$ Correctness):** A beautifully phrased explanation containing incorrect equations must be scored low on Factual Correctness.
- **Rule C (Model Confidence $\ne$ Ground Truth):** Evaluators must not rely on system confidence indicators when evaluating correctness.

### 5.3 Double-Annotation & Adjudication
- Every query is scored independently by two annotators under blinded system identifiers (`SYSTEM_A` vs `SYSTEM_B`).
- Inter-rater agreement is computed using quadratic weighted Cohen's Kappa ($\kappa_w$).
- Discrepancies $\ge 2$ Likert points trigger an adjudication review by a senior domain chair.
- **Original annotator ratings are strictly preserved** in the dataset for inter-rater reliability reporting; adjudicated consensus ratings are stored in parallel fields.

---

## 6. Neutral Usability Questionnaire (EXP-012)

Administered immediately following task completion. Evaluated on a 5-point Likert scale (1 = Strongly Disagree, 2 = Disagree, 3 = Neutral, 4 = Agree, 5 = Strongly Agree):

1. **Q01 (Concept Findability):** *"I could find the concept I was looking for."*
2. **Q02 (Recommendation Transparency):** *"I understood why a recommendation was shown."*
3. **Q03 (Estimate Comprehension):** *"I understood what 'Estimated knowledge' meant."*
4. **Q04 (Evidence Traceability):** *"I could identify the source supporting an answer or status."*
5. **Q05 (Map Navigation Ease):** *"I could navigate the knowledge map (dependencies and prerequisites)."*
6. **Q06 (Language Naturalness):** *"The language used across explanations and UI labels was understandable."*
