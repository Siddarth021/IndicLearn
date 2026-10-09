# IndicLearn — Research Evaluation & Human Validation Framework (Phase 9)

**Document Version:** 1.0  
**Phase:** Phase 9 — Research Evaluation & Human Validation Framework  
**Project:** A Dynamic Personalized Knowledge Map for Engineering Students using Indic Language NLP  

---

## 1. Executive Summary

Phase 9 establishes a reproducible, scientifically rigorous evaluation framework for **IndicLearn**. Rather than introducing speculative intelligence features or making uncalibrated claims, Phase 9 formulates formal research questions, provides controlled algorithmic benchmark runners (EXP-010, EXP-011), establishes standardized human validation instruments (EXP-012, EXP-013, EXP-014), and enforces strict anti-fabrication policies across the research lifecycle.

```text
Software Engineering Correctness (EXP-006 to EXP-009)
                      ↓
Controlled Algorithmic Benchmarks (EXP-010, EXP-011)
                      ↓
Human Usability & Linguistic Evaluation (EXP-012, EXP-013, EXP-014)
                      ↓
Controlled Learning-Effectiveness Clinical Trials (Future Planned Protocol)
```

---

## 2. Epistemic Taxonomy & Boundary Conditions

To maintain scientific integrity, IndicLearn enforces a strict distinction among four epistemic levels:

| Evaluation Level | Purpose | Baseline Instruments | Data Requirements | Scientific Claim Permitted |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1: Software Verification** | Validates API contracts, memory isolation, determinism, and data schemas | `pytest`, EXP-009 | Synthetic simulated states | Software correctness, API compliance, and fault tolerance |
| **Level 2: Controlled Algorithmic Benchmarks** | Evaluates multi-signal prioritization, ranking consistency, and rule adherence | EXP-010, EXP-011 | Synthetic knowledge states, curated question bank | Algorithmic consistency, rule adherence, and heuristic stability |
| **Level 3: Human Usability & Linguistic Evaluation** | Measures task completion, interface transparency, transliteration clarity, and grounding | EXP-012, EXP-013, EXP-014 | Real bilingual engineering participants under IRB approval | Subjective clarity, navigational usability, annotator agreement |
| **Level 4: Controlled Learning Effectiveness** | Measures academic achievement gains, cognitive retention, and learning acceleration | Pre/Post Controlled Trial (Protocol only) | Randomized cohorts with active control group | Causal educational improvement, knowledge acquisition |

> [!CAUTION]
> **Strict Anti-Fabrication Rule:**  
> Level 1 and Level 2 evaluations **never** permit claims of learning effectiveness, cognitive-load reduction, or exam performance improvement. If real human participants are not recruited, Level 3 and Level 4 evaluations must report `status: HUMAN_DATA_REQUIRED` or `NOT_RUN`.

---

## 3. Formal Research Questions (RQs)

### RQ1 — Multilingual Retrieval Performance
> **Question:** How accurately and reliably does IndicLearn retrieve relevant academic course evidence across standard English, Telugu native script, Romanized Telugu (Latin script), and Telugu-English code-mixed formulations?  
> **Metrics:** Recall@1, Recall@3, Recall@5, MRR, Gold Recall@5, Multi-Chunk Recall.  
> **Baseline (EXP-005A):** R@5 = 0.9200, MRR = 0.5677, Gold Recall@5 = 0.9100 across 50 answerable queries. Code-mixed (te-en) R@5 = 1.0000; native script (te) R@5 = 0.6667 (highlighting cross-script tokenization as a known limitation).

### RQ2 — Scope Control, Safety, and Boundary Enforcement
> **Question:** Can IndicLearn reliably discriminate in-scope engineering syllabus concepts from out-of-scope, off-topic, or ambiguous student prompts without hallucinating academic answers?  
> **Metrics:** Scope Accuracy, Macro Precision, Macro Recall, Macro F1, False Acceptance Rate (FAR), Confusion Matrix.  
> **Baseline (EXP-005B):** Scope Accuracy = 1.0000, FAR = 0.0000 across 45 queries (18 in-scope, 18 out-of-scope, 9 ambiguous). Zero false acceptances observed.

### RQ3 — Grounded RAG Generation & Citation Integrity
> **Question:** When an academic question is answerable, are generated explanations strictly grounded in retrieved evidence chunks, and do in-text citations accurately reflect course text without unsupported claims?  
> **Metrics:** Citation Validity Rate, Citation Coverage Rate, Unsupported Claim Rate, Groundedness Ratio.  
> **Baseline (EXP-005A):** Citation Validity = 1.0000, Citation Coverage = 1.0000, Unsupported Claims = 0.0000 on controlled grounding checks.

### RQ4 — Pedagogical Recommendation Rule Consistency
> **Question:** Does the Phase 7B recommendation engine prioritize unmastered prerequisites, penalize repetitive recent recommendations, and maintain student isolation across simulated student profiles?  
> **Metrics:** Scenario Pass Rate, Top-1 Recommendation Accuracy, Action Selection Accuracy, Reason Code Accuracy, Ranking Consistency.  
> **Benchmark (EXP-010):** 10/10 scenarios passed (100.0%), Top-1 Accuracy = 1.0000, Action Selection = 1.0000, Reason Code Accuracy = 1.0000, Ranking Consistency = 1.0000.

### RQ5 — Adaptive Assessment Selection & Closed-Loop Integration
> **Question:** Does the Phase 7A adaptive assessment engine calibrate item difficulty to latent student mastery, target concept weaknesses, avoid immediate question repetition, and update knowledge states in a closed loop?  
> **Metrics:** Difficulty Alignment Rate, Weakness Targeting Accuracy, Provenance Integrity, Repeat Avoidance Rate, Closed-Loop Evidence Rate.  
> **Benchmark (EXP-011):** 10/10 scenarios passed (100.0%), Difficulty Alignment = 1.0000, Weakness Targeting = 1.0000, Repeat Avoidance = 1.0000, Closed-Loop Evidence = 1.0000.

---

## 4. Experiment Registry Summary (EXP-001 to EXP-014)

| ID | Title | Level | Target Focus | Status | Result File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-001** | Ingestion Baseline | Level 1 | Chunking & ingestion | COMPLETED | `exp001_results.json` |
| **EXP-002** | Retrieval Baseline | Level 1 | BM25 vs Dense vs Hybrid | COMPLETED | `exp002_results.json` |
| **EXP-003** | Academic Knowledge Graph | Level 1 | Concept extraction & graph build | COMPLETED | `exp003_results.json` |
| **EXP-004** | Indic NLP Benchmark | Level 1 | Tokenization & transliteration | COMPLETED | `exp004_results.json` |
| **EXP-005A** | Research-Hardened RAG | Level 1 | Multilingual retrieval & grounding | COMPLETED | `exp005a_results.json` |
| **EXP-005B** | RAG Safety & Scope Control | Level 1 | Out-of-scope rejection & FAR | COMPLETED | `exp005b_results.json` |
| **EXP-006** | Student Knowledge Model | Level 1 | Evidence weighting & confidence | COMPLETED | `exp006_results.json` |
| **EXP-007** | Adaptive Assessment Core | Level 1 | Candidate selection & difficulty | COMPLETED | `exp007_results.json` |
| **EXP-008** | Recommendation Planner | Level 1 | Multi-signal prioritization & rules | COMPLETED | `exp008_results.json` |
| **EXP-009** | Student Learning Dashboard | Level 1 | Interface invariants & isolation | COMPLETED | `exp009_results.json` |
| **EXP-010** | Recommendation Rule Consistency | Level 2 | 10 controlled recommendation tests | **COMPLETED** | `exp010_results.json` |
| **EXP-011** | Adaptive Assessment Benchmark | Level 2 | 10 controlled assessment tests | **COMPLETED** | `exp011_results.json` |
| **EXP-012** | Dashboard Usability Protocol | Level 3 | Tasks T1-T5, Likert Q1-Q6 | **NOT_RUN** | `exp012_results.json` |
| **EXP-013** | RAG Human Annotation | Level 3 | 7-dimension double annotation | **HUMAN_DATA_REQUIRED** | `exp013_results.json` |
| **EXP-014** | Multilingual Human Evaluation | Level 3 | Disaggregated en, te, te-Latn, te-en | **HUMAN_DATA_REQUIRED** | `exp014_results.json` |

---

## 5. Statistical Framework Specification

All statistical evaluations in Phase 9 utilize dedicated, tested utilities in `src/indiclearn/evaluation/statistics/`:

1. **Descriptive Statistics (`descriptive.py`)**:
   - Sample size ($N$), Sample Mean ($\bar{x}$), Median ($\tilde{x}$), Sample Variance ($s^2$ with Bessel's correction $N-1$), Sample Standard Deviation ($s$), Interquartile Range ($IQR = Q_3 - Q_1$), Minimum, Maximum, and discrete percentage distributions.
2. **Inter-Rater Reliability (`agreement.py`)**:
   - **Percentage Agreement**: Proportion of exact ratings matches.
   - **Unweighted Cohen's Kappa ($\kappa$)**: Categorical agreement corrected for chance.
   - **Weighted Cohen's Kappa ($\kappa_w$)**: Linear and quadratic weighting for ordinal Likert scales (1 to 5), penalizing larger rating discrepancies more severely.
   - **Fleiss' Kappa**: Multi-rater agreement ($K > 2$ raters across $N$ subjects).
   - **Landis & Koch (1977) Benchmarks**: Standardized verbal interpretation brackets.
3. **Hypothesis Testing & Effect Size (`significance.py`)**:
   - **Student's $t$ Confidence Intervals**: Parametric interval estimation for continuous metrics.
   - **Percentile Bootstrap CI**: Non-parametric resampling ($B=1000$ iterations) with fixed random seeds for non-normal metrics.
   - **Cohen's $d$**: Standardized effect size with pooled standard deviation.
   - **Mann-Whitney $U$ Test**: Two-sided rank-sum test for independent ordinal/continuous distributions.
   - **Wilcoxon Signed-Rank Test**: Two-sided paired rank test for pre/post or repeated-measures comparisons.
   - **Holm-Bonferroni Correction**: Sequential step-down adjustment controlling Family-Wise Error Rate (FWER) without Bonferroni over-conservatism.

---

## 6. Research Limitations & Governance Notice

1. **No Learning Effectiveness Claim:** No real student clinical trial has yet been conducted. The system does not claim to improve student exam grades, reduce cognitive load, or accelerate learning.
2. **Heuristic Calibration:** The student knowledge model confidence updates, recency decay half-lives, and recommendation signal weights are transparent research heuristics; they are not calibrated Item Response Theory (IRT) or Bayesian Knowledge Tracing parameters.
3. **Language Diversity:** Romanized Telugu exhibits substantial spelling and transliteration variance across regional dialects; automated transliteration may degrade on informal phonetic spellings.
4. **Institutional Review:** The software evaluation templates, consent forms, and schemas do not constitute IRB approval. Formal ethical review must precede live data collection.
