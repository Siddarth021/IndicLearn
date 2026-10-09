# Human Evaluation & Validation Framework Directory

This directory contains schemas, protocols, annotation guidelines, and evaluation templates for human studies of the IndicLearn platform.

## Ethical Policy & Research Safeguards

1. **No Fabricated Data:** Absolutely no participant identities, Likert ratings, task completion logs, or inter-annotator agreement statistics may be invented.
2. **Current Evaluation Status:**
   ```text
   Human Evaluation: NOT YET CONDUCTED (Framework & Infrastructure Ready)
   ```
3. **Institutional Review:** This software repository and protocol do not constitute ethical or institutional review board (IRB) approval. Real participant recruitment requires prior formal institutional approval.
4. **Privacy & Pseudonymization:** No personally identifying information (PII) such as phone numbers, government IDs, home addresses, or real names may be stored. All records must use pseudonymous `participant_id` (e.g. `P_001`, `P_002`) or `annotator_id` (e.g. `ANN_01`, `ANN_02`).

## Contents

* `annotation_guidelines.md`: Strict instructions for human annotators grading RAG answers, recommendations, and assessment items.
* `annotator_schema.json`: JSON Schema for annotator demographic profile and background.
* `participant_schema.json`: JSON Schema for student participant records.
* `consent_template.md`: Informed consent document template for human evaluation participants.
* `rag_annotation_template.json`: Template for human scoring of RAG answers across 7 dimensions (1–5 Likert).
* `recommendation_annotation_template.json`: Template for grading pedagogical recommendation relevance and action appropriateness.
* `assessment_annotation_template.json`: Template for grading assessment item clarity and difficulty alignment.
* `usability_questionnaire.json`: Neutral 6-item usability questionnaire with Likert (1–5) scale.
* `example_annotation.json`: Illustrative structural example (**EXPLICITLY LABELED AS EXAMPLE / NOT REAL PARTICIPANT DATA**).
