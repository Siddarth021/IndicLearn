---
language:
- te
- en
license: cc-by-4.0
task_categories:
- translation
tags:
- telugu
- code-mixed
- code-switching
- education
- synthetic
- distillation
pretty_name: IndicLearn Telugu Lecturer-Register Technical Corpus
size_categories:
- 1K<n<10K
---

# IndicLearn Telugu Lecturer-Register Technical Corpus

Parallel **English → code-mixed Telugu** pairs for computer-science / engineering study
material: definitions, explanations and multiple-choice questions written in the register
Telugu-medium lecturers actually use — Telugu grammar with **English technical terms kept in
English**.

## What makes it different

Existing Telugu–English code-mixed corpora are social-media / sentiment text for language
identification. Existing technical-term-retention MT work is for Hindi, via constrained
decoding. This corpus targets the empty slot: **Telugu, technical-education register, as a
parallel training set.**

## Composition

| Split | te_style | m2_te (definitions) | m4_te (MCQs) |
|---|---|---|---|
| train | 2089 | 1283 | 806 |
| val | 157 | 98 | 59 |
| test | 349 | 199 | 150 |

~5,200 pairs total across the three tasks. Source domains: Operating Systems, Discrete
Structures & Mathematics, AI/NLP study notes.

Each row: `input` (English, prefixed `to Telugu: `), `target` (code-mixed Telugu),
`target_en` (English reference), `src` (task tag).

## How it was built (provenance — read this)

**The Telugu side is model-generated, not human-authored.** English source passages were
drawn from study notes; the code-mixed Telugu targets were produced by large language models
(teacher–student distillation) after the English content was fixed, then **auto-validated**:

- Telugu-script share within an expected band (to enforce the code-mixed register)
- alignment with the English source (no dropped/added content)
- MCQ structural well-formedness (options + answer present and consistent)

Rows failing validation were dropped. The data has **not** been human-reviewed at scale; a
human-verified subset is the recommended next step before any high-stakes use.

## Intended use & limitations

- **Use:** fine-tuning / evaluating code-mixed Telugu technical MT; the IndicLearn study app.
- **Not** a gold human-translation benchmark. Synthetic targets may carry teacher-model
  biases and occasional meaning errors on subtle logical wording.
- Telugu only; one education domain.

## Citation

```bibtex
@misc{indiclearn_te_corpus_2026,
  title  = {IndicLearn Telugu Lecturer-Register Technical Corpus},
  author = {Gundeti Siddarth},
  year   = {2026},
  note   = {~5.2K synthetic, auto-validated EN->code-mixed-TE pairs for CS education}
}
```
