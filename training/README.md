# IndicLearn — model training

This folder builds the data for, and trains, the four small offline models the app uses when no
LLM is reachable (and which make the first map instant even when one is):

| Model | Job in the app | Base | Trained on |
|---|---|---|---|
| **M1** concept filter | keeps real concepts, drops junk nodes from the rule-mined map | `google/muril-base-cased` | rule-mined candidates from 460 B.Tech articles, judged by a teacher LLM |
| **M2** description writer | one-line definition for a node the notes never define | `google/flan-t5-base` | teacher-written, student-friendly definitions (+ IndicTrans2 for Telugu/Hindi output) |
| **M3** relation classifier | turns co-occurrence links into `prereq` / `related` / dropped | `google/muril-base-cased` | teacher prerequisite/related pairs + LectureBank 2 (balanced) |
| **M4** quiz generator | a real MCQ with 3 believable wrong options | `google/flan-t5-base` | SciQ (human distractors) + teacher-written engineering MCQs |

## Pipeline (each step is resumable)

```
python training/collect_wikipedia.py      # 1. corpus: 28 B.Tech subjects x ~16 topics (topics.json), EN + TE/HI intros
python training/make_candidates.py        # 2. run the app's own rule-based map builder on every article
python training/teacher_label.py          # 3. a strong LLM labels candidates, writes definitions, links, MCQs
python training/build_datasets.py         # 4. -> datasets/m1..m4/{train,val,test}.jsonl + STATS.json
python training/kaggle_run.py train       # 5. train on a free Kaggle T4, install into models/trained/
python training/kaggle_run.py gemma       #    (benchmark the offline tutor LLM on 56 student doubts)
```

`train.py` can also run locally on a CUDA GPU: `python training/train.py --model m4`.

## Folder layout

```
training/
  topics.json            the B.Tech syllabus used to collect the corpus (edit to add subjects)
  collect_wikipedia.py   step 1          make_candidates.py  step 2       teacher_label.py  step 3
  build_datasets.py      step 4          train.py            step 5       kaggle_run.py     step 5 on Kaggle
  gemma_bench.py         offline-LLM benchmark (runs on Kaggle)
  datasets/              the four training sets + STATS.json        (small, versioned)
  reports/               test metrics + sample outputs per model    (written by training)
  data/                  raw corpus and intermediate files          (git-ignored, rebuildable)
```

## Honest notes

* Labels come from an LLM teacher (Gemini 3.x / Gemma 4 via the Gemini API), not from humans. Each
  row records which engine labelled it. Spot-check `data/interim/teacher.jsonl` before quoting numbers.
* Splits are by article, stratified by subject (no article appears in two splits). LectureBank is NLP/ML
  only, so it is used for training M3, never for its engineering test set.
* Wikipedia text is CC BY-SA 4.0; SciQ is CC BY-NC 3.0; LectureBank is for research use. Commercial
  textbooks were never used.
* Every model is compared with the rule it replaces (`reports/m*.json`): M1 vs "keep every candidate",
  M3 vs "every co-occurring pair is related", M2 vs "copy the sentence", M4 vs well-formedness.


## Operating Systems question bank (exam papers)

`python training/make_os_bank.py` writes `src/indiclearn/studio/courses/os_bank.jsonl`: for each of the 38 topics in
`courses/os_syllabus.json`, Sarvam writes fill-in-the-blank, MCQ, short, medium and long questions with model answers and
mark schemes; Gemini (a different model) then solves every MCQ blind and reviews every answer - only questions it agrees
with are kept. Resumable; report in `reports/os_bank.json` (500 questions on 2026-10-19). Numericals are not in the bank:
`courses/oscalc.py` generates them with computed answers (checked against textbook examples in `tests/studio/test_oscalc.py`).

## Operating Systems lessons

`python training/make_os_lessons.py` writes `src/indiclearn/studio/courses/os_lessons.jsonl`: one lesson per topic,
written by Sarvam (or Gemini: `WRITER=gemini`), reviewed and corrected by Gemini, then translated to Telugu (Telugu
sentences, English technical terms). `--te-only` finishes missing Telugu; the app also translates a lesson the first time
a student opens it in Telugu and caches it. For numerical topics the app replaces any model-written "conventions" with the
exact rules the solver uses (`courses/lessons.py: RULES`), so a lesson can never contradict its worked example.
