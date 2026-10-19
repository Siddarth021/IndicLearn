# IndicLearn Studio

**Learn engineering from your own notes, in your own language.** Upload a chapter, slides or a paper →
read it in the app → ask doubts in **English, తెలుగు, Romanized Telugu ("convolution ante enti?"), हिन्दी**
or by **voice** → get a step-by-step explanation in your language → take quizzes → watch a **mind map** of the
subject grow with what you learn.

## Run it

```bash
run.bat                       # Windows: creates .venv, installs requirements, opens http://127.0.0.1:8000
# or
pip install -r requirements.txt && pip install -r requirements-models.txt
PYTHONPATH=src python -m indiclearn.studio
```

Copy `.env.example` to `.env` and add keys (all optional). Check every engine with `check_ai.bat`.

## How it decides who answers

Every AI feature walks the same chain and never fails hard - if one engine is down it moves on:

| Order | Engine | Cost | Used for |
|---|---|---|---|
| 1 | **Gemini** (`gemini-3.8-flash`, `3.5-flash`, `flash-latest`, `2.5-flash`) | free tier, **~20 requests/day per model** - rotated | tutor, map, quizzes, translation |
| 2 | **Sarvam** `sarvam-105b` (+ Saaras speech-to-text, Bulbul voices) | free credits, then pay-as-you-go | same + **voice in / voice out** |
| 3 | **Gemma 4 E2B** via Ollama, on your PC | free, offline | same (map is upgraded in the background) |
| 4 | Trained **M1-M4** + rules | free, offline, CPU | map filter, definitions, prerequisite links, quizzes |

Students see answers **stream in word by word** (first words in ~1.5-3 s). Plain English and Telugu-script
questions skip the extra "understand" call, which halves the free-tier usage.

## What a student gets

* **Tutor** - grounded on the notes with page chips, answers in the language of the question (explicit
  "telugu lo cheppu" / "in Hindi" wins), English technical terms kept in English, reasoning steps, key terms,
  a quick-check MCQ and tap-to-ask follow-ups. 🌐 Convert any reply to 13 languages. 🔊 Listen (natural voice).
* **🎙 Voice doubts** - the server recogniser understands Telugu, English and the mix, is given the document's
  own terms, and stops by itself when the student stops talking.
* **Quizzes** - understanding-level MCQs written in the background right after upload (instant to open);
  every answer key is double-checked by a second blind pass, wrong-keyed questions are dropped.
* **Mind map** - AI concept map (or rules + M1/M2/M3 offline), labels never clipped, progress colours.
* **OS course** - 38 lessons (one per syllabus topic) in English, Telugu on request (translated once and cached):
  explanation, key terms, exam tips, common mistakes, a live exactly-solved example for numerical topics
  ("New example"), the solver's exact rules, a quick check, listen-aloud, "mark as learned" and a jump to a paper.
* **Exam papers (Operating Systems)** - the student sets the skeleton (total marks, time, units, rows such as
  "5 x 1 mark MCQ" or "2 x 4 marks numerical"; Mid-sem / End-sem / Unit-test presets) and gets a printable question
  paper. Answers stay hidden until asked for (per question, all at once, or as a separate answer-key PDF).
  Topics = the ticked syllabus units that the student's uploaded OS notes also cover. Numericals (CPU scheduling,
  page replacement, Banker's, disk scheduling, fits, paging, EAT) are computed exactly; theory comes from a
  500-question bank checked by a second model, plus fresh questions written from the student's own notes.
  **GATE-style** papers too (MCQ with -1/3 negative marks, MSQ, NAT - numerical MCQs/MSQs/NATs are computed exactly),
  and **"Match my college paper"**: upload a past paper (PDF, Word or a phone photo) and its marks pattern, time and
  units are copied into the builder.
* **Quiz review** - every answer reviewed at the end, "Retry the ones I got wrong" (options reshuffled); maths rendered with KaTeX.
* Works on phones (tutor opens as a closable full-screen sheet).

## Models

| | Model | Where it comes from |
|---|---|---|
| M1 concept filter | XLM-RoBERTa-base, fine-tuned | `training/` (teacher-labelled B.Tech corpus) |
| M2 description writer | Flan-T5-base, fine-tuned | `training/` |
| M3 prerequisite / related links | XLM-RoBERTa-base, fine-tuned | `training/` + LectureBank 2 |
| M4 quiz generator | Flan-T5-base, fine-tuned | `training/` + SciQ |
| Offline tutor | Gemma 4 E2B (Ollama) | `setup_offline.bat` |
| Search / speech / translation / voices | multilingual-e5-small, Telugu Whisper, IndicTrans2, MMS-TTS | `python scripts/download_models.py` |

Install the trained models: download the zip from the private Kaggle dataset
`gundetisiddarth/indiclearn-models`, then `python scripts/install_models.py <zip>`.
How they were built and how well they do: **`training/README.md`** and `training/reports/`.

## Layout

```
src/indiclearn/studio/   the app (FastAPI): brain.py (engines), tutor.py, quizbank.py, speech.py, kgraph*.py, neural.py
  courses/               subject courses: os_syllabus.json, os_bank.jsonl (question bank), oscalc.py (exact numericals),
                         papers.py (blueprint -> paper), exams.py (service), pdf.py (printable papers), qgen.py
frontend/                the web UI (plain ES modules + d3)
training/                corpus -> teacher labels -> datasets -> Kaggle training -> reports
scripts/                 download_models.py, install_models.py, setup_offline.py, check_ai.py
models/                  pretrained/ (downloaded) and trained/ (M1-M4)  - not in git
tests/studio/            app tests:  PYTHONPATH=src pytest tests/studio
old_useful_trash/        parked old files (not deleted, not in git)
```

Licence notes: MMS-TTS voices and NLLB are CC-BY-NC (non-commercial); Wikipedia-derived data is CC BY-SA.
