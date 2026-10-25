# IndicLearn — what was built, and why it matters

*Written for someone seeing this project for the first time. No background in NLP is assumed;
technical words are explained in the glossary at the bottom.*

Author: Gundeti Siddarth. Every number in this file comes from a report in `training/reports/`.

---

## The 30-second version

**IndicLearn** is a study app for engineering students who learn in Telugu. It runs on an ordinary
laptop with no internet.

Its main research result is a **translation model that writes Telugu the way Telugu lecturers
actually speak in class**: Telugu sentences, with the technical words (*stack*, *deadlock*, *page fault*)
kept in English. Standard translation tools do not do this. They translate every word into Telugu,
which sounds unnatural to students.

I trained that model, built the dataset it learned from, measured it against a standard tool,
and wrapped it in code that guarantees formulas are never corrupted.

---

## 1. The problem, with an example

A student reads this English sentence in an Operating Systems note:

> *A deadlock happens when two processes each wait for a resource the other holds.*

- **A standard translator** (IndicTrans2, from AI4Bharat) turns *every* word into Telugu, including
  "deadlock" and "process". Students never say those words in Telugu. They say "deadlock" and "process"
  in English.
- **A Telugu lecturer** says something like: *"రెండు processes ఒకదానికొకటి hold చేసే resource కోసం wait చేసినప్పుడు deadlock వస్తుంది."*
  Telugu grammar and connecting words, English technical terms.

Mixing two languages like this inside one sentence is called **code-mixing** (or code-switching).
The style of language used in a particular setting (here: a classroom) is called its **register**.

Existing public work does not cover this: the Telugu–English code-mixed datasets I found are
social-media text (used for tasks like detecting which language a word is in), and the research I found
on keeping technical terms untranslated is for Hindi and uses a decoding trick rather than a trained model.
So "Telugu + classroom register + a trained model + a dataset for it" is an open gap. This project fills it.

---

## 2. What was contributed

### Contribution 1 — A translator that produces classroom-style Telugu
- **What it is:** Google's `mT5-base` language model (582 million parameters) that I **fine-tuned**
  (i.e. continued training on my own examples) so that it translates English technical text into code-mixed Telugu.
- **Where it is:** `models/trained/te_style_translator/` (this folder also has a model card, `README.md`).
- **Result:** score **73.85** on a test set of 349 sentences it never saw during training (the score is
  explained in section 3).

### Contribution 2 — The dataset it learned from
- **What it is:** about 5,200 pairs of "English sentence → classroom-style Telugu sentence" covering definitions,
  explanations and multiple-choice questions on Operating Systems, Discrete Maths and AI/NLP.
- **Where it is:** `training/datasets/te_style/` (translator data), `m2_te/` and `m4_te/` (definition and quiz data),
  `te_plus/` (the same data plus human-translated sentences — see Contribution 5).
  A dataset card is in `training/datasets/te_style/README.md`.
- **How it was made (important):** the Telugu side was **written by large AI models**, not by human translators
  (a technique called *distillation*: a big model teaches a smaller one). Every pair was then checked automatically
  — right amount of Telugu script, content matches the English, quiz questions well-formed — and pairs that failed were dropped.
  It has **not** been checked by humans at scale.

### Contribution 3 — A guarantee that formulas and code are never broken
- **The risk:** translation models sometimes drop or scramble symbols. A lesson that shows a broken formula
  like `p → q` or `a₁₁` is worse than an untranslated one.
- **What the code does:** before translating, every formula and code snippet is replaced with a placeholder
  (`Z1Z`, `Z2Z`, …). The model translates the text around the placeholders. Afterwards each placeholder is put back **exactly**.
  If the model dropped one, repeated one, or invented one, that sentence is **left in English** instead of showing a damaged formula.
- **Where it is:** `src/indiclearn/studio/neural.py` (functions `_protect`, `_restore`, `to_telugu`).
  Tests that lock this behaviour: `tests/studio/test_offline_mode.py`.

### Contribution 4 — A model that decides which words stay in English
- **What it is:** a small model that, for each English word in a sentence, predicts whether a Telugu lecturer would
  **keep it in English** (like "stack") or **translate it** (like "the").
- **How it was trained:** the labels were extracted automatically from Contribution 2 — a word counts as "kept"
  if it appears in English letters in the Telugu side of a pair. 45,508 words for training, 6,815 for testing.
- **Result:** macro-F1 **0.823**. Two simple rules scored about 0.35 on the same test ("always keep English" and
  "keep capitalised words"). The model also produced a glossary of 3,024 terms with how often each is kept
  (`models/trained/te_term_keeper/glossary.json`).
- **Not yet wired into the app.** It is a standalone result.

### Contribution 5 — Human-translated data made the translator better
- I added 6,000 sentence pairs from **BPCC**, a public English–Telugu dataset translated by humans (AI4Bharat),
  and trained again. They are marked with a different instruction (`to Telugu (formal):`) so the model does not
  confuse formal Telugu with classroom Telugu.
- **Result:** score rose from 72.0 to **73.85**; quiz questions with valid format rose from 99.3% to 100%;
  the correct answer letter was kept in 69.3% of quiz items, up from 64%.
- **Caveat:** many of those 6,000 sentences are general news and film text, not technical. They help Telugu fluency;
  they do not teach technical style.

### Contribution 6 — Everything works with no internet
The app can run entirely on a laptop with 16 GB RAM and a 6 GB graphics card (RTX 3050):
the models above, plus Gemma (a small chat model, run through Ollama), Whisper (speech to text), MMS (text to speech),
and IndicTrans2 (general translation). One switch in `.env` — `INDICLEARN_OFFLINE=true` — keeps every request on the laptop.
With the switch off, the app uses online AI first (Gemini, Sarvam) and falls back to the local models.

---

## 3. Results, explained

### What the score means
The score is **chrF**: it compares the model's Telugu sentence with a reference sentence (the "right answer")
by counting how many groups of characters match, from 0 (nothing in common) to 100 (identical).
Higher is better. All systems below were tested on the **same 349 sentences** against the **same references**.

| System | chrF score | How much of the output is Telugu script |
|---|---|---|
| **My translator, version 2** (with human data) | **73.85** | 32.6% |
| My translator, version 1 | 72.0 | 31.4% |
| IndicBART (another model, same data) | 63.4 | 28.8% |
| **IndicTrans2** (standard tool, no fine-tuning) | **29.1** | **99.6%** |

Sources: `training/reports/te_plus__mt5.json`, `te_style__mt5.json`, `te_style__indicbart.json`, `te_style__indictrans2_baseline.json`.

### How to read this table honestly
The last column is the key. IndicTrans2 outputs almost pure Telugu (99.6%), because that is what it is built to do.
My model keeps English technical words, so only about a third of its output is Telugu script. The references are
classroom-style, so IndicTrans2 is being scored against a style it was never asked to produce.
**The big gap (29.1 vs 73.85) mostly shows this difference in style, not that my model is a better
translator in general.** For ordinary full-Telugu translation, IndicTrans2 is the stronger system.

What the table does show: for the *classroom-style* task, a standard tool does not produce the right style
and my fine-tuned model does.

---

## 4. A real example, with the mistakes

This is what the model printed when run on a laptop (`training/show_contribution.py`). Under each
Telugu line is what to notice.

**English:** A stack is a last-in-first-out data structure used for function calls.
**Model:** `Stack అనేది function calls కోసం ఉపయోగించే చివరి-in-first-out data structure.`
→ Good: "stack", "function calls", "data structure" stay in English. Flaw: "last-in-first-out" came out
half-Telugu (`చివరి-in-first-out`).

**English:** A page fault occurs when a process accesses a page not currently in main memory.
**Model:** `ఒక process ప్రస్తుతం main memory లో లేని page ను access చేసేటప్పుడు వచ్చే page fault.`
→ The best example: natural classroom Telugu with the right words kept in English.

**English:** Modus ponens is a rule of inference: if p is true and p implies q, then q is true.
**Model:** `Modus ponens అనేది inference యొక్క rule: p నిజం మరియు p q ను సూచిస్తుంది, అప్పుడు q నిజం          ం.`
→ Wrong: "p implies q" became "p suggests q", the word "if" is lost, and the sentence ends in stray spaces.

**English:** A deadlock happens when two processes each wait for a resource the other holds.
**Model:** `రెండు processes ఒక resource ను ఉంచినప్పుడు ఒక deadlock జరుగుతుంది.`
→ Wrong meaning: the idea that each process *waits for a resource the other holds* is lost.

**Summary:** the model is good at the *style*, and still makes *meaning* mistakes on logical wording.
These mistakes are kept in this file on purpose.

---

## 5. Limits — read before citing any number

1. **Fine-tuning, not new research architecture, not trained from scratch.** The base models (mT5, XLM-R, Flan-T5)
   were built by others. The new parts are the task, the dataset, the formula-safety code and the evaluation.
2. **The training data's Telugu is AI-written** and only automatically checked.
3. **The 29.1 vs 73.85 comparison mostly reflects a difference in style** (section 3).
4. **Part of the score is inflated** because English technical words are copied from the source straight into the answer,
   so they match the reference for free.
5. **Meaning mistakes remain** (section 4). A person who reads Telugu should check lessons before students rely on them.
6. **A negative result:** I first tried to make the models write Telugu definitions and quizzes directly. Quality was poor
   (1.1–1.6 out of 5 for correctness), so the design changed to "write in English, then translate".
7. **No human evaluation yet**, and the model and data are **not yet published** (the cards are ready for Hugging Face).
8. **The offline chat model (Gemma, small size) is weaker than Gemini**, especially in Telugu. That is the trade-off of having no internet.
9. Telugu only, and mostly one subject area (computer-science courses).

## 6. What to do next, most valuable first

1. **Human evaluation:** 15–20 Telugu speakers rate fluency and meaning of ~30 sentences each.
2. **Publish** the model, dataset and cards on Hugging Face so others can use and cite them.
3. **Reduce meaning mistakes** on logical sentences (more logic-heavy training data; generate several answers and reject bad ones).

---

## 7. How to run things

Open a terminal in the project folder (`D:\Indic_NLP`). `.venv\Scripts\python` means "the project's own Python".

| I want to… | Command |
|---|---|
| Set everything up for offline use (first time; install Ollama first from ollama.com) | `setup_offline.bat` |
| Check what the app will use (chat model, GPU, translators) | `check_ai.bat` |
| **See the demo** (classroom-style Telugu for 4 sentences) | `.venv\Scripts\python training\show_contribution.py` |
| Re-measure the standard-tool comparison (slow: about 2–3 hours on CPU; do not interrupt) | `.venv\Scripts\python training\baseline_fair.py` |
| Rebuild and re-test the word-keeping model (20 seconds) | `.venv\Scripts\python training\term_keeper.py` |
| Install a model downloaded from Kaggle | `.venv\Scripts\python scripts\install_telugu_models.py D:\results.zip` |
| Build the human-data mix (needs a Hugging Face login and accepting the BPCC terms) | `.venv\Scripts\python training\get_bpcc.py` |
| Retrain on a free Kaggle GPU | `.venv\Scripts\python training\kaggle_run.py indic --only te_plus --no-wait` |
| Start the app (offline is set in `.env`) | `run.bat` |
| Run the automated tests | `.venv\Scripts\python -m pytest tests\studio -q` |

Troubleshooting:
- **"CUDA out of memory"** (the 6 GB graphics card is shared with Gemma): run `ollama stop gemma4:e2b`, close the app, then check
  `nvidia-smi` (it should show under 1 GB used).
- **Library versions that work together on this laptop:** torch 2.6.0 (CUDA 12.4), transformers 4.57.x (must be below 5),
  sentencepiece, sacrebleu. Transformers 5 breaks the IndicTrans2 code.

## 8. Where things are

| What | Where |
|---|---|
| The translator model (version 2) and its model card | `models/trained/te_style_translator/` |
| The datasets and the dataset card | `training/datasets/te_style/`, `te_plus/`, `m2_te/`, `m4_te/` |
| The word-keeping model and glossary | `models/trained/te_term_keeper/` |
| Every measured number | `training/reports/` |
| Formula-safe translation code | `src/indiclearn/studio/neural.py` |
| The online/offline switch | `src/indiclearn/studio/brain.py` (`offline_mode`) and `.env` (`INDICLEARN_OFFLINE`) |
| Demo and measurement scripts | `training/show_contribution.py`, `training/baseline_fair.py`, `training/term_keeper.py` |

---

## Glossary

- **Fine-tuning:** taking a model someone else already trained and training it further on your own examples so it learns a specific task.
- **Model / language model:** a program that has learned patterns from text and can produce text (here: translations).
- **Dataset:** the examples a model learns from. Here: pairs of English and Telugu sentences.
- **Translation baseline:** a standard tool you compare against, to show whether your model is actually better.
- **Register:** the style of language used in a setting (classroom, news, chat).
- **Code-mixing:** using two languages in one sentence (Telugu with English technical words).
- **chrF:** a 0–100 score comparing a model's output with a reference by matching character groups.
- **Macro-F1:** a 0–1 score for a yes/no prediction that gives both answers equal weight; 1 is perfect.
- **Distillation:** a large model writes training examples for a smaller one.
- **Test set:** sentences held back during training, used only to measure the result.
- **GPU / CUDA:** the graphics card and the technology that lets models run fast on it.
- **Ollama / Gemma:** the tool and the small chat model that answer student questions offline.
- **BPCC:** a public English–Indian-language dataset, human-translated, from AI4Bharat.
- **IndicTrans2:** a standard English↔Indian-language translation model from AI4Bharat, used here as the comparison.
