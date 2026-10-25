# IndicLearn — Contributions to Indic-language NLP

Author: Gundeti Siddarth · Project: IndicLearn (offline study app for Telugu-medium engineering students)
Every number below comes from a report file in `training/reports/`. Nothing here is estimated.

---

## 1. The problem

Telugu-medium students study engineering from English notes. Standard machine translation
(IndicTrans2, NLLB) turns **everything** into Telugu, including technical terms ("stack" → "స్టాక్"),
which is not how lecturers teach. Lecturers speak Telugu sentences and **keep the technical terms in
English**. That register — code-mixed technical Telugu — was not served by any model I could find
(existing Telugu–English code-mixed corpora are social-media text; technical-term retention work exists
for Hindi using constrained decoding, not a fine-tuned model).

## 2. Contributions

| # | Contribution | What it is | Evidence |
|---|---|---|---|
| 1 | **Lecturer-register Telugu translator** | `google/mt5-base` (582M) fine-tuned to write Telugu with English technical terms kept | chrF **73.85** on 349 held-out items (v1: 72.0) |
| 2 | **Training corpus** | ~5,200 English → code-mixed Telugu pairs (`training/datasets/te_style`, `m2_te`, `m4_te`), auto-validated | 2,089 train / 157 val / 349 test for the translator |
| 3 | **Formula- and code-safe translation protocol** | Formulas/code are masked, translation runs around them, and restored exactly; a sentence whose notation cannot be restored verbatim stays in English | Locked by tests in `tests/studio/test_offline_mode.py` |
| 4 | **Technical-term retention model** | Predicts which English words a Telugu lecturer keeps in English | macro-F1 **0.823** vs 0.353 (always keep) and 0.357 (capitalisation rule) |
| 5 | **Human data improves it** | v2 adds 6,000 human-translated pairs (AI4Bharat BPCC) under a separate prompt tag | chrF 72.0 → **73.85**, well-formed MCQs 99.3% → 100%, answer-letter match 64% → 69.3% |
| 6 | **Fully offline system** | All of the above plus Gemma (Ollama), Whisper-Telugu, MMS voices, IndicTrans2 run on a 16 GB RAM / 6 GB RTX 3050 laptop with `INDICLEARN_OFFLINE=true` | `scripts/setup_offline.py`, `check_ai.bat` |

## 3. Results

### 3.1 Translator vs the standard baseline (same 349 test items, same references)

| System | chrF | Telugu-script share |
|---|---|---|
| **IndicLearn translator v2** (mT5-base + human BPCC) | **73.85** | 32.6% (English terms kept) |
| IndicLearn translator v1 (lecturer-style data only) | 72.0 | 31.4% |
| IndicBART variant (same data) | 63.4 | 28.8% |
| IndicTrans2 en→te (standard baseline) | 29.1 | 99.6% (translates the terms too) |

Source: `training/reports/te_style__mt5.json`, `te_plus__mt5.json`, `te_style__indicbart.json`,
`te_style__indictrans2_baseline.json`.

**How to read it.** The references are code-mixed, so IndicTrans2 is scored against a style it was never
asked to produce. The chrF gap mostly measures that **register difference**, not general translation
quality; on ordinary full-Telugu translation IndicTrans2 is the stronger system.

### 3.2 Term-retention model

45,508 training words, 6,815 test words (labels extracted automatically from the corpus: a word is "kept" if it
appears in Latin script in the Telugu target). Macro-F1 0.823; glossary of 3,024 terms with keep-rates.
Source: `training/reports/term_keeper.json`.

### 3.3 Live demo output (v2 model, run on the laptop GPU)

Command: `.venv\Scripts\python training\show_contribution.py`

```
ENGLISH   : A stack is a last-in-first-out data structure used for function calls.
OUR MODEL : Stack అనేది function calls కోసం ఉపయోగించే చివరి-in-first-out data structure.
            ^ Telugu script: 36%  (English technical terms kept)

ENGLISH   : A page fault occurs when a process accesses a page not currently in main memory.
OUR MODEL : ఒక process ప్రస్తుతం main memory లో లేని page ను access చేసేటప్పుడు వచ్చే page fault.
            ^ Telugu script: 49%  (English technical terms kept)

ENGLISH   : Modus ponens is a rule of inference: if p is true and p implies q, then q is true.
OUR MODEL : Modus ponens అనేది inference యొక్క rule: p నిజం మరియు p q ను సూచిస్తుంది, అప్పుడు q నిజం          ం.
            ^ Telugu script: 61%  (English technical terms kept)

ENGLISH   : A deadlock happens when two processes each wait for a resource the other holds.
OUR MODEL : రెండు processes ఒక resource ను ఉంచినప్పుడు ఒక deadlock జరుగుతుంది.
            ^ Telugu script: 56%  (English technical terms kept)
```

Honest reading of these four outputs:
- **Good:** sentences 1 and 2 keep the technical terms in English with natural Telugu grammar; sentence 2 is the strongest.
- **Errors (kept on purpose):** "last-in-first-out" is half-translated (`చివరి-in-first-out`); in sentence 3 "p implies q" became
  "p q ను సూచిస్తుంది" ("p suggests q"), the "if" is lost, and the line ends in stray spaces; sentence 4 changes
  the meaning (the "each wait for a resource the other holds" idea is lost).
- The side-by-side column for IndicTrans2 was not shown in this run because IndicTrans2 had not loaded yet; its numbers are in 3.1.

## 4. Limits (say these first)

1. **Fine-tuning, not a new architecture and not training from scratch.** Base models: mT5-base, XLM-R base, Flan-T5-base.
2. **The training data's Telugu side is AI-written** (teacher–student distillation), then auto-validated for
   Telugu-script share, source alignment and MCQ format. It is not human-reviewed at scale.
3. **The 6,000 human BPCC pairs are general-domain** (the "technical" filter was loose: some are news and film text). They add formal-Telugu fluency, not technical style.
4. **chrF on code-mixed references is partly inflated** because English terms are copied from the source.
5. **Meaning errors remain on logical wording** (3.3). Human proofreading is needed before classroom use.
6. **Negative result:** direct Telugu generation by M2/M4 (mT5/IndicBART) reached only 1.1–1.6/5 correctness; the translate-after-generate design replaced it.
7. **No human evaluation yet** and no publication yet.
8. The offline chat model (Gemma 4 E2B) is weaker than Gemini in Telugu; that is the cost of working with no internet.
9. Telugu only; one education domain.

## 5. Next steps (highest value first)

1. Human evaluation: 15–20 Telugu speakers rate fluency and meaning on ~30 sentences each.
2. Publish model, dataset and cards on Hugging Face (`models/trained/te_style_translator/README.md`, `training/datasets/te_style/README.md`).
3. Reduce meaning errors on logical sentences (sampling + rejection, more logic-heavy training data).

## 6. Run commands

Run from `D:\Indic_NLP`. Prefix is the project virtual environment.

```
# One-time offline setup (Ollama must be installed first; downloads CUDA PyTorch, Telugu models, Gemma)
setup_offline.bat

# Check what the app will use (LLM chain, device, models)
check_ai.bat
.venv\Scripts\python scripts\setup_offline.py --check

# Live demo: our model's lecturer-style Telugu vs standard translation
.venv\Scripts\python training\show_contribution.py

# Fair baseline: IndicTrans2 on the same 349 test items (CPU; takes about 2-3 hours; do not interrupt)
.venv\Scripts\pip install sacrebleu
.venv\Scripts\python training\baseline_fair.py

# Term-retention model: rebuild the dataset and retrain (20 seconds, CPU)
.venv\Scripts\python training\term_keeper.py

# Install a Telugu model downloaded from Kaggle (zip or folder)
.venv\Scripts\python scripts\install_telugu_models.py D:\results.zip

# Build the human-data mix (needs a Hugging Face login and acceptance of ai4bharat/BPCC terms)
.venv\Scripts\python -c "from huggingface_hub import login; login()"
.venv\Scripts\python training\get_bpcc.py

# Retrain on a free Kaggle GPU (needs a valid Kaggle token in secrets\kaggle.json)
.venv\Scripts\python training\kaggle_run.py indic --only te_plus --no-wait
.venv\Scripts\python training\kaggle_run.py status --kind indic

# Run the app fully offline (INDICLEARN_OFFLINE=true is set in .env)
run.bat

# Run the test suite
.venv\Scripts\python -m pytest tests\studio -q
```

If the GPU runs out of memory (6 GB shared with Gemma): `ollama stop gemma4:e2b`, close the app, check `nvidia-smi`.
Library versions that work together on this laptop: torch 2.6.0 (CUDA 12.4), transformers 4.57.x (below 5),
sentencepiece, sacrebleu.

## 7. Files

| What | Where |
|---|---|
| Translator model (v2) | `models/trained/te_style_translator/` (model card: `README.md` inside) |
| Translator dataset + card | `training/datasets/te_style/`, `training/datasets/te_plus/` |
| Term-retention model, glossary | `models/trained/te_term_keeper/` |
| Term-retention dataset | `training/datasets/term_keeper/` |
| Reports (all numbers above) | `training/reports/` |
| Formula-safe translation code | `src/indiclearn/studio/neural.py` (`_protect`, `_restore`, `to_telugu`) |
| Offline switch | `src/indiclearn/studio/brain.py` (`offline_mode`), `.env` (`INDICLEARN_OFFLINE`) |
