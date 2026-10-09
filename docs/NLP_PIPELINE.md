# Indic NLP & Cross-Lingual Concept Mapping Pipeline

**Status**: IMPLEMENTED (Phase 4 baseline verified on controlled benchmark).  
**Primary Focus**: Bridging student queries in English, Telugu script, Romanized Telugu, and Telugu-English code-mixed formulations to canonical academic concept IDs in the Knowledge Graph without making assumptions about student knowledge or mastery.

---

## 1. Architectural Overview

The NLP pipeline transforms free-form, multilingual, code-mixed engineering student queries into canonical academic concepts anchored in the Academic Knowledge Graph (`GraphStore`).

```text
Student Query (English / Telugu / Romanized Telugu / Code-Mixed)
                           ↓
    Stage 1: Language and Script Identification
         [HeuristicLanguageDetector Protocol]
                           ↓
    Stage 2: Token-Level Code-Mix Analysis
          [RuleBasedCodeMixDetector Protocol]
                           ↓
    Stage 3: Unicode & Romanization Normalization
    [UnifiedQueryNormalizer / RuleBasedRomanizationNormalizer]
         (Preserves Raw Input & Generates Ranked Variants)
                           ↓
    Stage 4: Graph-Grounded Technical Term Detection
            [GraphGroundedTermDetector Protocol]
         (Grounds in Canonical Concepts & Multilingual Aliases)
                           ↓
    Stage 5: Cross-Lingual Concept Mapping
            [Baselines A–D / HybridConceptMapper]
                           ↓
             Canonical concept_id(s) & Provenance
                           ↓
            Academic Knowledge Graph (`GraphStore`)
```

---

## 2. Pipeline Stages & Protocols

### Stage 1: Language & Script Identification
- **Protocol**: `LanguageDetector` (`indiclearn.nlp.language_detection.base`)
- **Implementation**: `HeuristicLanguageDetector` (`indiclearn.nlp.language_detection.detector`)
- **Supported Languages**:
  - `en` (English)
  - `te` (Telugu script)
  - `te-Latn` (Romanized Telugu)
  - `te-en` (Telugu-English code-mixed)
  - `en-te` (English-Telugu code-mixed)
  - `unknown`
- **Supported Scripts**:
  - `Latin`
  - `Telugu`
  - `Mixed`
  - `Unknown`
- **Methodology**: Evaluates Unicode character code-points (`0x0C00`–`0x0C7F` for Telugu script, ASCII letters for Latin), counts Romanized Telugu functional markers (`ante`, `enti`, `cheyyi`, `lo`, `ki`, `kosam`), detects engineering vocabulary, and balances language tags to prevent English technical vocabulary from forcing a Romanized query into English.

### Stage 2: Code-Mix Span Detection
- **Protocol**: `CodeMixDetector` (`indiclearn.nlp.code_mix.base`)
- **Implementation**: `RuleBasedCodeMixDetector` (`indiclearn.nlp.code_mix.detector`)
- **Outputs**:
  - Token spans with character offsets (`start`, `end`, `text`, `language`, `is_technical_term`)
  - Matrix language identification
  - Embedded language list
  - Code-Mixing Index (CMI): $CMI = \frac{\sum_{i=1}^n w_i - \max(w_L)}{\sum_{i=1}^n w_i}$

### Stage 3: Normalization & Ambiguity Preservation
- **Protocols**: `TextNormalizer` (`indiclearn.nlp.normalization.base`), `RomanizationNormalizer` (`indiclearn.nlp.romanization.base`)
- **Implementation**: `UnifiedQueryNormalizer`, `RuleBasedRomanizationNormalizer`
- **Core Principle**: **NEVER overwrite the student's original query.** The raw query is preserved immutably in `original_text`.
- **Normalization Operations**:
  1. Unicode NFC normalization (critical for Indic vowel signs and conjuncts).
  2. Whitespace and non-informative character hygiene.
  3. Romanized Telugu variation mapping (`yenti`/`emiti` $\rightarrow$ `enti`; `antey` $\rightarrow$ `ante`; `avuthundi` $\rightarrow$ `avutundi`).
  4. Generation of multiple candidate interpretations (`candidate_variants`) with explicit heuristic confidence scores.
  5. Back-transliteration to Telugu script representation (`transliterated_telugu`).

### Stage 4: Graph-Grounded Technical Term Detection
- **Protocol**: `TechnicalTermDetector` (`indiclearn.nlp.terminology.base`)
- **Implementation**: `GraphGroundedTermDetector` (`indiclearn.nlp.terminology.detector`)
- **Grounding**: Synchronized directly with the Academic Knowledge Graph (`GraphStore` / `Concept` repository). Does **NOT** maintain a disconnected secondary vocabulary.
- **Match Types**:
  - `canonical`: Exact match with `Concept.canonical_name`
  - `alias`: Match with academic textbook synonyms
  - `acronym`: Case-aware uppercase technical acronyms (`LTI`, `ROC`, `CTFT`, `DTFT`)
  - `telugu_alias`: Script match (`కన్వల్యూషన్`, `లాప్లేస్ ట్రాన్స్‌ఫార్మ్`)
  - `romanized_alias`: Romanized phonetic aliases (`convolution ante`)

### Stage 5: Concept Mapping Baselines & Hybrid Engine
- **Protocol**: `ConceptMapper` (`indiclearn.nlp.concept_mapping.base`)
- **Baselines Implemented**:
  - **Baseline A (`ExactConceptMapper`)**: String matching against canonical name and aliases. Fails on free-form questions, establishing the necessity of NLP normalization.
  - **Baseline B (`LexicalConceptMapper`)**: Normalized lexical matching combining extracted technical terms and token overlap.
  - **Baseline C (`EmbeddingConceptMapper`)**: Dense semantic candidate ranking via embedding vectors.
  - **Baseline D (`HybridConceptMapper`)**: Production system fusing terminology (0.50), lexical (0.20), and semantic (0.30) signals.

---

## 3. Strict Decoupling from Learner Model

Under **Phase 4 Research Mandates**:
- A query like `"convolution ante enti?"` conveys only that the student has queried about convolution.
- **NO student knowledge state**, **NO mastery score** ($\mu, \sigma$), and **NO ignorance or weakness assumptions** are inferred or stored in `NLPAnalysisResult` or `ConceptMappingResult`.
- The Academic Knowledge Graph remains completely immutable with respect to learner queries.

---

## 4. Controlled Benchmark & Empirical Findings (EXP-004)

Evaluated on `data/evaluation/concept_mapping_eval.json` (32 manually annotated gold queries across Signals & Systems concepts):

| Metric | Baseline A (Exact) | Baseline B (Lexical) | Baseline C (Embedding) | Baseline D (Hybrid) |
|---|---|---|---|---|
| **Recall@1** | 0.0000 | 1.0000 | 0.0625 | 1.0000 |
| **Recall@3** | 0.0000 | 1.0000 | 0.1562 | 1.0000 |
| **MRR** | 0.0000 | 1.0000 | 0.1042 | 1.0000 |

### Language & Script Identification Performance
- **Language Accuracy**: 96.88% (31/32)
- **Script Accuracy**: 100.00% (32/32)
- **Per-Category F1**:
  - `en`: 1.0000
  - `te`: 0.9333 (1 query with Latin acronym `LTI` classified as code-mixed `te-en`)
  - `te-Latn`: 1.0000
  - `te-en`: 0.9412

### Technical Term Detection
- **Precision**: 1.0000
- **Recall**: 1.0000
- **F1**: 1.0000 (36 true positive term instances across 32 queries)

---

## 5. Limitations & Future Work

1. **Heuristic Lexicon Scope**: The Romanized Telugu dictionary is intentionally curated for conversational and engineering query particles. Colloquial slang or rare dialectal variants outside this core require learned transformer transliteration models (e.g., IndicBART / Aksharantar).
2. **Intent Classification**: Query intent classification (`EXPLAIN_CONCEPT`, `DEFINE`, `FORMULA`, `DERIVATION`, `EXAMPLE`, `QUIZ`) is specified in interface design but explicitly deferred to Phase 5.
3. **Complex Sentence Boundaries**: Long compound questions containing multiple clauses require dependency parsing or fine-grained multilingual token tagging in future iterations.
