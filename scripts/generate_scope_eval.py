"""Generate curated curriculum scope evaluation dataset (Phase 5.2).

Produces:
    data/evaluation/scope_eval.json

Contains 45 carefully constructed queries across:
- in_scope (18 queries)
- out_of_scope (18 queries, including all 10 mandated adversarial/hard negatives)
- ambiguous (9 queries, under-specified generic queries)

Spans:
- English, Telugu script, Romanized Telugu, and Code-Mixed
- Easy, Medium, Hard
- Explicit justification 'reason' and 'gold_concept_ids'
"""

from __future__ import annotations

import json
from pathlib import Path


def generate_scope_eval_dataset() -> list[dict]:
    queries = [
        # =====================================================================
        # 1. IN_SCOPE QUERIES (18 queries)
        # =====================================================================
        {
            "query_id": "SCOPE_IN_001",
            "query": "What is an LTI system and what two properties define it?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_lti_system"],
            "language": "en",
            "difficulty": "easy",
            "reason": "Direct inquiry on core canonical Signals & Systems concept with explicit properties (linearity and time-invariance).",
        },
        {
            "query_id": "SCOPE_IN_002",
            "query": "Explain convolution in continuous-time LTI systems.",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_convolution", "cpt_signals_and_systems_lti_system"],
            "language": "en",
            "difficulty": "easy",
            "reason": "Foundational curriculum relationship between convolution integral and LTI system response.",
        },
        {
            "query_id": "SCOPE_IN_003",
            "query": "What is the difference between an energy signal and a power signal?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_energy_signal", "cpt_signals_and_systems_power_signal"],
            "language": "en",
            "difficulty": "medium",
            "reason": "Curriculum comparison between finite energy and non-zero finite average power signal classifications.",
        },
        {
            "query_id": "SCOPE_IN_004",
            "query": "State the Nyquist sampling theorem for bandlimited continuous signals.",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_sampling_theorem"],
            "language": "en",
            "difficulty": "easy",
            "reason": "Canonical continuous-time sampling rate criterion (fs >= 2*B).",
        },
        {
            "query_id": "SCOPE_IN_005",
            "query": "Why does an ideal reconstruction filter require a sinc interpolation in the time domain?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_reconstruction_filter", "cpt_signals_and_systems_sampling_theorem"],
            "language": "en",
            "difficulty": "hard",
            "reason": "Advanced continuous-time signal reconstruction inquiry explicitly covered in course Unit 1.",
        },
        {
            "query_id": "SCOPE_IN_006",
            "query": "How do we combine an input signal with an impulse response to obtain system output without using the word convolution?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_convolution", "cpt_signals_and_systems_impulse_response"],
            "language": "en",
            "difficulty": "hard",
            "reason": "Hard conceptual paraphrase testing retrieval and scope grounding without canonical keyword string matching.",
        },
        {
            "query_id": "SCOPE_IN_007",
            "query": "What is the relationship between frequency response and impulse response for an LTI system?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_frequency_response", "cpt_signals_and_systems_impulse_response"],
            "language": "en",
            "difficulty": "medium",
            "reason": "Fourier transform relationship between h(t) and H(j omega).",
        },
        {
            "query_id": "SCOPE_IN_008",
            "query": "Explain how causality is determined from the impulse response of a continuous system.",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_causal_system", "cpt_signals_and_systems_impulse_response"],
            "language": "en",
            "difficulty": "medium",
            "reason": "Causality property criterion (h(t) = 0 for t < 0) in Signals & Systems syllabus.",
        },
        {
            "query_id": "SCOPE_IN_009",
            "query": "కన్వల్యూషన్ అంటే ఏమిటి మరియు LTI సిస్టమ్స్‌లో దీని ప్రాముఖ్యత ఏమిటి?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_convolution", "cpt_signals_and_systems_lti_system"],
            "language": "te",
            "difficulty": "easy",
            "reason": "Native Telugu script formulation of core convolution concept.",
        },
        {
            "query_id": "SCOPE_IN_010",
            "query": "లీనియర్ టైమ్-ఇన్వేరియంట్ సిస్టమ్ యొక్క ఇంపల్స్ రెస్పాన్స్ ఎలా పనిచేస్తుంది?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_impulse_response", "cpt_signals_and_systems_lti_system"],
            "language": "te",
            "difficulty": "medium",
            "reason": "Native Telugu formulation of impulse response in LTI systems.",
        },
        {
            "query_id": "SCOPE_IN_011",
            "query": "శాంప్లింగ్ సిద్ధాంతం ప్రకారం నైక్విస్ట్ రేటును ఎలా లెక్కిస్తారు?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_sampling_theorem"],
            "language": "te",
            "difficulty": "medium",
            "reason": "Native Telugu calculation inquiry for Nyquist sampling rate.",
        },
        {
            "query_id": "SCOPE_IN_012",
            "query": "LTI system lo impulse response and frequency response madhya relation enti?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_impulse_response", "cpt_signals_and_systems_frequency_response"],
            "language": "te-Latn",
            "difficulty": "medium",
            "reason": "Romanized Telugu inquiry on frequency response and impulse response.",
        },
        {
            "query_id": "SCOPE_IN_013",
            "query": "convolution ante enti? input signal mariyu impulse response ni ela combine chestharu?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_convolution"],
            "language": "te-Latn",
            "difficulty": "easy",
            "reason": "Conversational Romanized Telugu question on convolution mechanics.",
        },
        {
            "query_id": "SCOPE_IN_014",
            "query": "Continuous time lo energy signal mariyu power signal difference yenti explain cheyyi",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_energy_signal", "cpt_signals_and_systems_power_signal"],
            "language": "te-Latn",
            "difficulty": "medium",
            "reason": "Romanized Telugu comparison between energy and power signals.",
        },
        {
            "query_id": "SCOPE_IN_015",
            "query": "LTI system causal avvali ante impulse response h(t) meeda condition enti?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_causal_system", "cpt_signals_and_systems_impulse_response"],
            "language": "te-en",
            "difficulty": "medium",
            "reason": "Code-mixed engineering question on causality condition.",
        },
        {
            "query_id": "SCOPE_IN_016",
            "query": "CTFT properties lo time shifting property output frequency spectrum ni ela change chesthundi?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_ctft"],
            "language": "te-en",
            "difficulty": "hard",
            "reason": "Code-mixed inquiry on Continuous-Time Fourier Transform time shifting property.",
        },
        {
            "query_id": "SCOPE_IN_017",
            "query": "Continuous-Time Fourier Transform formula for a non-periodic signal x(t)",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_ctft"],
            "language": "en",
            "difficulty": "easy",
            "reason": "Direct formula retrieval for continuous-time Fourier transform.",
        },
        {
            "query_id": "SCOPE_IN_018",
            "query": "Bandlimited signal reconstruction lo aliasing effect ni ela prevent cheyali?",
            "gold_scope": "in_scope",
            "gold_concept_ids": ["cpt_signals_and_systems_sampling_theorem"],
            "language": "te-en",
            "difficulty": "medium",
            "reason": "Code-mixed question on aliasing prevention via Nyquist sampling criterion.",
        },

        # =====================================================================
        # 2. OUT_OF_SCOPE QUERIES (18 queries: includes 10 mandated + hard negatives)
        # =====================================================================
        {
            "query_id": "SCOPE_OUT_001",
            "query": "How do you design an FIR digital filter using the Kaiser window method?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares generic word 'filter' but belongs to Digital Signal Processing filter synthesis, absent from course.",
        },
        {
            "query_id": "SCOPE_OUT_002",
            "query": "Explain how the Kalman filter performs recursive state estimation in linear dynamic systems.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares generic words 'filter' and 'system' but belongs to advanced estimation theory and state-space control.",
        },
        {
            "query_id": "SCOPE_OUT_003",
            "query": "How is the Discrete Cosine Transform (DCT) applied in 2D JPEG image compression?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares generic word 'transform' but belongs to 2D image processing and compression standards.",
        },
        {
            "query_id": "SCOPE_OUT_004",
            "query": "Implement the Cooley-Tukey radix-2 FFT butterfly algorithm for decimation in time.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares 'algorithm' and relates to discrete Fourier computation, but decimation in time butterfly is DSP curriculum.",
        },
        {
            "query_id": "SCOPE_OUT_005",
            "query": "How does the Adaptive LMS filter algorithm update its weight vector via gradient descent?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares words 'filter' and 'algorithm' but belongs to adaptive signal processing and machine learning.",
        },
        {
            "query_id": "SCOPE_OUT_006",
            "query": "Explain analog Butterworth filter design and pole placement on the s-plane.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares 'filter' but Butterworth polynomial filter synthesis is specialized analog filter design.",
        },
        {
            "query_id": "SCOPE_OUT_007",
            "query": "How do you construct a Bode plot to evaluate gain margin and phase margin in feedback control?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares 'system' concepts but Bode plot stability margins belong to Control Systems engineering.",
        },
        {
            "query_id": "SCOPE_OUT_008",
            "query": "How is the controllability matrix rank computed in a modern state-space control system?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares 'system' but matrix rank state-space controllability belongs to Control Systems.",
        },
        {
            "query_id": "SCOPE_OUT_009",
            "query": "How do you tune proportional, integral, and derivative gains in a closed-loop PID controller?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "medium",
            "reason": "Closed-loop PID controller tuning belongs to Control Theory.",
        },
        {
            "query_id": "SCOPE_OUT_010",
            "query": "How is the 2D Fourier transform used to remove periodic noise patterns in digital image processing?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Hard negative: Shares 'Fourier transform' but specifically asks about 2D image filtering and pixel noise.",
        },
        {
            "query_id": "SCOPE_OUT_011",
            "query": "What is a spatial filter in 2D image processing convolution?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Hard negative: Shares 'filter' and 'convolution' but asks about 2D image processing kernels.",
        },
        {
            "query_id": "SCOPE_OUT_012",
            "query": "Explain Fourier transform in lossy image compression standards.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Hard negative: Shares 'Fourier transform' but targets image compression formats.",
        },
        {
            "query_id": "SCOPE_OUT_013",
            "query": "Explain how the Viterbi decoding algorithm corrects errors in convolutional code channels.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Shares 'convolutional' but belongs to digital communication channel coding.",
        },
        {
            "query_id": "SCOPE_OUT_014",
            "query": "Bilinear transformation method dwara analog filter poles ni digital IIR filter poles loki ela map chestharu?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "te-Latn",
            "difficulty": "hard",
            "reason": "Romanized Telugu inquiry on digital IIR bilinear transformation synthesis.",
        },
        {
            "query_id": "SCOPE_OUT_015",
            "query": "డిస్క్రీట్ వేవ్‌లెట్ ట్రాన్స్‌ఫార్మ్ (DWT) మల్టీ-రిజల్యూషన్ అనాలిసిస్‌లో సిగ్నల్స్‌ను ఎలా డీకంపోజ్ చేస్తుంది?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "te",
            "difficulty": "hard",
            "reason": "Telugu script query on Discrete Wavelet Transform multi-resolution analysis.",
        },
        {
            "query_id": "SCOPE_OUT_016",
            "query": "State space representation lo controllability matrix rank ela check chesi system control avthundo cheppandi",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "te-en",
            "difficulty": "hard",
            "reason": "Code-mixed question on state-space controllability matrix.",
        },
        {
            "query_id": "SCOPE_OUT_017",
            "query": "JPEG ఇమేజ్ కంప్రెషన్‌లో డిస్క్రీట్ కొసైన్ ట్రాన్స్‌ఫార్మ్ (DCT) ను ఎలా ఉపయోగిస్తారు?",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "te",
            "difficulty": "hard",
            "reason": "Telugu script question on Discrete Cosine Transform in JPEG image compression.",
        },
        {
            "query_id": "SCOPE_OUT_018",
            "query": "Use convolution to design a JPEG image compression algorithm and quantize DCT coefficients.",
            "gold_scope": "out_of_scope",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "hard",
            "reason": "Concept-name bait attack: Inserts 'convolution' into an unrelated image compression prompt.",
        },

        # =====================================================================
        # 3. AMBIGUOUS QUERIES (9 queries: under-specified, broad generic terms)
        # =====================================================================
        {
            "query_id": "SCOPE_AMB_001",
            "query": "Explain filters.",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "medium",
            "reason": "Under-specified generic query: 'filters' spans electronics, optics, software, audio, and water purification.",
        },
        {
            "query_id": "SCOPE_AMB_002",
            "query": "What is a system?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "medium",
            "reason": "Broad generic term without disciplinary context (could refer to operating systems, thermodynamic systems, etc.).",
        },
        {
            "query_id": "SCOPE_AMB_003",
            "query": "Explain transforms.",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "medium",
            "reason": "Under-specified generic query: could refer to Laplace, Fourier, Z, Wavelet, or affine geometric transforms.",
        },
        {
            "query_id": "SCOPE_AMB_004",
            "query": "What are signals?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "en",
            "difficulty": "medium",
            "reason": "Generic query without specifying continuous, discrete, biological, or digital signals.",
        },
        {
            "query_id": "SCOPE_AMB_005",
            "query": "filter ante enti?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "te-Latn",
            "difficulty": "medium",
            "reason": "Romanized Telugu under-specified generic query without technical qualifiers.",
        },
        {
            "query_id": "SCOPE_AMB_006",
            "query": "system ante enti?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "te-Latn",
            "difficulty": "medium",
            "reason": "Romanized Telugu broad question lacking curriculum context.",
        },
        {
            "query_id": "SCOPE_AMB_007",
            "query": "ఫిల్టర్ అంటే ఏమిటి?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "te",
            "difficulty": "medium",
            "reason": "Telugu script generic query asking broadly what a filter is without curriculum context.",
        },
        {
            "query_id": "SCOPE_AMB_008",
            "query": "సిస్టమ్ అంటే ఏమిటి?",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "te",
            "difficulty": "medium",
            "reason": "Telugu script broad query on generic systems.",
        },
        {
            "query_id": "SCOPE_AMB_009",
            "query": "filters ni explain cheyyi",
            "gold_scope": "ambiguous",
            "gold_concept_ids": [],
            "language": "te-Latn",
            "difficulty": "medium",
            "reason": "Romanized Telugu command asking to explain filters without specifying continuous or reconstruction filters.",
        },
    ]
    return queries


def main() -> None:
    queries = generate_scope_eval_dataset()
    out_path = Path("data/evaluation/scope_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "benchmark_id": "SCOPE-EVAL-45",
        "description": "Curriculum scope and ambiguity evaluation benchmark for Signals and Systems",
        "total_queries": len(queries),
        "scope_distribution": {
            "in_scope": sum(1 for q in queries if q["gold_scope"] == "in_scope"),
            "out_of_scope": sum(1 for q in queries if q["gold_scope"] == "out_of_scope"),
            "ambiguous": sum(1 for q in queries if q["gold_scope"] == "ambiguous"),
        },
        "language_distribution": {
            "en": sum(1 for q in queries if q["language"] == "en"),
            "te": sum(1 for q in queries if q["language"] == "te"),
            "te-Latn": sum(1 for q in queries if q["language"] == "te-Latn"),
            "te-en": sum(1 for q in queries if q["language"] == "te-en"),
        },
        "queries": queries,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(queries)} queries in {out_path.resolve()}")
    print(f"Scope Distribution: {data['scope_distribution']}")
    print(f"Language Distribution: {data['language_distribution']}")


if __name__ == "__main__":
    main()
