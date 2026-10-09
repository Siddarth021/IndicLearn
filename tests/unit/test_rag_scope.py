"""Unit tests for Curriculum Scope Assessment, Ambiguity Gating, and RAG Safety (Phase 5.2).

Tests:
- Strong in-scope vs weak in-scope
- Out-of-scope domain detection
- Ambiguous query phrases
- Generic-term-only penalties
- Adversarial lexical overlaps (FIR Kaiser, Kalman, DCT, LMS, PID)
- Concept mapping margin (large vs small margin)
- Evidence consistency checking
- Native Telugu script, Romanized Telugu, and Code-mixed queries
- Safety attacks (concept-name bait, generic keyword injection)
"""

from __future__ import annotations

from indiclearn.nlp.models import (
    CodeMixResult,
    ConceptCandidate,
    ConceptMappingResult,
    Language,
    LanguageDetectionResult,
    NLPAnalysisResult,
    NormalizationResult,
    Script,
)
from indiclearn.rag.scope import HeuristicScopeAssessor
from indiclearn.retrieval.base import RetrievalResult


def _make_dummy_nlp_result(
    query: str,
    primary_concept: ConceptCandidate | None = None,
    candidates: list[ConceptCandidate] | None = None,
    score_margin: float | None = None,
) -> NLPAnalysisResult:
    return NLPAnalysisResult(
        original_text=query,
        language_detection=LanguageDetectionResult(
            primary_language=Language.ENGLISH,
            detected_script=Script.LATIN,
            confidence=1.0,
        ),
        code_mix=CodeMixResult(
            is_code_mixed=False,
            matrix_language="en",
            cmi_score=0.0,
        ),
        normalization=NormalizationResult(
            original_text=query,
            canonical_normalized_text=query.lower(),
        ),
        technical_terms=[],
        concept_mapping=ConceptMappingResult(
            query=query,
            subject_id="signals_and_systems",
            primary_concept=primary_concept,
            candidates=candidates or ([primary_concept] if primary_concept else []),
            score_margin=score_margin,
            mapping_confidence=primary_concept.score if primary_concept else 0.0,
        ),
    )


def test_scope_strong_in_scope():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_convolution",
        canonical_name="Convolution",
        score=0.95,
        match_method="exact_canonical",
    )
    nlp_res = _make_dummy_nlp_result("Explain convolution in LTI systems", primary_concept=cand, score_margin=0.45)
    cands = [
        RetrievalResult(
            chunk_id="c1",
            document_id="doc_conv",
            rank=1,
            text="Convolution is a mathematical operation combining input and impulse response.",
            score=0.015,
            metadata={"concept_ids": ["cpt_signals_and_systems_convolution"]},
        )
    ]

    decision = assessor.assess_scope("Explain convolution in LTI systems", nlp_res, cands)
    assert decision.scope == "IN_SCOPE"
    assert decision.confidence_score >= 0.70
    assert decision.primary_concept_id == "cpt_signals_and_systems_convolution"


def test_scope_out_of_scope_adversarial_kaiser_filter():
    assessor = HeuristicScopeAssessor()
    # Concept mapper might mistakenly latch onto 'reconstruction_filter' because of word 'filter'
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_reconstruction_filter",
        canonical_name="Ideal Reconstruction Filter",
        score=0.55,
        match_method="normalized_lexical",
    )
    query = "FIR filter design using Kaiser window parameters"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=cand, score_margin=0.05)
    cands = [
        RetrievalResult(chunk_id="c1", document_id="doc1", rank=1, text="An ideal reconstruction filter acts as a lowpass filter.", score=0.010)
    ]

    decision = assessor.assess_scope(query, nlp_res, cands)
    # Must reject as OUT_OF_SCOPE despite the word 'filter' matching a concept
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("digital_filter_design" in ind for ind in decision.out_of_scope_indicators)


def test_scope_out_of_scope_kalman_filter():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_lti_system",
        canonical_name="LTI System",
        score=0.50,
        match_method="normalized_lexical",
    )
    query = "Kalman filter state estimation in linear state space systems"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=cand, score_margin=0.02)
    cands = [RetrievalResult(chunk_id="c1", document_id="doc1", rank=1, text="Linear time invariant systems.", score=0.009)]

    decision = assessor.assess_scope(query, nlp_res, cands)
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("control_systems" in ind for ind in decision.out_of_scope_indicators)


def test_scope_out_of_scope_jpeg_dct():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_ctft",
        canonical_name="Continuous-Time Fourier Transform",
        score=0.48,
        match_method="normalized_lexical",
    )
    query = "Discrete Cosine Transform (DCT) in JPEG image compression"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=cand, score_margin=0.03)

    decision = assessor.assess_scope(query, nlp_res, [])
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("image_and_multidimensional" in ind for ind in decision.out_of_scope_indicators)


def test_scope_out_of_scope_cooley_tukey_fft():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_ctft",
        canonical_name="Continuous-Time Fourier Transform",
        score=0.52,
        match_method="normalized_lexical",
    )
    query = "Cooley-Tukey butterfly algorithm for decimation in time"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=cand, score_margin=0.02)

    decision = assessor.assess_scope(query, nlp_res, [])
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("dsp_fft_algorithms" in ind for ind in decision.out_of_scope_indicators)


def test_scope_out_of_scope_pid_controller():
    assessor = HeuristicScopeAssessor()
    query = "How do you tune proportional and integral gains in a PID controller?"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=None)

    decision = assessor.assess_scope(query, nlp_res, [])
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("control_systems" in ind for ind in decision.out_of_scope_indicators)


def test_scope_concept_name_bait_attack():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_convolution",
        canonical_name="Convolution",
        score=0.85,
        match_method="exact_canonical",
    )
    # Attacker inserts 'convolution' to bait the system into answering an unrelated JPEG compression question
    query = "Use convolution to design a JPEG compression algorithm with quantization"
    nlp_res = _make_dummy_nlp_result(query, primary_concept=cand, score_margin=0.40)

    decision = assessor.assess_scope(query, nlp_res, [])
    # Must still reject as OUT_OF_SCOPE because JPEG is detected
    assert decision.scope == "OUT_OF_SCOPE"
    assert any("image_and_multidimensional" in ind for ind in decision.out_of_scope_indicators)


def test_scope_ambiguous_under_specified_queries():
    assessor = HeuristicScopeAssessor()

    ambiguous_examples = [
        "Explain filters.",
        "What is a system?",
        "Explain transforms",
        "filter ante enti?",
        "system ante enti?",
        "ఫిల్టర్ అంటే ఏమిటి?",
        "సిస్టమ్ అంటే ఏమిటి?",
    ]

    for q in ambiguous_examples:
        cand = ConceptCandidate(
            concept_id="cpt_signals_and_systems_reconstruction_filter",
            canonical_name="Ideal Reconstruction Filter",
            score=0.50,
            match_method="normalized_lexical",
        )
        nlp_res = _make_dummy_nlp_result(q, primary_concept=cand, score_margin=0.02)
        decision = assessor.assess_scope(q, nlp_res, [])
        assert decision.scope == "AMBIGUOUS", f"Failed on ambiguous query: {q}"


def test_scope_telugu_and_romanized_valid_in_scope():
    assessor = HeuristicScopeAssessor()

    # Native Telugu script in-scope
    c1 = ConceptCandidate(
        concept_id="cpt_signals_and_systems_convolution",
        canonical_name="Convolution",
        score=0.90,
        match_method="alias_match",
    )
    nlp_te = _make_dummy_nlp_result("కన్వల్యూషన్ అంటే ఏమిటి?", primary_concept=c1, score_margin=0.50)
    cands_te = [RetrievalResult(chunk_id="c1", document_id="doc1", rank=1, text="Convolution integral.", score=0.012)]
    d_te = assessor.assess_scope("కన్వల్యూషన్ అంటే ఏమిటి?", nlp_te, cands_te)
    assert d_te.scope == "IN_SCOPE"

    # Romanized Telugu in-scope
    nlp_rom = _make_dummy_nlp_result("convolution ante enti?", primary_concept=c1, score_margin=0.50)
    d_rom = assessor.assess_scope("convolution ante enti?", nlp_rom, cands_te)
    assert d_rom.scope == "IN_SCOPE"

    # Code-mixed in-scope
    c2 = ConceptCandidate(
        concept_id="cpt_signals_and_systems_lti_system",
        canonical_name="LTI System",
        score=0.88,
        match_method="exact_canonical",
    )
    nlp_cm = _make_dummy_nlp_result("LTI system ni explain cheyyi", primary_concept=c2, score_margin=0.40)
    cands_cm = [RetrievalResult(chunk_id="c2", document_id="doc1", rank=1, text="Linear time invariant systems.", score=0.015)]
    d_cm = assessor.assess_scope("LTI system ni explain cheyyi", nlp_cm, cands_cm)
    assert d_cm.scope == "IN_SCOPE"


def test_scope_evidence_consistency_check():
    assessor = HeuristicScopeAssessor()
    cand = ConceptCandidate(
        concept_id="cpt_signals_and_systems_power_signal",
        canonical_name="Power Signal",
        score=0.75,
        match_method="exact_canonical",
    )
    # Query maps to Power Signal, but retrieved chunks contain zero mention of power
    nlp_res = _make_dummy_nlp_result("What is a power signal?", primary_concept=cand, score_margin=0.30)
    unrelated_cands = [
        RetrievalResult(chunk_id="c1", document_id="doc1", rank=1, text="A resistor obeys Ohm's law and dissipates thermal energy.", score=0.010, metadata={"concept_ids": []}),
    ]
    decision = assessor.assess_scope("What is a power signal?", nlp_res, unrelated_cands)
    # The consistency score is 0.0, which is flagged
    assert decision.evidence_consistency_score == 0.0
