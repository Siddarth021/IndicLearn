"""Unit tests evaluating citation attacks and safety boundary auditing (Phase 5.1).

Tests requirement 16:
- [S999] fabricated citation
- Multiple invalid citations ([S998, S999])
- Invalid page number
- Wrong document reference
- Citation to unrelated evidence
- Citation omitted
- Citation attached to unsupported sentence
"""

from __future__ import annotations

from indiclearn.llm.mock import MockLLMProvider
from indiclearn.rag.citation import CitationValidator
from indiclearn.rag.claims import ClaimLevelAuditor
from indiclearn.rag.grounding import GroundingVerifier
from indiclearn.rag.models import EvidenceItem, EvidenceSet


def _create_mock_evidence() -> EvidenceSet:
    return EvidenceSet(
        items=[
            EvidenceItem(
                source_id="S1",
                document_id="signals_unit1.pdf",
                chunk_id="chunk_01",
                text="Convolution is a mathematical operation that combines an input signal x(t) with an impulse response h(t) to produce the output y(t) of an LTI system.",
                page_number=5,
                section="Convolution",
                score=0.92,
                rank=1,
            ),
            EvidenceItem(
                source_id="S2",
                document_id="signals_unit1.pdf",
                chunk_id="chunk_02",
                text="The impulse response completely characterizes continuous-time Linear Time-Invariant systems.",
                page_number=6,
                section="LTI Systems",
                score=0.88,
                rank=2,
            ),
        ],
        query="Explain convolution in LTI systems",
        subject_id="signals_and_systems",
    )


def test_attack_single_fabricated_citation_s999():
    """Verify that a single fabricated citation [S999] is detected and rejected."""
    evidence = _create_mock_evidence()
    verifier = GroundingVerifier()
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="incorrect_citation")
    response = mock.generate("Explain convolution")
    assert "[S999]" in response.text

    # Grounding verifier check
    report = verifier.verify_grounding(response.text, evidence)
    assert report.status == "UNSUPPORTED"
    assert report.check.is_grounded is False
    assert "S999" in report.invalid_citations
    assert report.check.unsupported_citation_count == 1

    # Claim-level audit check
    claim_res = auditor.audit_claims(response.text, evidence)
    assert claim_res.unsupported_citation_claims >= 1
    assert any("S999" in c.contradictions[0] for c in claim_res.claims if c.contradictions)


def test_attack_multiple_invalid_citations():
    """Verify that multiple invalid citations [S998, S999] are all detected."""
    evidence = _create_mock_evidence()
    verifier = GroundingVerifier()
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="multiple_invalid_citations")
    response = mock.generate("Explain convolution")

    report = verifier.verify_grounding(response.text, evidence)
    assert report.status == "UNSUPPORTED"
    assert report.check.is_grounded is False
    assert "S998" in report.invalid_citations
    assert "S999" in report.invalid_citations
    assert report.check.unsupported_citation_count == 2
    assert report.check.citation_validity_rate == 0.0

    claim_res = auditor.audit_claims(response.text, evidence)
    assert claim_res.unsupported_citation_claims >= 1


def test_attack_citation_omitted():
    """Verify that omitting citations for substantive factual content results in zero coverage."""
    evidence = _create_mock_evidence()
    verifier = GroundingVerifier()
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="citation_omitted")
    response = mock.generate("Explain convolution")

    report = verifier.verify_grounding(response.text, evidence)
    assert report.status == "UNSUPPORTED"
    assert report.check.is_grounded is False
    assert report.check.citation_coverage == 0.0
    assert len(report.check.unsupported_claims) > 0

    claim_res = auditor.audit_claims(response.text, evidence)
    assert claim_res.uncited_claims > 0
    assert claim_res.supported_claims == 0


def test_attack_unsupported_substantive_sentence():
    """Verify that partial citation where substantive sentences lack citation is flagged."""
    evidence = _create_mock_evidence()
    verifier = GroundingVerifier(min_coverage_threshold=0.8)
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="unsupported_sentence")
    prompt = """=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: signals_unit1.pdf | Page 5 | Section: Convolution | Chunk: chunk_01
Convolution is a mathematical operation.
===================================
Explain convolution"""
    response = mock.generate(prompt)

    report = verifier.verify_grounding(response.text, evidence)
    # 1 of 2 sentences cited -> coverage is 0.5 < 0.8 threshold
    assert report.status == "PARTIALLY_GROUNDED"
    assert report.check.citation_coverage < 0.8
    assert len(report.check.unsupported_claims) >= 1

    claim_res = auditor.audit_claims(response.text, evidence)
    assert claim_res.uncited_claims >= 1
    assert claim_res.supported_claims >= 1


def test_attack_citation_to_unrelated_evidence():
    """Verify that citing evidence for an unrelated topic (e.g. baking cake vs DSP) is flagged by claim audit."""
    evidence = _create_mock_evidence()
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="unrelated_citation")
    prompt = """=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: signals_unit1.pdf | Page 5 | Section: Convolution | Chunk: chunk_01
Convolution is a mathematical operation.
===================================
Explain convolution"""
    response = mock.generate(prompt)
    assert "[S1]" in response.text
    assert "cake" in response.text.lower()

    # The citation identifier [S1] is syntactically valid in CitationValidator
    validator = CitationValidator()
    _all_c, valid_c, invalid_ids = validator.validate_citations(response.text, evidence)
    assert len(valid_c) == 1
    assert len(invalid_ids) == 0

    # But claim-level audit detects zero topical overlap between cake baking and convolution
    claim_res = auditor.audit_claims(response.text, evidence)
    assert claim_res.unrelated_evidence_claims >= 1
    assert any(c.support_status == "UNRELATED_EVIDENCE" for c in claim_res.claims)


def test_attack_invalid_page_number():
    """Verify that claiming an incorrect page number is detected by claim metadata auditing."""
    evidence = _create_mock_evidence()  # S1 is on page 5
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="invalid_page_citation")
    prompt = """=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: signals_unit1.pdf | Page 5 | Section: Convolution | Chunk: chunk_01
Convolution is commutative and associative.
===================================
Explain convolution"""
    response = mock.generate(prompt)
    assert "Page 999" in response.text

    claim_res = auditor.audit_claims(response.text, evidence)
    assert len(claim_res.metadata_contradictions) >= 1
    assert any("page 999" in c.lower() for c in claim_res.metadata_contradictions)


def test_attack_wrong_document_reference():
    """Verify that referencing the wrong document identifier is flagged by claim metadata auditing."""
    evidence = _create_mock_evidence()  # S1 is from signals_unit1.pdf
    auditor = ClaimLevelAuditor()

    mock = MockLLMProvider(mode="wrong_document_citation")
    prompt = """=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: signals_unit1.pdf | Page 5 | Section: Convolution | Chunk: chunk_01
Convolution combines input signal with impulse response.
===================================
Explain convolution"""
    response = mock.generate(prompt)
    assert "wrong_reference.pdf" in response.text

    claim_res = auditor.audit_claims(response.text, evidence)
    assert len(claim_res.metadata_contradictions) >= 1
    assert any("wrong_reference.pdf" in c.lower() for c in claim_res.metadata_contradictions)
