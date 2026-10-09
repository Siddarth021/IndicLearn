"""Unit tests for MockLLMProvider simulation modes (Phase 5)."""

from __future__ import annotations

from indiclearn.llm.base import LLMMessage
from indiclearn.llm.mock import MockLLMProvider


def test_mock_llm_grounded_mode_cites_sources():
    mock = MockLLMProvider(mode="grounded")
    prompt = """
=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Document: notes.pdf | Page 1 | Section: Conv | Chunk: c1
Convolution is commutative.

[SOURCE 2] (ID: S2)
Document: notes.pdf | Page 2 | Section: Impulse | Chunk: c2
Impulse response specifies LTI.
===================================

Question: Explain convolution
"""
    resp = mock.generate(prompt)
    assert "[S1]" in resp.text
    assert "[S2]" in resp.text
    assert resp.prompt_tokens is not None
    assert resp.completion_tokens is not None


def test_mock_llm_insufficient_evidence_mode():
    mock = MockLLMProvider(mode="insufficient_evidence")
    resp_en = mock.generate("Question: Explain quantum gravity")
    assert "could not find sufficient information" in resp_en.text.lower()

    resp_te = mock.generate("ప్రశ్న: క్వాంటం గ్రావిటీ గురించి వివరించండి")
    assert "సమాచారం దొరకలేదు" in resp_te.text

    resp_roman = mock.generate("quantum gravity ante enti?")
    assert "information ledu" in resp_roman.text


def test_mock_llm_unsupported_mode():
    mock = MockLLMProvider(mode="unsupported")
    resp = mock.generate("Explain convolution")
    assert "[S" not in resp.text


def test_mock_llm_incorrect_citation_mode():
    mock = MockLLMProvider(mode="incorrect_citation")
    resp = mock.generate("Explain convolution")
    assert "[S999]" in resp.text


def test_mock_llm_missing_citation_mode():
    mock = MockLLMProvider(mode="missing_citation")
    prompt = """
=== RETRIEVED ACADEMIC EVIDENCE ===
[SOURCE 1] (ID: S1)
Text 1
===================================
Question: Explain
"""
    resp = mock.generate(prompt)
    assert "[S1]" in resp.text
    assert "commutative and associative" in resp.text


def test_mock_llm_generate_messages():
    mock = MockLLMProvider(mode="grounded")
    messages = [
        LLMMessage(role="system", content="System instruction"),
        LLMMessage(role="user", content="=== RETRIEVED ACADEMIC EVIDENCE ===\n[SOURCE 1] (ID: S1)\nText\n===================================\nUser question"),
    ]
    resp = mock.generate_messages(messages)
    assert "[S1]" in resp.text
