"""Deterministic Mock LLM Provider for offline unit testing and pipeline validation (Phase 5).

Does NOT simulate a real cognitive language model. Allows explicit, deterministic
simulation of:
- Grounded answers citing supplied evidence [S1], [S2]
- Insufficient evidence notifications
- Unsupported answers (claims without citations)
- Incorrect/fabricated citations ([S999])
- Missing/partial citations
"""

from __future__ import annotations

import re
from typing import Any

from indiclearn.llm.base import LLMMessage, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Configurable mock LLM provider for deterministic RAG testing."""

    def __init__(
        self,
        mode: str = "grounded",
        model_name: str = "mock-llm-v1",
        canned_responses: dict[str, str] | None = None,
    ) -> None:
        """Initialize mock provider.

        Args:
            mode: One of 'grounded', 'insufficient_evidence', 'unsupported',
                  'incorrect_citation', 'missing_citation', 'custom'.
            model_name: Mock model identifier.
            canned_responses: Explicit prompt substring to response mapping.
        """
        self.mode = mode
        self._model_name = model_name
        self.canned_responses = canned_responses or {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def _extract_source_ids(self, prompt: str) -> list[str]:
        """Extract available source IDs strictly from the evidence section of the prompt."""
        evidence_start = prompt.find("=== RETRIEVED ACADEMIC EVIDENCE ===")
        search_text = prompt[evidence_start:] if evidence_start != -1 else prompt
        return list(dict.fromkeys(re.findall(r"\b(S\d+)\b", search_text)))

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate deterministic response based on configured mode and extracted context."""
        # 1. Check custom canned responses first
        for key, resp in self.canned_responses.items():
            if key.lower() in prompt.lower():
                return LLMResponse(
                    text=resp,
                    model_name=self._model_name,
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(resp.split()),
                    metadata={"mode": "custom", "matched_key": key},
                )

        available_sources = self._extract_source_ids(prompt)
        s1 = available_sources[0] if available_sources else "S1"
        s2 = available_sources[1] if len(available_sources) > 1 else s1

        # 2. Handle modes
        if self.mode == "insufficient_evidence":
            # Check if query asks in Telugu or English
            if any(0x0C00 <= ord(c) <= 0x0C7F for c in prompt):
                text = "అందించిన విద్యా విషయాలలో ఈ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి తగినంత సమాచారం దొరకలేదు."
            elif "ante enti" in prompt.lower() or "cheyyi" in prompt.lower():
                text = "Upload chesina academic material lo e prasnaku saripoyina information ledu."
            else:
                text = "I could not find sufficient information in the uploaded academic materials to answer this question reliably."

        elif self.mode == "unsupported" or self.mode == "citation_omitted":
            text = (
                "Convolution is a fundamental mathematical operation used extensively across DSP, "
                "computer vision, and neural networks to filter audio and image matrices."
            )

        elif self.mode == "incorrect_citation":
            text = "According to the course notes [S999], convolution is an integral operation."

        elif self.mode == "multiple_invalid_citations":
            text = (
                "According to the lecture [S999], convolution is commutative, and per the appendix "
                "[S998], impulse response completely specifies LTI systems."
            )

        elif self.mode == "unrelated_citation":
            text = (
                f"Baking an authentic Italian chocolate cake requires two cups of flour, cocoa powder, "
                f"and whole milk [{s1}]."
            )

        elif self.mode == "invalid_page_citation":
            text = (
                f"Convolution is commutative and associative as proved on Page 999 of the textbook [{s1}]."
            )

        elif self.mode == "wrong_document_citation":
            text = (
                f"According to wrong_reference.pdf, convolution combines input signal with impulse response [{s1}]."
            )

        elif self.mode == "missing_citation" or self.mode == "unsupported_sentence":
            text = (
                f"Convolution combines the input signal with the impulse response [{s1}]. "
                "It is also commutative and associative across all linear continuous time domains."
            )

        else:  # 'grounded' (default)
            # Detect language context from prompt
            is_telugu_script = any(0x0C00 <= ord(c) <= 0x0C7F for c in prompt)
            is_romanized = "ante enti" in prompt.lower() or "cheyyi" in prompt.lower()

            if is_telugu_script:
                text = (
                    f"కన్వల్యూషన్ అనేది ఇన్పుట్ సిగ్నల్ మరియు ఇంపల్స్ రెస్పాన్స్ కలయిక ద్వారా "
                    f"LTI సిస్టమ్ అవుట్‌పుట్‌ను నిర్ణయించే గణిత ప్రక్రియ [{s1}]. "
                    f"ఇది లీనియారిటీ మరియు టైమ్-ఇన్వేరియన్స్ లక్షణాలను కలిగి ఉంటుంది [{s2}]."
                )
            elif is_romanized:
                text = (
                    f"Convolution anedi input signal mariyu impulse response ni combine chesi "
                    f"LTI system output ni compute chese mathematical operation [{s1}]. "
                    f"Idi continuous-time systems lo chala important [{s2}]."
                )
            else:
                text = (
                    f"Convolution is a fundamental mathematical operation combining an input signal with "
                    f"the system's impulse response to determine the output of an LTI system [{s1}]. "
                    f"For linear time-invariant systems, this completely characterizes the continuous-time response [{s2}]."
                )

        return LLMResponse(
            text=text,
            model_name=self._model_name,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            metadata={"mode": self.mode, "sources_detected": available_sources},
        )

    def generate_messages(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate response by concatenating messages."""
        full_prompt = "\n\n".join(f"{m.role}: {m.content}" for m in messages)
        return self.generate(full_prompt, **kwargs)
