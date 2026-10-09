"""LLM Provider package (Phase 5 & 5.1)."""

from __future__ import annotations

import os
from typing import Any

from indiclearn.llm.base import LLMMessage, LLMProvider, LLMResponse
from indiclearn.llm.mock import MockLLMProvider
from indiclearn.llm.providers.openai_compatible import OpenAICompatibleProvider


def get_llm_provider(provider_type: str | None = None, **kwargs: Any) -> LLMProvider:
    """Factory creating an LLMProvider based on explicit name or environment variable."""
    p_type = (provider_type or os.getenv("LLM_PROVIDER") or "mock").lower().strip()
    if p_type in ("real", "openai", "openai_compatible", "vllm", "ollama"):
        return OpenAICompatibleProvider(**kwargs)
    return MockLLMProvider(**kwargs)


__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "get_llm_provider",
]
