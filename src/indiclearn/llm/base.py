"""Base protocols and models for Large Language Model providers (Phase 5).

Decouples RAG generation and prompt synthesis from external model vendors
(OpenAI, Anthropic, Gemini, Ollama, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """A single chat message in a conversation or prompt."""

    role: str = Field(description="Role of the author: 'system', 'user', or 'assistant'")
    content: str = Field(description="Textual content of the message")


class LLMResponse(BaseModel):
    """Structured response returned by an LLM provider."""

    text: str = Field(description="Generated answer text")
    model_name: str = Field(description="Model identifier that produced the completion")
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for abstracting language model completions."""

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model."""
        ...

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate completion from a raw text prompt."""
        ...

    def generate_messages(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        """Generate completion from structured chat messages."""
        ...
