"""Generic OpenAI-compatible LLM Provider (Phase 5).

Supports OpenAI, Ollama, vLLM, LocalAI, and other OpenAI-compatible chat completion endpoints.
Uses standard library urllib to avoid mandatory external vendor SDK dependencies.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from indiclearn.llm.base import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """LLM Provider for OpenAI-compatible REST endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("INDICLEARN_LLM__API_KEY")
        )
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self._model_name = (
            model_name
            or os.getenv("LLM_MODEL")
            or os.getenv("INDICLEARN_LLM__MODEL")
            or "gpt-4o-mini"
        )
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        """Create provider from environment variables."""
        return cls()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_configured(self) -> bool:
        """Return True if credentials or local endpoint are available."""
        return bool(self.api_key or "localhost" in self.base_url or "127.0.0.1" in self.base_url)

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        messages = [LLMMessage(role="user", content=prompt)]
        return self.generate_messages(messages, **kwargs)

    def generate_messages(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise ValueError(
                "API key is required for OpenAICompatibleProvider when not targeting a local endpoint. "
                "Set OPENAI_API_KEY or INDICLEARN_LLM__API_KEY."
            )

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self._model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", 0.0),
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))

            choice = resp_json.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content", "")
            usage = resp_json.get("usage", {})

            return LLMResponse(
                text=text,
                model_name=self._model_name,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                metadata={"raw_response": resp_json},
            )
        except urllib.error.URLError as e:
            logger.error(f"Failed to communicate with LLM endpoint {endpoint}: {e}")
            raise RuntimeError(f"LLM generation failed: {e}") from e
