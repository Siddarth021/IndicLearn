"""SentenceTransformer embedding provider (Phase 2).

Integrates huggingface/sentence-transformers models behind the EmbeddingProvider protocol.
Supports GPU acceleration, document/query prefixes (for asymmetric models like E5),
batching, and L2 normalization.
"""

from __future__ import annotations

import logging
from typing import Any

from indiclearn.embeddings.base import EmbeddingProvider, EmbeddingResult

logger = logging.getLogger("indiclearn.embeddings")


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using the sentence-transformers library."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        # Determine device
        if device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", self._model_name, self.device)
        self.model = SentenceTransformer(self._model_name, device=self.device)
        # Determine actual dimension
        if hasattr(self.model, "get_embedding_dimension"):
            self._dimension = int(self.model.get_embedding_dimension() or 0)
        else:
            self._dimension = int(self.model.get_sentence_embedding_dimension() or 0)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[EmbeddingResult]:
        """Embed a list of document chunk texts."""
        if not texts:
            return []

        prepared_texts = [f"{self.document_prefix}{t}" if t else " " for t in texts]

        raw_embeddings: Any = self.model.encode(
            prepared_texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )

        results: list[EmbeddingResult] = []
        for i, vec in enumerate(raw_embeddings):
            # If original text was empty, handle cleanly
            if not texts[i].strip():
                vec_list = [0.0] * self._dimension
            else:
                vec_list = [float(x) for x in vec]

            results.append(
                EmbeddingResult(
                    vector=vec_list,
                    model_name=self._model_name,
                    dimension=self._dimension,
                    normalized=self.normalize_embeddings,
                )
            )
        return results

    def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single search query."""
        if not text.strip():
            return EmbeddingResult(
                vector=[0.0] * self._dimension,
                model_name=self._model_name,
                dimension=self._dimension,
                normalized=self.normalize_embeddings,
            )

        prepared = f"{self.query_prefix}{text}"
        raw: Any = self.model.encode(
            [prepared],
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        vec_list = [float(x) for x in raw[0]]
        return EmbeddingResult(
            vector=vec_list,
            model_name=self._model_name,
            dimension=self._dimension,
            normalized=self.normalize_embeddings,
        )
