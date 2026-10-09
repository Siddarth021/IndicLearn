"""Typed exceptions for retrieval and vector storage (Phase 2)."""

from __future__ import annotations


class RetrievalError(Exception):
    """Base exception for retrieval errors."""


class IndexVersionMismatchError(RetrievalError):
    """Raised when an index persisted with one model or dimension is loaded with incompatible settings."""

    def __init__(
        self,
        index_model: str,
        index_dim: int,
        expected_model: str,
        expected_dim: int,
    ) -> None:
        msg = (
            f"Index version mismatch: persisted index was built with "
            f"model='{index_model}' (dim={index_dim}), but current configuration "
            f"expects model='{expected_model}' (dim={expected_dim}). "
            f"Re-indexing is required."
        )
        super().__init__(msg)
        self.index_model = index_model
        self.index_dim = index_dim
        self.expected_model = expected_model
        self.expected_dim = expected_dim


class IndexCorruptError(RetrievalError):
    """Raised when an index file or sidecar metadata is missing or corrupted."""
