"""Retrieval package for IndicLearn (Phase 2)."""

from indiclearn.retrieval.base import Reranker, RetrievalResult, Retriever, VectorStore
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.dense import DenseRetriever
from indiclearn.retrieval.exceptions import (
    IndexCorruptError,
    IndexVersionMismatchError,
    RetrievalError,
)
from indiclearn.retrieval.faiss_vector_store import FaissVectorStore
from indiclearn.retrieval.hybrid import HybridRetriever
from indiclearn.retrieval.in_memory_vector_store import InMemoryVectorStore
from indiclearn.retrieval.pipeline import RetrievalPipeline, preprocess_query
from indiclearn.retrieval.tfidf import TFIDFRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "FaissVectorStore",
    "HybridRetriever",
    "InMemoryVectorStore",
    "IndexCorruptError",
    "IndexVersionMismatchError",
    "Reranker",
    "RetrievalError",
    "RetrievalPipeline",
    "RetrievalResult",
    "Retriever",
    "TFIDFRetriever",
    "VectorStore",
    "preprocess_query",
]
