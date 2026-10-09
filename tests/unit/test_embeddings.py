"""Unit tests for embedding providers and caching (Phase 2)."""

import math

from indiclearn.embeddings.cache import CachedEmbeddingProvider, EmbeddingCache, compute_cache_key
from indiclearn.embeddings.mock import MockEmbeddingProvider


def test_mock_embedding_provider_dimension_and_norm():
    provider = MockEmbeddingProvider(dimension=128)
    assert provider.dimension == 128
    assert provider.model_name == "mock-embedding-v1"

    emb = provider.embed_query("convolution integral")
    assert emb.dimension == 128
    assert emb.normalized is True
    assert len(emb.vector) == 128

    norm = math.sqrt(sum(x * x for x in emb.vector))
    assert math.isclose(norm, 1.0, rel_tol=1e-5)


def test_mock_embedding_provider_determinism_and_batch():
    provider = MockEmbeddingProvider(dimension=64)
    res1 = provider.embed_query("signal and system")
    res2 = provider.embed_query("signal and system")
    assert res1.vector == res2.vector

    batch = provider.embed_documents(["doc one", "doc two", ""])
    assert len(batch) == 3
    assert batch[0].dimension == 64
    assert batch[1].dimension == 64
    assert batch[2].vector == [0.0] * 64


def test_embedding_cache_and_cached_provider(tmp_path):
    cache_file = tmp_path / "embedding_cache.json"
    cache = EmbeddingCache(persistence_path=cache_file)

    call_count = 0

    class CountingMockProvider(MockEmbeddingProvider):
        def embed_documents(self, texts):
            nonlocal call_count
            call_count += len(texts)
            return super().embed_documents(texts)

        def embed_query(self, text):
            nonlocal call_count
            call_count += 1
            return super().embed_query(text)

    inner = CountingMockProvider(dimension=32)
    cached_provider = CachedEmbeddingProvider(inner, cache=cache)

    # First query -> cache miss
    _ = cached_provider.embed_query("test query")
    assert call_count == 1
    assert cache.size() == 1

    # Second query identical -> cache hit, call_count unchanged
    _ = cached_provider.embed_query("test query")
    assert call_count == 1

    # Batch with mixed hits and misses
    _ = cached_provider.embed_documents(["test query", "new document"])
    assert call_count == 2
    assert cache.size() == 2

    # Reload from disk
    new_cache = EmbeddingCache(persistence_path=cache_file)
    assert new_cache.size() == 2
    assert new_cache.contains("test query", inner.model_name, inner.dimension)


def test_cache_key_incorporates_model_and_version():
    k1 = compute_cache_key("text", "model-a", 128, True, "v1")
    k2 = compute_cache_key("text", "model-b", 128, True, "v1")
    k3 = compute_cache_key("text", "model-a", 256, True, "v1")
    k4 = compute_cache_key("text", "model-a", 128, True, "v2")

    assert len({k1, k2, k3, k4}) == 4  # All distinct
