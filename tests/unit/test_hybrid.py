"""Unit tests for HybridRetriever (Phase 2)."""

from indiclearn.domain.documents import DocumentChunk
from indiclearn.embeddings.base import EmbeddingResult
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.dense import DenseRetriever
from indiclearn.retrieval.hybrid import HybridRetriever, min_max_normalize
from indiclearn.retrieval.in_memory_vector_store import InMemoryVectorStore


class StaticEmbeddingProvider:
    model_name = "static-test"
    dimension = 2

    def embed_documents(self, texts):
        results = []
        for t in texts:
            vec = [1.0, 0.0] if "convolution" in t.lower() else [0.0, 1.0]
            results.append(
                EmbeddingResult(
                    vector=vec, model_name=self.model_name, dimension=self.dimension, normalized=True
                )
            )
        return results

    def embed_query(self, text):
        vec = [1.0, 0.0] if "convolution" in text.lower() else [0.0, 1.0]
        return EmbeddingResult(
            vector=vec, model_name=self.model_name, dimension=self.dimension, normalized=True
        )


def test_min_max_normalize():
    assert min_max_normalize([]) == []
    assert min_max_normalize([5.0]) == [1.0]
    assert min_max_normalize([10.0, 10.0]) == [1.0, 1.0]
    normed = min_max_normalize([0.0, 5.0, 10.0])
    assert normed == [0.0, 0.5, 1.0]


def test_hybrid_retriever_rrf_and_linear():
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        subject_id="s1",
        page_number=1,
        text="Convolution integral LTI systems",
        char_start=0,
        char_end=32,
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc1",
        subject_id="s1",
        page_number=2,
        text="Frequency response Bode plot",
        char_start=0,
        char_end=28,
    )

    # BM25 branch
    bm25 = BM25Retriever()
    bm25.index([c1, c2])

    # Dense branch
    emb_provider = StaticEmbeddingProvider()
    vstore = InMemoryVectorStore(embedding_model="static-test", embedding_dimension=2)
    embs = emb_provider.embed_documents([c1.text, c2.text])
    vstore.add([c1, c2], embs)
    dense = DenseRetriever(embedding_provider=emb_provider, vector_store=vstore)

    # 1. RRF Fusion
    hybrid_rrf = HybridRetriever(
        lexical_retriever=bm25,
        dense_retriever=dense,
        strategy="rrf",
    )
    results_rrf = hybrid_rrf.retrieve("convolution", top_k=2)
    assert len(results_rrf) >= 1
    assert results_rrf[0].chunk_id == "c1"
    assert results_rrf[0].rank == 1

    # 2. Linear Normalized Fusion
    hybrid_linear = HybridRetriever(
        lexical_retriever=bm25,
        dense_retriever=dense,
        strategy="linear",
        weight_lexical=0.6,
        weight_dense=0.4,
    )
    results_linear = hybrid_linear.retrieve("convolution", top_k=2)
    assert len(results_linear) >= 1
    assert results_linear[0].chunk_id == "c1"


def test_hybrid_edge_cases():
    bm25 = BM25Retriever()
    bm25.index([])
    vstore = InMemoryVectorStore(embedding_model="static-test", embedding_dimension=2)
    dense = DenseRetriever(
        embedding_provider=StaticEmbeddingProvider(),
        vector_store=vstore,
    )
    hybrid = HybridRetriever(lexical_retriever=bm25, dense_retriever=dense)

    # Empty corpus
    assert hybrid.retrieve("query") == []
    # Empty query string
    assert hybrid.retrieve("") == []
