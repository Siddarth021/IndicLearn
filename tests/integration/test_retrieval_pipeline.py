"""Integration tests connecting IngestionService to RetrievalPipeline (Phase 2)."""

from indiclearn.embeddings.mock import MockEmbeddingProvider
from indiclearn.ingestion.service import IngestionService
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.dense import DenseRetriever
from indiclearn.retrieval.faiss_vector_store import FaissVectorStore
from indiclearn.retrieval.hybrid import HybridRetriever
from indiclearn.retrieval.pipeline import RetrievalPipeline, preprocess_query


def test_preprocess_query():
    assert preprocess_query("  convolution  ") == "convolution"
    assert preprocess_query("") == ""


def test_end_to_end_ingestion_and_retrieval(service: IngestionService, tmp_path):
    # 1. Ingest text files
    doc1_path = tmp_path / "doc1.txt"
    doc1_path.write_text(
        "Continuous-time convolution is an integral operation characterizing LTI systems.",
        encoding="utf-8",
    )
    doc2_path = tmp_path / "doc2.txt"
    doc2_path.write_text(
        "Fourier transform represents continuous signals in the frequency domain with spectrum.",
        encoding="utf-8",
    )
    res1 = service.ingest(doc1_path, subject_id="signals")
    res2 = service.ingest(doc2_path, subject_id="signals")

    chunks = res1.chunks + res2.chunks
    assert len(chunks) == 2

    # 2. Build BM25 index
    bm25 = BM25Retriever()
    bm25.index(chunks)

    # 3. Build FAISS vector index
    provider = MockEmbeddingProvider(model_name="mock-integration", dimension=64)
    vstore = FaissVectorStore(embedding_model="mock-integration", embedding_dimension=64)
    embeddings = provider.embed_documents([c.text for c in chunks])
    vstore.add(chunks, embeddings)
    dense = DenseRetriever(embedding_provider=provider, vector_store=vstore)

    # 4. Lexical pipeline execution
    lexical_pipeline = RetrievalPipeline(retriever=bm25)
    lex_results = lexical_pipeline.run("convolution in LTI systems", top_k=2)
    assert len(lex_results) >= 1
    assert lex_results[0].chunk_id == chunks[0].chunk_id
    assert lex_results[0].rank == 1
    assert lex_results[0].metadata["subject_id"] == "signals"

    # 5. Hybrid Retriever
    hybrid = HybridRetriever(lexical_retriever=bm25, dense_retriever=dense, strategy="rrf", weight_lexical=0.8, weight_dense=0.2)
    hybrid_pipeline = RetrievalPipeline(retriever=hybrid)
    results = hybrid_pipeline.run("convolution in LTI systems", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].rank == 1

    # Test filtering through pipeline
    filtered_results = hybrid_pipeline.run(
        "convolution",
        top_k=2,
        filters={"document_id": "doc_nonexistent"},
    )
    assert len(filtered_results) == 0

    # Test filtering to doc1 specifically returns chunks[0]
    doc1_results = hybrid_pipeline.run(
        "convolution",
        top_k=2,
        filters={"document_id": chunks[0].document_id},
    )
    assert len(doc1_results) >= 1
    assert doc1_results[0].chunk_id == chunks[0].chunk_id
