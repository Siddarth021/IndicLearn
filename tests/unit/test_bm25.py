"""Unit tests for BM25 lexical retriever (Phase 2)."""

from indiclearn.domain.documents import DocumentChunk
from indiclearn.retrieval.bm25 import BM25Retriever, tokenize


def test_bm25_tokenize_indic_and_latin():
    text = "కన్వల్యూషన్ convolution 123! LTI-system"
    tokens = tokenize(text)
    assert "కన్వల్యూషన్" in tokens
    assert "convolution" in tokens
    assert "123" in tokens
    assert "lti" in tokens
    assert "system" in tokens


def test_bm25_retrieval_ranking_and_filters():
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        subject_id="signals",
        page_number=1,
        text="Convolution integral determines the output of an LTI system.",
        char_start=0,
        char_end=60,
        concept_ids=["convolution"],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc1",
        subject_id="signals",
        page_number=2,
        text="Continuous Fourier transform frequency spectrum analysis.",
        char_start=61,
        char_end=120,
        concept_ids=["fourier"],
    )
    c3 = DocumentChunk(
        chunk_id="c3",
        document_id="doc2",
        subject_id="signals",
        page_number=1,
        text="Nyquist sampling theorem avoids aliasing distortion.",
        char_start=0,
        char_end=52,
        concept_ids=["sampling"],
    )

    bm25 = BM25Retriever()
    bm25.index([c1, c2, c3])

    # Basic search
    results = bm25.retrieve("convolution", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1
    assert results[0].score > 0

    # Filter by document_id
    filtered = bm25.retrieve("sampling", top_k=2, filters={"document_id": "doc1"})
    assert len(filtered) == 0

    filtered_ok = bm25.retrieve("sampling", top_k=2, filters={"document_id": "doc2"})
    assert len(filtered_ok) == 1
    assert filtered_ok[0].chunk_id == "c3"

    # Filter by concept_id
    filtered_concept = bm25.retrieve("frequency", top_k=2, filters={"concept_id": "fourier"})
    assert len(filtered_concept) == 1
    assert filtered_concept[0].chunk_id == "c2"


def test_bm25_edge_cases():
    bm25 = BM25Retriever()
    # Empty index
    assert bm25.retrieve("anything") == []

    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        subject_id="s1",
        page_number=1,
        text="Sample text",
        char_start=0,
        char_end=11,
    )
    bm25.index([chunk])

    # Empty query
    assert bm25.retrieve("") == []
    assert bm25.retrieve("   ") == []

    # Out of vocabulary query
    assert bm25.retrieve("nonexistentterm12345") == []
