"""Unit tests for TF-IDF lexical retriever (Phase 2)."""

from indiclearn.domain.documents import DocumentChunk
from indiclearn.retrieval.tfidf import TFIDFRetriever


def test_tfidf_retrieval_and_filtering():
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        subject_id="signals",
        page_number=1,
        text="Convolution integral computes LTI output.",
        char_start=0,
        char_end=41,
        concept_ids=["convolution"],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        subject_id="signals",
        page_number=1,
        text="Laplace transform region of convergence ROC.",
        char_start=0,
        char_end=44,
        concept_ids=["laplace"],
    )

    tfidf = TFIDFRetriever()
    tfidf.index([c1, c2])

    results = tfidf.retrieve("convolution", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0

    # Filter by document_id
    filtered = tfidf.retrieve("convergence", top_k=2, filters={"document_id": "doc1"})
    assert len(filtered) == 0

    filtered_ok = tfidf.retrieve("convergence", top_k=2, filters={"document_id": "doc2"})
    assert len(filtered_ok) == 1
    assert filtered_ok[0].chunk_id == "c2"


def test_tfidf_edge_cases():
    tfidf = TFIDFRetriever()
    assert tfidf.retrieve("query") == []

    chunk = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        subject_id="s1",
        page_number=1,
        text="Basic chunk text.",
        char_start=0,
        char_end=17,
    )
    tfidf.index([chunk])

    assert tfidf.retrieve("") == []
    assert tfidf.retrieve("unseenquerytermxyz") == []
