"""Unit tests for VectorStore implementations (InMemory and FAISS) (Phase 2)."""

import pytest

from indiclearn.domain.documents import DocumentChunk
from indiclearn.embeddings.base import EmbeddingResult
from indiclearn.retrieval.exceptions import IndexCorruptError, IndexVersionMismatchError
from indiclearn.retrieval.faiss_vector_store import FaissVectorStore
from indiclearn.retrieval.in_memory_vector_store import InMemoryVectorStore


@pytest.fixture(params=["in_memory", "faiss"])
def vector_store(request):
    dim = 4
    model = "test-model-4d"
    if request.param == "in_memory":
        return InMemoryVectorStore(embedding_model=model, embedding_dimension=dim)
    return FaissVectorStore(embedding_model=model, embedding_dimension=dim)


def test_vector_store_add_search_and_dedupe(vector_store):
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        subject_id="signals",
        page_number=1,
        text="Convolution in LTI systems",
        char_start=0,
        char_end=26,
        concept_ids=["convolution"],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc1",
        subject_id="signals",
        page_number=2,
        text="Fourier transform frequency",
        char_start=27,
        char_end=54,
        concept_ids=["fourier"],
    )

    # Unit vectors
    emb1 = EmbeddingResult(
        vector=[1.0, 0.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )
    emb2 = EmbeddingResult(
        vector=[0.0, 1.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )

    vector_store.add([c1, c2], [emb1, emb2])
    assert vector_store.count() == 2

    # Duplicate add should be ignored
    vector_store.add([c1], [emb1])
    assert vector_store.count() == 2

    # Search for vector closest to emb1
    results = vector_store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1
    assert results[0].metadata["concept_ids"] == ["convolution"]


def test_vector_store_filtering(vector_store):
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc1",
        subject_id="signals",
        page_number=1,
        text="Chunk one",
        char_start=0,
        char_end=9,
        concept_ids=["concept_a"],
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc2",
        subject_id="dsp",
        page_number=1,
        text="Chunk two",
        char_start=0,
        char_end=9,
        concept_ids=["concept_b"],
    )

    emb1 = EmbeddingResult(
        vector=[1.0, 0.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )
    emb2 = EmbeddingResult(
        vector=[0.0, 1.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )
    vector_store.add([c1, c2], [emb1, emb2])

    # Filter by subject
    dsp_res = vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=5, filters={"subject_id": "dsp"})
    assert len(dsp_res) == 1
    assert dsp_res[0].chunk_id == "c2"

    # Filter by concept_id
    concept_res = vector_store.search(
        [1.0, 0.0, 0.0, 0.0], top_k=5, filters={"concept_id": "concept_a"}
    )
    assert len(concept_res) == 1
    assert concept_res[0].chunk_id == "c1"


def test_vector_store_delete(vector_store):
    c1 = DocumentChunk(
        chunk_id="c1",
        document_id="doc_del",
        subject_id="s1",
        page_number=1,
        text="Delete me",
        char_start=0,
        char_end=9,
    )
    c2 = DocumentChunk(
        chunk_id="c2",
        document_id="doc_keep",
        subject_id="s1",
        page_number=1,
        text="Keep me",
        char_start=0,
        char_end=7,
    )

    emb1 = EmbeddingResult(
        vector=[1.0, 0.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )
    emb2 = EmbeddingResult(
        vector=[0.0, 1.0, 0.0, 0.0], model_name="test-model-4d", dimension=4, normalized=True
    )
    vector_store.add([c1, c2], [emb1, emb2])
    assert vector_store.count() == 2

    deleted = vector_store.delete("doc_del")
    assert deleted == 1
    assert vector_store.count() == 1

    remaining = vector_store.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert len(remaining) == 1
    assert remaining[0].chunk_id == "c2"


def test_vector_store_persistence_and_mismatch(tmp_path):
    # Test In-Memory Persistence
    im_store = InMemoryVectorStore(embedding_model="modelA", embedding_dimension=4)
    c = DocumentChunk(
        chunk_id="c1",
        document_id="d1",
        subject_id="s1",
        page_number=1,
        text="Persistence test",
        char_start=0,
        char_end=16,
    )
    emb = EmbeddingResult(vector=[0.5, 0.5, 0.5, 0.5], model_name="modelA", dimension=4)
    im_store.add([c], [emb])

    save_path = tmp_path / "in_memory_index.json"
    im_store.save(save_path)

    loaded_store = InMemoryVectorStore(embedding_model="modelA", embedding_dimension=4)
    loaded_store.load(save_path)
    assert loaded_store.count() == 1

    # Mismatched model must raise IndexVersionMismatchError
    mismatched_store = InMemoryVectorStore(embedding_model="modelB", embedding_dimension=4)
    with pytest.raises(IndexVersionMismatchError):
        mismatched_store.load(save_path)

    # Missing file must raise IndexCorruptError
    with pytest.raises(IndexCorruptError):
        loaded_store.load(tmp_path / "nonexistent.json")

    # Test FAISS Persistence
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()
    faiss_store = FaissVectorStore(embedding_model="modelA", embedding_dimension=4)
    faiss_store.add([c], [emb])
    faiss_store.save(faiss_dir / "index")

    loaded_faiss = FaissVectorStore(embedding_model="modelA", embedding_dimension=4)
    loaded_faiss.load(faiss_dir / "index")
    assert loaded_faiss.count() == 1

    # Mismatched FAISS model must raise IndexVersionMismatchError
    mismatched_faiss = FaissVectorStore(embedding_model="modelB", embedding_dimension=4)
    with pytest.raises(IndexVersionMismatchError):
        mismatched_faiss.load(faiss_dir / "index")
