"""Integration tests for the complete end-to-end RAG Pipeline (Phase 5)."""

from __future__ import annotations

from indiclearn.domain.concepts import Concept
from indiclearn.domain.documents import DocumentChunk
from indiclearn.knowledge_graph.networkx_store import NetworkXGraphStore
from indiclearn.llm.mock import MockLLMProvider
from indiclearn.nlp.pipeline import NLPPipeline
from indiclearn.rag.models import RAGRequest
from indiclearn.rag.pipeline import RAGPipeline
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.hybrid import HybridRetriever
from indiclearn.retrieval.in_memory_vector_store import InMemoryVectorStore


def _build_test_pipeline() -> RAGPipeline:
    subject_id = "signals_and_systems"

    # 1. Knowledge Graph with Academic Concepts
    graph_store = NetworkXGraphStore(default_subject_id=subject_id)
    cpt_conv = Concept(
        concept_id="cpt_signals_and_systems_convolution",
        canonical_name="Convolution",
        subject_id=subject_id,
        concept_type="concept",
        aliases=["కన్వల్యూషన్", "convolution integral"],
        description="Mathematical operation combining two signals",
    )
    cpt_lti = Concept(
        concept_id="cpt_signals_and_systems_lti_system",
        canonical_name="LTI System",
        subject_id=subject_id,
        concept_type="concept",
        aliases=["linear time-invariant system"],
        description="System that is both linear and time-invariant",
    )
    graph_store.add_concept(cpt_conv)
    graph_store.add_concept(cpt_lti)
    from indiclearn.domain.concepts import ConceptRelation, RelationType

    rel = ConceptRelation(
        relation_id="rel_lti_conv",
        source_concept_id="cpt_signals_and_systems_lti_system",
        target_concept_id="cpt_signals_and_systems_convolution",
        relation_type=RelationType.PREREQUISITE_OF,
        subject_id=subject_id,
    )
    graph_store.add_relation(rel)

    # 2. NLP Pipeline
    nlp_pipeline = NLPPipeline.from_graph_store(graph_store)

    # 3. Documents & Chunks
    chunks = [
        DocumentChunk(
            chunk_id="chunk_001",
            document_id="signals_notes.pdf",
            subject_id=subject_id,
            page_number=1,
            section="LTI Systems and Convolution",
            text="Convolution is a mathematical operation combining input signal x(t) and impulse response h(t) to produce output y(t).",
            char_start=0,
            char_end=125,
            concept_ids=["cpt_signals_and_systems_convolution"],
        ),
        DocumentChunk(
            chunk_id="chunk_002",
            document_id="signals_notes.pdf",
            subject_id=subject_id,
            page_number=2,
            section="LTI Properties",
            text="A linear time-invariant (LTI) system satisfies linearity and time-invariance. It is fully characterized by its impulse response.",
            char_start=126,
            char_end=250,
            concept_ids=["cpt_signals_and_systems_lti_system"],
        ),
    ]

    # 4. Hybrid Retriever
    bm25 = BM25Retriever()
    bm25.index(chunks)

    from indiclearn.embeddings.mock import MockEmbeddingProvider
    from indiclearn.retrieval.dense import DenseRetriever

    vector_store = InMemoryVectorStore()
    embedder = MockEmbeddingProvider(dimension=384)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    vector_store.add(chunks, embeddings)
    dense = DenseRetriever(embedding_provider=embedder, vector_store=vector_store)

    retriever = HybridRetriever(lexical_retriever=bm25, dense_retriever=dense, strategy="rrf")

    # 5. LLM Provider
    llm = MockLLMProvider(mode="grounded")

    return RAGPipeline(
        nlp_pipeline=nlp_pipeline,
        retriever=retriever,
        llm_provider=llm,
        graph_store=graph_store,
    )


def test_rag_pipeline_end_to_end_answerable():
    pipeline = _build_test_pipeline()
    request = RAGRequest(
        query="Explain convolution and its relationship with LTI systems",
        subject_id="signals_and_systems",
        requested_language="en",
    )

    response = pipeline.answer(request)

    assert response.is_answerable is True
    assert response.status == "GROUNDED"
    assert len(response.citations) > 0
    assert len(response.evidence.items) > 0

    # Verify complete provenance chain
    first_citation = response.citations[0]
    assert first_citation.is_valid is True
    assert first_citation.source_id.startswith("S")

    evidence_item = response.evidence.get_by_source_id(first_citation.source_id)
    assert evidence_item is not None
    assert evidence_item.chunk_id in ["chunk_001", "chunk_002"]
    assert evidence_item.document_id == "signals_notes.pdf"
    assert evidence_item.page_number in [1, 2]

    # Verify telemetry metadata
    assert response.metadata.model_name == "mock-llm-v1"
    assert response.metadata.requested_language == "en"
    assert response.metadata.latency_ms is not None
    assert response.metadata.retrieval_time_ms is not None


def test_rag_pipeline_insufficient_evidence():
    pipeline = _build_test_pipeline()
    # Query completely outside the syllabus / knowledge graph
    request = RAGRequest(
        query="Explain the architecture of an FIR filter using Kaiser windowing",
        subject_id="signals_and_systems",
        requested_language="en",
    )

    response = pipeline.answer(request)

    assert response.is_answerable is False
    assert response.status in {"INSUFFICIENT_EVIDENCE", "OUT_OF_SCOPE"}
    assert len(response.citations) == 0
    assert (
        "could not find sufficient information" in response.answer.lower()
        or "outside the scope" in response.answer.lower()
    )
    assert response.evidence.insufficient_evidence is True


def test_rag_pipeline_multilingual_telugu():
    pipeline = _build_test_pipeline()
    # Telugu query mapping to convolution
    request = RAGRequest(
        query="కన్వల్యూషన్ అంటే ఏమిటి?",
        subject_id="signals_and_systems",
        requested_language="te",
    )

    response = pipeline.answer(request)

    assert response.is_answerable is True
    assert response.status == "GROUNDED"
    assert len(response.citations) > 0
    assert response.metadata.requested_language == "te"
