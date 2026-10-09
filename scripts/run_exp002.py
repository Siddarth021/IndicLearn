"""Executable script for EXP-002: Multilingual and Indic Retrieval Evaluation.

Runs comparative benchmarking across lexical baselines, dense embedding models,
and hybrid retrievers on the Signals and Systems multilingual evaluation dataset.
Outputs results to JSON and prints formatted markdown tables.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add src to pythonpath
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from indiclearn.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider
from indiclearn.evaluation.exp002_runner import ExperimentRunner
from indiclearn.retrieval.bm25 import BM25Retriever
from indiclearn.retrieval.dense import DenseRetriever
from indiclearn.retrieval.faiss_vector_store import FaissVectorStore
from indiclearn.retrieval.hybrid import HybridRetriever
from indiclearn.retrieval.tfidf import TFIDFRetriever


def main() -> None:
    dataset_path = ROOT / "data" / "evaluation" / "signals_and_systems_eval.json"
    runner = ExperimentRunner(dataset_path)
    chunks = runner.corpus_chunks
    print(f"Loaded {len(chunks)} corpus chunks and {len(runner.queries)} test queries.")

    results: list[dict] = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Hardware: {hw_name} (device={device})")

    # 1. TF-IDF Baseline
    print("\n--- Evaluating TF-IDF Baseline ---")
    tfidf = TFIDFRetriever()
    tfidf.index(chunks)
    res_tfidf = runner.evaluate_retriever(
        name="TF-IDF Baseline",
        retriever=tfidf,
        model_type="Lexical",
        dimension=None,
        model_size_mb=0.1,
        hardware="CPU",
    )
    results.append(res_tfidf)

    # 2. BM25 Baseline
    print("\n--- Evaluating BM25 Baseline ---")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.index(chunks)
    res_bm25 = runner.evaluate_retriever(
        name="BM25 Baseline",
        retriever=bm25,
        model_type="Lexical",
        dimension=None,
        model_size_mb=0.1,
        hardware="CPU",
    )
    results.append(res_bm25)

    # 3. Model C: all-MiniLM-L6-v2 (English Baseline)
    print("\n--- Evaluating Category C: all-MiniLM-L6-v2 ---")
    try:
        prov_c = SentenceTransformerEmbeddingProvider(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device=device,
        )
        store_c = FaissVectorStore(
            embedding_model=prov_c.model_name, embedding_dimension=prov_c.dimension
        )
        embs_c = prov_c.embed_documents([c.text for c in chunks])
        store_c.add(chunks, embs_c)
        dense_c = DenseRetriever(prov_c, store_c)

        res_c = runner.evaluate_retriever(
            name="all-MiniLM-L6-v2 (EN baseline)",
            retriever=dense_c,
            model_type="Dense (English)",
            dimension=prov_c.dimension,
            model_size_mb=90.0,
            hardware=hw_name,
        )
        results.append(res_c)
    except (RuntimeError, ValueError, OSError, TypeError) as e:
        print(f"Skipping Model C due to error: {e}")

    # 4. Model A: paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)
    print("\n--- Evaluating Category A: paraphrase-multilingual-MiniLM-L12-v2 ---")
    dense_a = None
    try:
        prov_a = SentenceTransformerEmbeddingProvider(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device=device,
        )
        store_a = FaissVectorStore(
            embedding_model=prov_a.model_name, embedding_dimension=prov_a.dimension
        )
        embs_a = prov_a.embed_documents([c.text for c in chunks])
        store_a.add(chunks, embs_a)
        dense_a = DenseRetriever(prov_a, store_a)

        res_a = runner.evaluate_retriever(
            name="paraphrase-multilingual-MiniLM-L12-v2",
            retriever=dense_a,
            model_type="Dense (Multilingual)",
            dimension=prov_a.dimension,
            model_size_mb=470.0,
            hardware=hw_name,
        )
        results.append(res_a)
    except (RuntimeError, ValueError, OSError, TypeError) as e:
        print(f"Skipping Model A due to error: {e}")

    # 5. Model B: intfloat/multilingual-e5-small (High-capacity Multilingual/Indic with asymmetric prefixes)
    print("\n--- Evaluating Category B: intfloat/multilingual-e5-small ---")
    dense_b = None
    try:
        prov_b = SentenceTransformerEmbeddingProvider(
            model_name="intfloat/multilingual-e5-small",
            device=device,
            document_prefix="passage: ",
            query_prefix="query: ",
        )
        store_b = FaissVectorStore(
            embedding_model=prov_b.model_name, embedding_dimension=prov_b.dimension
        )
        embs_b = prov_b.embed_documents([c.text for c in chunks])
        store_b.add(chunks, embs_b)
        dense_b = DenseRetriever(prov_b, store_b)

        res_b = runner.evaluate_retriever(
            name="multilingual-e5-small (asymmetric)",
            retriever=dense_b,
            model_type="Dense (Multilingual/Indic)",
            dimension=prov_b.dimension,
            model_size_mb=470.0,
            hardware=hw_name,
        )
        results.append(res_b)
    except (RuntimeError, ValueError, OSError, TypeError) as e:
        print(f"Skipping Model B due to error: {e}")

    # Note on legacy l3cube-pune/indic-sentence-bert-nli
    results.append({
        "name": "l3cube-pune/indic-sentence-bert-nli",
        "model_type": "Dense (Indic-focused)",
        "dimension": 768,
        "model_size_mb": 440.0,
        "hardware": "N/A",
        "mean_latency_ms": 0.0,
        "status": "BLOCKED / NOT YET EVALUATED",
        "blocked_reason": "Model published with legacy pytorch_model.bin without safetensors; transformers 5.8.1 blocks torch.load on torch < 2.6 per CVE-2025-32434",
        "overall_metrics": {
            "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0, "MRR": 0.0, "nDCG@10": 0.0
        },
        "dimensional_metrics": {
            "en": {"Recall@3": 0.0}, "te": {"Recall@3": 0.0}, "te-Latn": {"Recall@3": 0.0}, "te-en": {"Recall@3": 0.0}
        },
        "failures_count": 0,
        "failures": []
    })

    # 6. Hybrid Retrieval: BM25 + Best Dense (Model A)
    best_dense = dense_a if dense_a is not None else dense_b
    if best_dense is not None:
        print("\n--- Evaluating Hybrid: BM25 + Multilingual Dense (RRF) ---")
        hybrid_rrf = HybridRetriever(
            lexical_retriever=bm25,
            dense_retriever=best_dense,
            strategy="rrf",
            weight_lexical=0.5,
            weight_dense=0.5,
            rrf_k=60,
        )
        res_hybrid_rrf = runner.evaluate_retriever(
            name="Hybrid (BM25 + Multi-Dense RRF)",
            retriever=hybrid_rrf,
            model_type="Hybrid (RRF)",
            dimension=best_dense.embedding_provider.dimension,
            model_size_mb=470.1,
            hardware=hw_name,
        )
        results.append(res_hybrid_rrf)

        print("\n--- Evaluating Hybrid: BM25 + Multilingual Dense (Linear) ---")
        hybrid_linear = HybridRetriever(
            lexical_retriever=bm25,
            dense_retriever=best_dense,
            strategy="linear",
            weight_lexical=0.5,
            weight_dense=0.5,
        )
        res_hybrid_linear = runner.evaluate_retriever(
            name="Hybrid (BM25 + Multi-Dense Linear)",
            retriever=hybrid_linear,
            model_type="Hybrid (Linear MinMax)",
            dimension=best_dense.embedding_provider.dimension,
            model_size_mb=470.1,
            hardware=hw_name,
        )
        results.append(res_hybrid_linear)

    # Save output
    out_file = ROOT / "data" / "evaluation" / "exp002_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved raw results to {out_file}")

    # Print summary table
    print("\n" + "=" * 90)
    print("EXP-002 RETRIEVAL BENCHMARK SUMMARY (OVERALL)")
    print("=" * 90)
    header = f"{'System':<35} | {'R@1':<7} | {'R@3':<7} | {'R@5':<7} | {'MRR':<7} | {'nDCG@10':<7} | {'Lat(ms)':<7}"
    print(header)
    print("-" * 90)
    for r in results:
        om = r["overall_metrics"]
        print(
            f"{r['name']:<35} | {om['Recall@1']:<7.4f} | {om['Recall@3']:<7.4f} | {om['Recall@5']:<7.4f} | {om['MRR']:<7.4f} | {om['nDCG@10']:<7.4f} | {r['mean_latency_ms']:<7.2f}"
        )

    # Print language breakdown table
    print("\n" + "=" * 90)
    print("EXP-002 CROSS-LINGUAL BREAKDOWN (RECALL@3 by Query Formulation)")
    print("=" * 90)
    print(f"{'System':<35} | {'EN -> EN':<10} | {'TE -> EN':<10} | {'Rom-TE -> EN':<12} | {'CodeMix -> EN':<12}")
    print("-" * 90)
    for r in results:
        dm = r["dimensional_metrics"]
        en_r3 = dm.get("en", {}).get("Recall@3", 0.0)
        te_r3 = dm.get("te", {}).get("Recall@3", 0.0)
        rom_r3 = dm.get("te-Latn", {}).get("Recall@3", 0.0)
        mix_r3 = dm.get("te-en", {}).get("Recall@3", 0.0)
        print(
            f"{r['name']:<35} | {en_r3:<10.4f} | {te_r3:<10.4f} | {rom_r3:<12.4f} | {mix_r3:<12.4f}"
        )


if __name__ == "__main__":
    main()
