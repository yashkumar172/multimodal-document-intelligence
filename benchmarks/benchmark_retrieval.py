"""
Retrieval ablation over the eval questions (no Gemini calls):

  1. Dense-only   MiniLM embeddings in ChromaDB
  2. Sparse-only  BM25
  3. Hybrid       BM25 + dense fused with RRF
  4. Two-Stage    Hybrid + cross-encoder rerank (what the app uses)

Reports HitRate@K, MRR@K and NDCG@K for K in {3, 5, 10}, plus p50/p95 retrieval
latency, and writes the tables to benchmarks/benchmark_results.md.

Usage: python benchmarks/benchmark_retrieval.py [--repeats N] [--fused N]
"""

import argparse
import math
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "eval"))

from metrics import hit_rate_at_k, is_relevant, load_eval_sets, mrr_at_k, ndcg_at_k, percentile

from src import embeddings, pipeline, vector_store

KS = (3, 5, 10)
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.md")


def dense_only(question):
    query_vector = embeddings.embed_texts([question])[0]
    return vector_store.query(query_vector, top_k=max(KS))


def sparse_only(question):
    return vector_store.bm25_query(question, top_k=max(KS))


def hybrid(question):
    query_vector = embeddings.embed_texts([question])[0]
    dense_hits = vector_store.query(query_vector, top_k=pipeline.CANDIDATES_PER_RETRIEVER)
    bm25_hits = vector_store.bm25_query(question, top_k=pipeline.CANDIDATES_PER_RETRIEVER)
    fused = vector_store.rrf_merge(
        [bm25_hits, dense_hits], rrf_k=pipeline.RRF_K, top_n=pipeline.FUSED_CANDIDATES
    )
    return fused[:max(KS)]


def two_stage(question):
    hits, _timings = pipeline.retrieve(question, top_k=max(KS))
    return hits


CONFIGS = [
    ("Dense-only", dense_only),
    ("Sparse-only (BM25)", sparse_only),
    ("Hybrid (BM25 + Dense, RRF)", hybrid),
    ("Two-Stage (Hybrid + Reranker)", two_stage),
]


def relevant_chunk_counts():
    """How many indexed chunks each (document, page) has, for the NDCG ideal."""
    metadatas = vector_store.get_collection().get()["metadatas"]
    return Counter((m["document_name"], m["page_number"]) for m in metadatas)


def run(items, repeats):
    counts = relevant_chunk_counts()
    # Warm-up: loads MiniLM, the cross-encoder and the BM25 index, so the first
    # timed query does not include model loading.
    for _, retrieve in CONFIGS:
        retrieve("warm-up query about artificial intelligence")

    results = {}
    for name, retrieve in CONFIGS:
        per_question = []
        latencies = []
        for item in items:
            for _ in range(repeats):
                start = time.perf_counter()
                hits = retrieve(item["question"])
                latencies.append((time.perf_counter() - start) * 1000)
            relevant = [is_relevant(h["metadata"], item) for h in hits]
            total_relevant = counts[(item["document_name"], item["expected_page"])]
            scores = {"set": item["set"]}
            for k in KS:
                scores[f"hit@{k}"] = hit_rate_at_k(relevant, k)
                scores[f"mrr@{k}"] = mrr_at_k(relevant, k)
                scores[f"ndcg@{k}"] = ndcg_at_k(relevant, k, total_relevant)
            per_question.append(scores)
        results[name] = {
            "questions": per_question,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
        }
        print(f"done: {name}")
    return results


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def build_tables(results, items, repeats):
    lines = []
    metric_cols = [f"{m}@{k}" for m in ("hit", "mrr", "ndcg") for k in KS]
    header = ["Configuration"] + [
        c.replace("hit", "HitRate").replace("mrr", "MRR").replace("ndcg", "NDCG") for c in metric_cols
    ] + ["p50 ms", "p95 ms"]
    lines.append(f"## All questions ({len(items)})\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for name, r in results.items():
        values = [mean(r["questions"], c) for c in metric_cols]
        lines.append(
            "| " + " | ".join([name] + [f"{v:.3f}" for v in values] + [f"{r['p50']:.0f}", f"{r['p95']:.0f}"]) + " |"
        )

    for set_name, label in (("generated", "Generated from chunks"), ("paraphrased", "Hand-paraphrased")):
        subset_n = sum(1 for i in items if i["set"] == set_name)
        lines.append(f"\n## {label} ({subset_n} questions)\n")
        lines.append("| Configuration | HitRate@5 | HitRate@10 | MRR@10 | NDCG@10 |")
        lines.append("|---|---|---|---|---|")
        for name, r in results.items():
            rows = [q for q in r["questions"] if q["set"] == set_name]
            lines.append(
                f"| {name} | {mean(rows, 'hit@5'):.3f} | {mean(rows, 'hit@10'):.3f} | "
                f"{mean(rows, 'mrr@10'):.3f} | {mean(rows, 'ndcg@10'):.3f} |"
            )

    lines.append(
        f"\nLatency is retrieval only (query embedding + search + fusion + rerank, no Gemini), "
        f"over {len(items)} questions x {repeats} runs after a warm-up, on CPU. "
        f"Two-Stage reranks the top {pipeline.FUSED_CANDIDATES} fused candidates."
    )
    return "\n".join(lines)


def check_finite(results):
    for name, r in results.items():
        values = [v for q in r["questions"] for k, v in q.items() if k != "set"] + [r["p50"], r["p95"]]
        bad = [v for v in values if not math.isfinite(v)]
        assert not bad, f"{name} has non-finite metrics: {bad}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3, help="timed runs per question")
    parser.add_argument("--fused", type=int, default=pipeline.FUSED_CANDIDATES,
                        help="candidates kept after RRF (and reranked in Two-Stage)")
    args = parser.parse_args()
    pipeline.FUSED_CANDIDATES = args.fused

    if vector_store.get_collection().count() == 0:
        sys.exit("No documents indexed. Upload PDFs in the app (or ingest them) first.")
    items = [
        i for i in load_eval_sets()
        if i["document_name"] in vector_store.list_documents()
    ]
    if not items:
        sys.exit("None of the eval questions' documents are indexed.")

    results = run(items, args.repeats)
    check_finite(results)
    tables = build_tables(results, items, args.repeats)
    print("\n" + tables)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("# Retrieval Benchmark\n\n" + tables + "\n")
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
