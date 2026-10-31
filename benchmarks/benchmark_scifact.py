"""
Retrieval benchmark on BEIR SciFact (5,183 abstracts, 300 test claims), using
this project's own retrieval pieces (no Gemini calls):

  1. Dense-only   MiniLM embeddings in ChromaDB
  2. Sparse-only  BM25 (the app's tokenizer and BM25Index)
  3. Hybrid       BM25 + dense fused with RRF (k=60, 30 candidates each, keep 25)
  4. Two-Stage    Hybrid + the app's cross-encoder reranker

Documents (title + abstract) are split with the app's chunker (200 tokens, 40
overlap). Every retriever works on chunks exactly as in the app; the ranked
chunks are then collapsed to documents, each document scored by its best chunk
(its first appearance in the ranking). Metrics are computed per document.

Reports nDCG@10, Recall@10 and MRR@10 with 95% bootstrap confidence intervals
(1,000 resamples over queries), plus p50/p95 latency per query, and writes the
tables to benchmarks/scifact_results.md.

The dataset is downloaded once to benchmarks/beir_cache/ (gitignored), which
also holds the ChromaDB index, so the app's chroma_db/ is never touched.

Usage: python benchmarks/benchmark_scifact.py [--model PATH_OR_NAME] [--label NAME]
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "eval"))

import chromadb
import torch
from metrics import mrr_at_k, ndcg_at_k, percentile
from sentence_transformers import SentenceTransformer

from src import chunking, embeddings, pipeline, reranker
from src.vector_store import BM25Index, rrf_merge

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BENCH_DIR, "beir_cache")
DATA_DIR = os.path.join(CACHE_DIR, "scifact")
CHROMA_DIR = os.path.join(CACHE_DIR, "chroma")
RESULTS_PATH = os.path.join(BENCH_DIR, "scifact_results.md")
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

K = 10
SEED = 42
BOOTSTRAP_RESAMPLES = 1000


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data

def download_scifact():
    if os.path.exists(os.path.join(DATA_DIR, "corpus.jsonl")):
        return
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "scifact.zip")
    print(f"Downloading {DATASET_URL} ...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(CACHE_DIR)


def load_scifact(split="test"):
    """Returns (corpus {doc_id: text}, queries {qid: text}, qrels {qid: {doc_id: score}})."""
    download_scifact()
    corpus = {}
    with open(os.path.join(DATA_DIR, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            corpus[d["_id"]] = f"{d['title']}\n{d['text']}".strip()

    qrels = {}
    with open(os.path.join(DATA_DIR, "qrels", f"{split}.tsv"), encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            qid, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(qid, {})[doc_id] = int(score)

    queries = {}
    with open(os.path.join(DATA_DIR, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["_id"] in qrels:
                queries[q["_id"]] = q["text"]
    return corpus, queries, qrels


def build_chunks(corpus):
    """Chunk each document with the app's chunker; each document is one 'page'."""
    chunks = []
    for doc_id, text in corpus.items():
        for c in chunking.chunk_pages([(1, text)], doc_id):
            chunks.append({"id": c["chunk_id"], "content": c["content"], "doc_id": doc_id})
    return chunks


# ---------------------------------------------------------------- index

class SciFactIndex:
    """Dense (ChromaDB, separate persist dir) + BM25 indexes over SciFact chunks."""

    def __init__(self, chunks, model_name):
        self.model = SentenceTransformer(model_name, device="cpu")
        ids = [c["id"] for c in chunks]
        texts = [c["content"] for c in chunks]
        metas = [{"doc_id": c["doc_id"]} for c in chunks]
        self.bm25 = BM25Index(ids, texts, metas)

        # One collection per model, so a fine-tuned model gets its own index.
        tag = hashlib.sha1(os.path.abspath(model_name).encode() if os.path.exists(model_name)
                           else model_name.encode()).hexdigest()[:12]
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_or_create_collection(f"scifact_{tag}")
        if self.collection.count() != len(chunks):
            client.delete_collection(self.collection.name)
            self.collection = client.get_or_create_collection(f"scifact_{tag}")
            print(f"Embedding {len(chunks)} chunks with {model_name} on CPU (one-off, cached) ...")
            start = time.perf_counter()
            vectors = self.model.encode(texts, batch_size=64, show_progress_bar=True)
            print(f"  embedded in {time.perf_counter() - start:.0f} s")
            batch = 5000  # below ChromaDB's max batch size
            for i in range(0, len(ids), batch):
                self.collection.add(
                    ids=ids[i:i + batch],
                    embeddings=vectors[i:i + batch].tolist(),
                    documents=texts[i:i + batch],
                    metadatas=metas[i:i + batch],
                )

    def dense(self, question, top_k):
        vector = self.model.encode([question], show_progress_bar=False)[0].tolist()
        r = self.collection.query(query_embeddings=[vector], n_results=top_k)
        return [
            {"id": i, "content": d, "metadata": m}
            for i, d, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])
        ]

    def sparse(self, question, top_k):
        return self.bm25.search(question, top_k=top_k)


def to_documents(hits, k=K):
    """Collapse ranked chunks to ranked documents; a document ranks at its best chunk."""
    docs = []
    for h in hits:
        doc_id = h["metadata"]["doc_id"]
        if doc_id not in docs:
            docs.append(doc_id)
            if len(docs) == k:
                break
    return docs


def make_configs(index, label=""):
    n = pipeline.CANDIDATES_PER_RETRIEVER

    def dense_only(q):
        return to_documents(index.dense(q, n))

    def sparse_only(q):
        return to_documents(index.sparse(q, n))

    def fused(q):
        return rrf_merge([index.sparse(q, n), index.dense(q, n)],
                         rrf_k=pipeline.RRF_K, top_n=pipeline.FUSED_CANDIDATES)

    def hybrid(q):
        return to_documents(fused(q))

    def two_stage(q):
        # The reranker scores all 25 fused chunks (the app keeps the best 10
        # chunks; here the reranked order is collapsed to the top 10 documents).
        candidates = fused(q)
        return to_documents(reranker.rerank(q, candidates, top_n=len(candidates)))

    suffix = f" ({label})" if label else ""
    return [
        (f"Dense-only{suffix}", dense_only),
        ("Sparse-only (BM25)", sparse_only),
        (f"Hybrid (BM25 + Dense, RRF){suffix}", hybrid),
        (f"Two-Stage (Hybrid + Reranker){suffix}", two_stage),
    ]


# ---------------------------------------------------------------- metrics

def recall_at_k(relevant, k, total_relevant):
    return sum(relevant[:k]) / total_relevant if total_relevant else 0.0


def bootstrap_ci(values, resamples=BOOTSTRAP_RESAMPLES, seed=SEED):
    """95% percentile bootstrap CI of the mean, resampling queries with replacement."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(configs, queries, qrels):
    # Warm-up loads MiniLM, the cross-encoder and fills caches, so the first
    # timed query does not include model loading.
    for _, retrieve in configs:
        retrieve("warm-up query about protein expression in cells")

    results = {}
    for name, retrieve in configs:
        per_query = {"ndcg": [], "recall": [], "mrr": [], "docs": []}
        latencies = []
        start_all = time.perf_counter()
        for qid, question in queries.items():
            start = time.perf_counter()
            docs = retrieve(question)
            latencies.append((time.perf_counter() - start) * 1000)
            gold = qrels[qid]
            relevant = [d in gold for d in docs]
            per_query["ndcg"].append(ndcg_at_k(relevant, K, len(gold)))
            per_query["recall"].append(recall_at_k(relevant, K, len(gold)))
            per_query["mrr"].append(mrr_at_k(relevant, K))
            per_query["docs"].append(len(docs))
        results[name] = {**per_query, "p50": percentile(latencies, 50), "p95": percentile(latencies, 95)}
        print(f"done: {name} in {time.perf_counter() - start_all:.0f} s "
              f"(avg {np.mean(per_query['docs']):.1f} documents returned)")
    return results


def check_finite(results):
    for name, r in results.items():
        values = r["ndcg"] + r["recall"] + r["mrr"] + [r["p50"], r["p95"]]
        assert all(math.isfinite(v) for v in values), f"{name} has non-finite metrics"


# ---------------------------------------------------------------- report

def build_tables(results, n_queries, n_docs, n_chunks, model_name):
    lines = [
        f"Model: `{model_name}`. {n_queries} test queries, {n_docs} documents split into "
        f"{n_chunks} chunks (200 tokens, 40 overlap). Document score = best chunk.\n",
        "| Configuration | nDCG@10 [95% CI] | Recall@10 | MRR@10 [95% CI] | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        ndcg, mrr = np.mean(r["ndcg"]), np.mean(r["mrr"])
        n_lo, n_hi = bootstrap_ci(r["ndcg"])
        m_lo, m_hi = bootstrap_ci(r["mrr"])
        lines.append(
            f"| {name} | {ndcg:.3f} [{n_lo:.3f}, {n_hi:.3f}] | {np.mean(r['recall']):.3f} | "
            f"{mrr:.3f} [{m_lo:.3f}, {m_hi:.3f}] | {r['p50']:.0f} | {r['p95']:.0f} |"
        )

    # Paired bootstrap: the same resampled queries for both configurations.
    names = list(results)
    base = names[0]
    lines += [
        f"\n### nDCG@10 difference vs {base} (paired bootstrap, 95% CI)\n",
        "| Configuration | Δ nDCG@10 [95% CI] |",
        "|---|---|",
    ]
    for name in names[1:]:
        diff = np.array(results[name]["ndcg"]) - np.array(results[base]["ndcg"])
        lo, hi = bootstrap_ci(diff)
        lines.append(f"| {name} | {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")

    lines.append(
        f"\nLatency is retrieval only (query embedding + search + fusion + rerank), one timed "
        f"run per query after a warm-up, on CPU. Two-Stage reranks the top "
        f"{pipeline.FUSED_CANDIDATES} fused chunks. CIs: {BOOTSTRAP_RESAMPLES} bootstrap "
        f"resamples over queries, seed {SEED}."
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=embeddings.MODEL_NAME,
                        help="embedding model name or local path (e.g. a fine-tuned model)")
    parser.add_argument("--label", default="",
                        help='suffix for the dense rows, e.g. "fine-tuned"')
    parser.add_argument("--limit", type=int, default=0, help="only the first N queries (smoke test)")
    args = parser.parse_args()
    set_seeds()
    sys.stdout.reconfigure(encoding="utf-8")  # the Windows console defaults to cp1252

    corpus, queries, qrels = load_scifact("test")
    if args.limit:
        queries = dict(list(queries.items())[:args.limit])
    chunks = build_chunks(corpus)
    print(f"{len(corpus)} documents -> {len(chunks)} chunks; {len(queries)} test queries")

    index = SciFactIndex(chunks, args.model)
    results = evaluate(make_configs(index, args.label), queries, qrels)
    check_finite(results)
    tables = build_tables(results, len(queries), len(corpus), len(chunks), args.model)
    print("\n" + tables)

    if args.limit:
        print("\n--limit set: results not written.")
        return
    section = f"## {args.label or 'Base model'}\n\n{tables}\n"
    with open(RESULTS_PATH, "w" if not args.label else "a", encoding="utf-8") as f:
        if not args.label:
            f.write("# Retrieval Benchmark: BEIR SciFact\n\n")
        f.write(("\n" if args.label else "") + section)
    print(f"\nWritten to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
