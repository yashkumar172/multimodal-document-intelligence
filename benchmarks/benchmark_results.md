# Retrieval Benchmark

## All questions (27)

| Configuration | HitRate@3 | HitRate@5 | HitRate@10 | MRR@3 | MRR@5 | MRR@10 | NDCG@3 | NDCG@5 | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense-only | 0.778 | 0.963 | 0.963 | 0.636 | 0.677 | 0.677 | 0.436 | 0.396 | 0.448 | 16 | 19 |
| Sparse-only (BM25) | 0.815 | 0.889 | 0.926 | 0.759 | 0.776 | 0.781 | 0.475 | 0.411 | 0.432 | 1 | 2 |
| Hybrid (BM25 + Dense, RRF) | 0.889 | 0.926 | 0.963 | 0.833 | 0.843 | 0.847 | 0.549 | 0.470 | 0.498 | 19 | 23 |
| Two-Stage (Hybrid + Reranker) | 0.963 | 0.963 | 0.963 | 0.907 | 0.907 | 0.907 | 0.621 | 0.533 | 0.554 | 828 | 909 |

## Generated from chunks (15 questions)

| Configuration | HitRate@5 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Dense-only | 1.000 | 1.000 | 0.666 | 0.469 |
| Sparse-only (BM25) | 0.867 | 0.933 | 0.842 | 0.485 |
| Hybrid (BM25 + Dense, RRF) | 0.933 | 1.000 | 0.857 | 0.516 |
| Two-Stage (Hybrid + Reranker) | 1.000 | 1.000 | 0.933 | 0.575 |

## Hand-paraphrased (12 questions)

| Configuration | HitRate@5 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Dense-only | 0.917 | 0.917 | 0.690 | 0.422 |
| Sparse-only (BM25) | 0.917 | 0.917 | 0.704 | 0.366 |
| Hybrid (BM25 + Dense, RRF) | 0.917 | 0.917 | 0.833 | 0.475 |
| Two-Stage (Hybrid + Reranker) | 0.917 | 0.917 | 0.875 | 0.527 |

Latency is retrieval only (query embedding + search + fusion + rerank, no Gemini), over 27 questions x 3 runs after a warm-up, on CPU. Two-Stage reranks the top 25 fused candidates.
