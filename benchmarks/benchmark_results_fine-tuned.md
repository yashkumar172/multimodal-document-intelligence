# Retrieval Benchmark (fine-tuned embeddings: `finetune/models/minilm-scifact`)

## All questions (27)

| Configuration | HitRate@3 | HitRate@5 | HitRate@10 | MRR@3 | MRR@5 | MRR@10 | NDCG@3 | NDCG@5 | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Dense-only | 0.852 | 0.852 | 0.926 | 0.679 | 0.679 | 0.691 | 0.451 | 0.379 | 0.452 | 14 | 18 |
| Sparse-only (BM25) | 0.815 | 0.889 | 0.926 | 0.759 | 0.776 | 0.781 | 0.475 | 0.411 | 0.432 | 1 | 2 |
| Hybrid (BM25 + Dense, RRF) | 0.889 | 0.926 | 0.926 | 0.833 | 0.843 | 0.843 | 0.540 | 0.475 | 0.485 | 16 | 19 |
| Two-Stage (Hybrid + Reranker) | 0.963 | 0.963 | 0.963 | 0.907 | 0.907 | 0.907 | 0.621 | 0.528 | 0.558 | 744 | 803 |

## Generated from chunks (15 questions)

| Configuration | HitRate@5 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Dense-only | 0.800 | 0.933 | 0.644 | 0.450 |
| Sparse-only (BM25) | 0.867 | 0.933 | 0.842 | 0.485 |
| Hybrid (BM25 + Dense, RRF) | 0.933 | 0.933 | 0.850 | 0.501 |
| Two-Stage (Hybrid + Reranker) | 1.000 | 1.000 | 0.933 | 0.581 |

## Hand-paraphrased (12 questions)

| Configuration | HitRate@5 | HitRate@10 | MRR@10 | NDCG@10 |
|---|---|---|---|---|
| Dense-only | 0.917 | 0.917 | 0.750 | 0.453 |
| Sparse-only (BM25) | 0.917 | 0.917 | 0.704 | 0.366 |
| Hybrid (BM25 + Dense, RRF) | 0.917 | 0.917 | 0.833 | 0.464 |
| Two-Stage (Hybrid + Reranker) | 0.917 | 0.917 | 0.875 | 0.529 |

Latency is retrieval only (query embedding + search + fusion + rerank, no Gemini), over 27 questions x 3 runs after a warm-up, on CPU. Two-Stage reranks the top 25 fused candidates.
