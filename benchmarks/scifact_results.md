# Retrieval Benchmark: BEIR SciFact

## Base model

Model: `sentence-transformers/all-MiniLM-L6-v2`. 300 test queries, 5183 documents split into 12198 chunks (200 tokens, 40 overlap). Document score = best chunk.

| Configuration | nDCG@10 [95% CI] | Recall@10 | MRR@10 [95% CI] | p50 ms | p95 ms |
|---|---|---|---|---|---|
| Dense-only | 0.665 [0.620, 0.712] | 0.814 | 0.623 [0.576, 0.674] | 18 | 22 |
| Sparse-only (BM25) | 0.599 [0.554, 0.645] | 0.736 | 0.559 [0.512, 0.607] | 57 | 107 |
| Hybrid (BM25 + Dense, RRF) | 0.671 [0.628, 0.716] | 0.816 | 0.632 [0.587, 0.678] | 74 | 127 |
| Two-Stage (Hybrid + Reranker) | 0.710 [0.670, 0.752] | 0.844 | 0.677 [0.633, 0.723] | 826 | 957 |

### nDCG@10 difference vs Dense-only (paired bootstrap, 95% CI)

| Configuration | Δ nDCG@10 [95% CI] |
|---|---|
| Sparse-only (BM25) | -0.066 [-0.108, -0.027] |
| Hybrid (BM25 + Dense, RRF) | +0.006 [-0.022, +0.031] |
| Two-Stage (Hybrid + Reranker) | +0.044 [+0.012, +0.074] |

Latency is retrieval only (query embedding + search + fusion + rerank), one timed run per query after a warm-up, on CPU. Two-Stage reranks the top 25 fused chunks. CIs: 1000 bootstrap resamples over queries, seed 42.
