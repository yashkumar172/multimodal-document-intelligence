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

## fine-tuned

Model: `finetune/models/minilm-scifact`. 300 test queries, 5183 documents split into 12198 chunks (200 tokens, 40 overlap). Document score = best chunk.

| Configuration | nDCG@10 [95% CI] | Recall@10 | MRR@10 [95% CI] | p50 ms | p95 ms |
|---|---|---|---|---|---|
| Dense-only (fine-tuned) | 0.721 [0.681, 0.764] | 0.858 | 0.681 [0.637, 0.728] | 15 | 17 |
| Hybrid (BM25 + Dense, RRF) (fine-tuned) | 0.719 [0.677, 0.761] | 0.844 | 0.689 [0.640, 0.738] | 74 | 130 |
| Two-Stage (Hybrid + Reranker) (fine-tuned) | 0.719 [0.680, 0.761] | 0.863 | 0.684 [0.640, 0.731] | 734 | 846 |

### nDCG@10 difference vs Dense-only (fine-tuned) (paired bootstrap, 95% CI)

| Configuration | Δ nDCG@10 [95% CI] |
|---|---|
| Hybrid (BM25 + Dense, RRF) (fine-tuned) | -0.001 [-0.028, +0.024] |
| Two-Stage (Hybrid + Reranker) (fine-tuned) | -0.002 [-0.030, +0.026] |

Latency is retrieval only (query embedding + search + fusion + rerank), one timed run per query after a warm-up, on CPU. Two-Stage reranks the top 25 fused chunks. CIs: 1000 bootstrap resamples over queries, seed 42.

### Difference vs the base model (paired bootstrap, 95% CI)

| Configuration | Δ nDCG@10 [95% CI] | Δ MRR@10 [95% CI] |
|---|---|---|
| Dense-only (fine-tuned) vs Dense-only | +0.055 [+0.035, +0.077] | +0.057 [+0.033, +0.082] |
| Hybrid (BM25 + Dense, RRF) (fine-tuned) vs Hybrid (BM25 + Dense, RRF) | +0.048 [+0.030, +0.068] | +0.057 [+0.036, +0.081] |
| Two-Stage (Hybrid + Reranker) (fine-tuned) vs Two-Stage (Hybrid + Reranker) | +0.009 [+0.002, +0.019] | +0.007 [+0.001, +0.015] |
