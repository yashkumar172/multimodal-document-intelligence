# SPEC.md

# Multimodal Document Intelligence System using RAG and Vision-Language APIs


# 1. Project Overview

## Objective

Build an AI-powered document intelligence system that can retrieve, analyze, and answer questions from technical documents such as:

- Research papers
- Lecture notes
- Technical reports
- Machine learning papers
- Documentation


The system will use:

- Retrieval Augmented Generation (RAG)
- Transformer-based embeddings
- Vector databases
- Large Language Models (LLMs)
- Vision-Language APIs for visual document understanding


The goal is to build an AI engineering system similar to enterprise document assistants used for knowledge retrieval.


---

# 2. Problem Statement

Traditional document search systems depend on keyword matching and fail to understand semantic meaning.

They also struggle with:

- Long documents
- Complex technical content
- Figures
- Tables
- Graphs


This project aims to create a system that can:

- Understand document content
- Retrieve relevant information
- Answer questions using document context
- Provide source citations
- Compare multiple documents
- Analyze visual information when required


---

# 3. Project Scope


## Core MVP Features

The first version should implement:


1. Document upload

Supported:

- PDF files


2. Text extraction

Extract:

- Text
- Page numbers
- Metadata


3. Document chunking


4. Text embedding generation


5. Vector database storage


6. Hybrid search (keyword BM25 + semantic) with cross-encoder reranking


7. Question answering using RAG


8. Source citation


---

# 4. Advanced Features


After completing MVP:


## Visual Document Understanding

Support:

- Figures
- Graphs
- Histograms
- Tables
- Architecture diagrams


Use a pretrained Vision-Language API for visual understanding.


Chosen:

- Gemini Vision API (see section 16 for rationale)


The project does not train a Vision-Language Model.


The system integrates existing vision-language models into an AI pipeline.


---

# 5. System Architecture


```

                 User

                  |

                  |

            Streamlit Interface

                  |

                  |

            Document Upload

                  |

                  |

          ----------------------

          |                    |

        Text              Images/Figures

          |                    |

          |                    |

 Text Extraction       Vision-Language API

          |                    |

          |             Image Description

          |                    |

          ----------------------

                  |

                  |

          Document Processing

                  |

                  |

          Text Chunk Generation

                  |

                  |

       Sentence Transformer Embeddings

                  |

                  |

              ChromaDB

          Vector Database

                  |

                  |

             User Query

                  |

                  |

          Query Embedding

                  |

                  |

  Hybrid Retrieval (BM25 + Dense, RRF)

                  |

                  |

       Cross-Encoder Reranking

                  |

                  |

          Retrieved Context

                  |

                  |

              LLM

                  |

                  |

        Answer + Citations


```


---

# 6. Document Processing Pipeline


## PDF Processing


Tools:

- PyMuPDF
- PDFPlumber


Extract:

- Text
- Page number
- Metadata


## Figure/Table Extraction Approach


PyMuPDF's embedded-image extraction misses vector-drawn charts (most matplotlib-style figures are vector graphics, not embedded bitmaps) and has no way to locate tables.


Approach used instead:


```

PDF Page

     |

Render full page to image (PyMuPDF get_pixmap)

     |

Send page image to Vision-Language API

     |

Description (covers any figures/tables/diagrams on that page)

```


Whole-page rendering is simpler than detecting and cropping individual figure/table regions, and is sufficient for this project's scope.


---

# 7. Text Processing


The extracted text should be:


1. Cleaned

2. Split into meaningful chunks

3. Converted into embeddings


Chunk size:


```
200 tokens, 40 token overlap
```


Tokens are counted with the embedding model's own tokenizer (all-MiniLM-L6-v2, see section 8).
The model reads at most 256 tokens (including 2 special tokens), so 200-token chunks are always
embedded in full; nothing is silently truncated.


(The 200 vs 800 token comparison in section 15 is a later experiment, not a blocker for the MVP build.)


Chunk metadata:


```
document_name

page_number

chunk_id

section

content

```


---

# 8. Embedding System


Use transformer-based embedding models.


Initial model:


```
sentence-transformers/all-MiniLM-L6-v2
```


Purpose:


Convert text into semantic vectors for similarity search.


Example:


Text:

```
Transformer uses self-attention mechanism.
```


Embedding:


```
[0.21,0.53,0.12,...]

```


---

# 9. Vector Database


Use:


```
ChromaDB
```


Store:


- Embeddings
- Document metadata
- Page information


Required operations:


- Similarity search (dense, top 30 candidates)
- Metadata filtering by document (for comparison)
- Source tracking


A BM25 keyword index (rank-bm25, BM25Okapi) is kept in memory alongside ChromaDB.
It is rebuilt from the chunks stored in ChromaDB whenever documents are added or
removed, so both indexes always hold the same chunks (no separate file to drift).


---

# 10. Retrieval Augmented Generation Pipeline


Workflow:


```

User Question

       |

Question Embedding              BM25 Tokenization

       |                               |

Dense Search (top 30)           BM25 Search (top 30)

       |                               |

       +---------------+---------------+

                       |

Reciprocal Rank Fusion (k = 60), keep top 25

                       |

Cross-Encoder Reranking, keep top 10

                       |

Relevant Document Chunks

       |

Context Creation

       |

LLM

       |

Final Answer


```


The answer should be generated using retrieved document information.


Retrieval settings:


- Stage 1: BM25 (top 30) and dense MiniLM search (top 30), fused with
  Reciprocal Rank Fusion: score(d) = sum of 1 / (60 + rank of d in each list);
  duplicates merged; top 25 kept.
- Stage 2: cross-encoder reranker (cross-encoder/ms-marco-MiniLM-L-6-v2, local
  CPU) scores the 25 (question, chunk) pairs; the top 10 go to the LLM
  (~2,000 tokens of context with 200-token chunks).
- BM25 tokens keep word-internal hyphens, so codes like "XJ-4471-B" stay whole.
- Comparison runs both stages separately for each document, filtered to it.
- Each answer reports the latency of every stage (embedding, BM25, dense, RRF,
  reranking, LLM).


---

# 11. Vision-Language Integration


The system can optionally analyze visual information.


The role of the Vision-Language Model is:

- Understanding images
- Generating descriptions
- Extracting visual information


Supported content:


## Graphs

Example questions:

- What trend is shown?
- Which model performs better?


## Histograms

Example questions:

- Describe the distribution.
- Identify possible outliers.


## Tables

Example questions:

- Which method gives the highest accuracy?


## Diagrams

Example questions:

- Explain this architecture.


---

# 12. Multimodal Indexing Strategy


Images are converted into text descriptions.


Pipeline:


```

Image/Figure

      |

Vision-Language API

      |

Text Description

      |

Embedding Model

      |

Vector Database


```


This allows visual information to participate in text-based retrieval.


---

# 13. Document Comparison


Optional advanced feature.


The system should compare two documents.


Example:


Input:


```
Paper_A.pdf

Paper_B.pdf

```


Question:


```
Compare the methodologies used in both papers.
```


Output:


```

Paper A:

Uses CNN based architecture.


Paper B:

Uses Transformer based architecture.


Main Difference:

Paper B uses self-attention instead of convolution.

```


---

# 14. Evaluation


The system should be evaluated using:


## Retrieval Evaluation


Metrics:


- HitRate@K (Recall@K: the correct page is in the top K)

- MRR@K (Mean Reciprocal Rank)

- NDCG@K (Normalized Discounted Cumulative Gain)

- Precision@K


for K = 3, 5 and 10, plus p50/p95 retrieval latency.


Example:


Question:

"What dataset was used?"


Expected:

Correct document section retrieval.


---

## Answer Evaluation


Measure:


- Answer relevance

- Faithfulness

- Citation correctness

- Hallucination rate


## Evaluation Method


No pre-existing labeled dataset is available, so build one manually:


- 15 question/answer pairs generated from passages of a test PDF, each linked to its source page (wording is close to the source text).
- 12 hand-paraphrased question/answer pairs on the same PDF, worded differently from the text, for a stricter test.
- A retrieved chunk is relevant if it comes from the expected page. Score retrieval with the metrics above.
- Compare four retrieval configurations on the same questions (benchmarks/benchmark_retrieval.py): Dense-only, Sparse-only (BM25), Hybrid (RRF), Two-Stage (Hybrid + reranker).
- Score faithfulness/hallucination/citation correctness manually (pass/fail per answer) rather than using an automated evaluation framework (e.g. RAGAS) — not worth the added dependency for this scope.


---

# 15. Experiments


Optional stretch work — attempt only after Phases 1-3 are working. Not required for a functioning MVP.


## Embedding Comparison


Compare:


- MiniLM

- BGE embeddings (small variant, e.g. bge-small-en)

- E5 embeddings (small variant, e.g. e5-small)


Use the small variants only — full-size BGE/E5 are noticeably slower on a CPU-only laptop.


Evaluate:


- Retrieval accuracy

- Latency


---

## Chunk Size Analysis


Compare:


Small chunks:

```
200 tokens
```


Large chunks:

```
800 tokens
```


Analyze effect on retrieval quality.


---

# 16. Technology Stack


## Programming Language

Python


## Interface

Streamlit


## Document Processing

- PyMuPDF
- PDFPlumber


## Embeddings

- Sentence Transformers


## Vector Database

ChromaDB


## Keyword Search

rank-bm25 (BM25Okapi), in memory


## Reranker

Sentence Transformers cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90 MB,
runs on CPU; reranking 25 candidates takes about 0.8 s on a Ryzen 5 5500U)


## LLM

Gemini API only (has a genuinely usable free tier; GPT/Claude APIs do not — keep them as a later swap-in only if budget becomes available)

Model note (confirmed empirically during development): the standard "flash" tier
(e.g. gemini-3.8-flash / gemini-flash-latest) is capped at only ~20 free requests
per day per project — too low for real use. The "flash-lite" tier
(gemini-flash-lite-latest) has a much higher free quota and is used instead for
both answer generation and vision description. Trade-off: lite gives noticeably
less detailed outputs on complex figures (e.g. it may skip precise value
rankings a chart shows) in exchange for being usable at all on the free tier.


## Vision Understanding

Gemini Vision API (Gemini is natively multimodal, so one API key/account covers both the LLM and vision roles — currently gemini-flash-lite-latest for both, see note above)


---

# 17. Development Plan


## Phase 1

Build basic RAG:


- PDF upload
- Text extraction
- Chunking
- Embeddings
- Vector search
- Question answering


---

## Phase 2

Improve AI system:


- Better retrieval (two-stage hybrid search: BM25 + dense with RRF, then cross-encoder reranking)
- Citations (clean source/page-number display in the Streamlit UI — this is a differentiator, worth real polish here)
- Evaluation pipeline


(Embedding experiments moved to section 15 as optional stretch work — not part of the core phase plan.)


---

## Phase 3

Add multimodal capability:


- Figure understanding
- Graph explanation
- Table analysis


---

## Phase 4

Add advanced features:


- Document comparison (priority differentiator — build this before attempting the section 15 experiments if time is short)
- Multi-document reasoning


---

# 18. Learning Objectives


After completing this project, understand:


- How RAG systems work
- How embeddings represent meaning
- How vector databases perform search
- How LLM applications are built
- How hallucination occurs
- How AI systems are evaluated
- How multimodal AI applications are designed


---

# 19. Development Rules


- Write modular code.
- Keep API keys secure using environment variables.
- Implement MVP before advanced features.
- Test each module independently.
- Maintain clear documentation.
- Avoid unnecessary complexity.


---

# Final Project Title


## Multimodal Document Intelligence System using Retrieval Augmented Generation and Vision-Language APIs
