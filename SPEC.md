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


6. Semantic search


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

        Similarity Retrieval

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


- Similarity search
- Top-k retrieval (k = 10 chunks per question; 10 per document when comparing)
- Source tracking


---

# 10. Retrieval Augmented Generation Pipeline


Workflow:


```

User Question

       |

Question Embedding

       |

Vector Similarity Search

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


- Recall@K

- Precision@K


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


- Hand-write ~15-20 question/answer pairs against your own test PDFs, noting the correct source page for each.
- Score retrieval (Recall@K/Precision@K) by checking whether the correct page appears in the retrieved chunks.
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


- Better retrieval
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
