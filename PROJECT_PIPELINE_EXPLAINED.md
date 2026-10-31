# Project Pipeline Explained

## Multimodal Document Intelligence System using RAG and Vision-Language APIs

A technical deep dive into **exactly what the code does**, file by file and step by step: the design decisions, the trade-offs behind them, the measured results and the known limitations. Every statement below is based on the actual source code in this repository (`app.py`, `src/`, `eval/`, `benchmarks/`, `tests/`). For a shorter overview, see the [README](README.md).

---

## Table of Contents

1. [One-Paragraph Summary (the "elevator pitch")](#1-one-paragraph-summary)
2. [Tech Stack and Why Each Piece Was Chosen](#2-tech-stack)
3. [Folder Structure: What Each File Does](#3-folder-structure)
4. [The Big Picture: Full Pipeline Diagram](#4-full-pipeline-diagram)
5. [Pipeline A: Ingestion (Upload → Index)](#5-pipeline-a-ingestion)
6. [Pipeline B: Question Answering (RAG)](#6-pipeline-b-question-answering-rag)
7. [Pipeline C: Document Comparison](#7-pipeline-c-document-comparison)
8. [Pipeline D: Evaluation](#8-pipeline-d-evaluation)
9. [The Streamlit User Interface](#9-the-streamlit-user-interface)
10. [Worked Example: One PDF, End to End](#10-worked-example)
11. [Key Design Decisions (and the reasoning)](#11-key-design-decisions)
12. [Known Limitations](#12-known-limitations)
13. [Testing Strategy](#13-testing-strategy)
14. [How to Run the Project](#14-how-to-run)
15. [Technical Interview Questions with Answers](#15-technical-interview-questions)
16. [Core Concepts Glossary](#16-glossary)

---

## 1. One-Paragraph Summary

The user uploads one or more PDF files in a Streamlit web app. The system pulls the **text** out of every page (pdfplumber) and, for pages that probably contain **figures, charts, or tables**, it renders the whole page as an image (PyMuPDF) and asks **Gemini Vision** to describe it in words. Both the page text and the figure descriptions are split into 200-token **chunks** (sized to fit the embedding model's 256-token input limit), turned into 384-number **embedding vectors** with the `all-MiniLM-L6-v2` Sentence Transformer, and stored in a **ChromaDB** vector database together with the document name and page number. When the user asks a question, **two-stage hybrid retrieval** finds the context: **BM25 keyword search** and **dense (embedding) search** each return 30 candidates, **Reciprocal Rank Fusion** merges them into the best 25, and a **cross-encoder reranker** picks the **10 most relevant chunks**. Those chunks are pasted into a prompt for **Gemini** (`gemini-flash-lite-latest`), which writes an answer **using only that context** and **cites page numbers**. The app shows how long each stage took, can **compare two documents** side by side, and comes with an **evaluation script** and a **retrieval benchmark** (HitRate@K, MRR@K, NDCG@K, latency).

---

## 2. Tech Stack

| Layer | Tool | Where in code | Why this choice |
|---|---|---|---|
| UI | **Streamlit** | `app.py` | Build a web UI in pure Python; built-in chat, file upload, session state. |
| PDF text extraction | **pdfplumber** | `src/pdf_utils.py` | Reliable page-by-page text extraction with correct page numbers. |
| PDF page rendering | **PyMuPDF** (`pymupdf`) | `src/pdf_render.py` | Fast rendering of a page to a PNG image; can also detect embedded images. |
| Tokenizer for chunking | **MiniLM's own WordPiece tokenizer** (`transformers.AutoTokenizer`) | `src/chunking.py` | Chunks are measured in exactly the tokens the embedding model counts, so no chunk ever exceeds its 256-token limit. |
| Embeddings | **Sentence Transformers** – `all-MiniLM-L6-v2` | `src/embeddings.py` | Small (~90 MB), fast on CPU, free, good semantic quality; outputs 384-dim vectors. |
| Vector database | **ChromaDB** (persistent) | `src/vector_store.py` | Local, no server needed, supports metadata filtering (`where=`), saves to disk. |
| Keyword search | **rank-bm25** (`BM25Okapi`) | `src/vector_store.py` | Classic keyword ranking; catches exact terms (codes, numbers, names) that embeddings blur. Small, pure Python, in memory. |
| Reranker | **Sentence Transformers `CrossEncoder`**: `cross-encoder/ms-marco-MiniLM-L-6-v2` | `src/reranker.py` | Reads the question and a passage together, so it judges relevance more accurately than comparing two separate embeddings. Small (~90 MB, 22M parameters), runs on CPU. |
| LLM + Vision | **Google Gemini** (`google-genai`), model `gemini-flash-lite-latest` | `src/gemini_client.py` | One API key covers both text generation and image understanding; usable free tier. |
| Secrets | **python-dotenv** | `app.py`, eval scripts | Reads `GEMINI_API_KEY` from `.env` so the key is never hard-coded. |

> **Why `flash-lite` and not `flash`?** During development it was found that the normal "flash" tier allows only ~20 free requests/day, which is too few (ingesting one PDF can need up to 15 vision calls). "flash-lite" has a much larger free quota. The trade-off is slightly less detailed descriptions of complex figures.

---

## 3. Folder Structure

```
AIML Project DSAI/
│
├── app.py                    ← Streamlit web app (entry point: `streamlit run app.py`)
├── requirements.txt          ← Python dependencies
├── .env / .env.example       ← GEMINI_API_KEY lives here (never committed; listed in .gitignore)
├── SPEC.md                   ← Original project specification
│
├── src/                      ← All core logic, one responsibility per module
│   ├── pdf_utils.py          ← Step 1: extract text per page (pdfplumber)
│   ├── pdf_render.py         ← Step 2: find "visual" pages + render page → PNG (PyMuPDF)
│   ├── gemini_client.py      ← All calls to Gemini: describe image, answer, compare, retry
│   ├── chunking.py           ← Step 3: split text into 200-token chunks with 40 overlap
│   ├── embeddings.py         ← Step 4: text → 384-dim vectors (MiniLM)
│   ├── vector_store.py       ← Step 5: ChromaDB storage + dense search, BM25 index, RRF fusion
│   ├── reranker.py           ← Cross-encoder reranking (stage 2 of retrieval)
│   └── pipeline.py           ← ORCHESTRATOR: wires the modules together
│
├── chroma_db/                ← ChromaDB's on-disk storage (created automatically)
│
├── eval/
│   ├── build_eval_set.py     ← Generates question/answer/page test set from indexed chunks
│   ├── eval_set.json         ← The 15 generated test questions
│   ├── eval_set_paraphrased.json ← 12 hand-paraphrased test questions
│   ├── metrics.py            ← HitRate@K, MRR@K, NDCG@K, Precision@K, percentiles
│   ├── run_evaluation.py     ← Runs every question through the full pipeline (with Gemini)
│   └── eval_report.md        ← Output report (Recall@K, MRR@K, Precision@K, per-question table)
│
├── benchmarks/
│   ├── benchmark_retrieval.py ← Ablation: dense vs BM25 vs hybrid vs two-stage (no Gemini)
│   └── benchmark_results.md  ← Latest benchmark tables
│
└── tests/                    ← Manual test scripts for each phase
    ├── test_gemini_key.py, list_gemini_models.py
    ├── test_phase1_manual.py, test_full_pipeline.py
    ├── test_phase3_multimodal.py, test_phase4_comparison.py
    ├── test_hybrid_retrieval.py
    └── cleanup_test_doc.py
```

**Architecture:** the code is **layered**.
`app.py` (UI) → only talks to `pipeline.py` (orchestration) → which calls small single-purpose modules. The UI never touches ChromaDB or Gemini directly (except `vector_store.list_documents()` to fill a dropdown). Each module can be tested on its own, and swapping one component (the vector database, the embedding model, the LLM provider) changes one file.

---

## 4. Full Pipeline Diagram

```
                              ┌──────────────────────┐
                              │   User (browser)     │
                              └──────────┬───────────┘
                                         │
                              ┌──────────▼───────────┐
                              │  Streamlit  app.py   │
                              └───┬─────────┬────────┘
              Upload PDF(s)       │         │       Ask question / Compare
     ═════════════════════════════╪═════════╪══════════════════════════════
     INGESTION  (pipeline.ingest_pdf)       │   QUERY (pipeline.answer_question)
                                  │         │
                    ┌─────────────▼───┐     │       ┌────────────────────────┐
                    │  PDF bytes      │     │       │ Question text          │
                    └──┬───────────┬──┘     │       └──────────┬─────────────┘
                       │           │        │                  │
        ┌──────────────▼──┐   ┌────▼─────────────┐    ┌────────▼────────────┐
        │ pdfplumber      │   │ find_visual_pages│    │ MiniLM embedding    │
        │ text per page   │   │ (image? caption?)│    │ (same model!)       │
        └───────┬─────────┘   └────┬─────────────┘    └────────┬────────────┘
                │                  │ max 15 pages               │
                │             ┌────▼─────────────┐    ┌────────▼────────────┐
                │             │ PyMuPDF render   │    │ Dense top 30 +      │
                │             │ page → PNG 150dpi│    │ BM25 top 30         │
                │             └────┬─────────────┘    │ → RRF (k=60) top 25 │
                │                  │                  │ → cross-encoder     │
                │                  │                  │   rerank → top 10   │
                │                  │                  └────────┬────────────┘
                │             ┌────▼─────────────┐             │
                │             │ Gemini Vision    │    ┌────────▼────────────┐
                │             │ "describe page"  │    │ Build prompt:       │
                │             └────┬─────────────┘    │ rules + history +   │
                │                  │ description text │ labelled context +  │
        ┌───────▼─────────┐  ┌─────▼────────────┐     │ question            │
        │ chunk (text)    │  │ chunk (visual)   │     └────────┬────────────┘
        │ 200 tok / 40 ov │  │ 200 tok / 40 ov  │     ┌────────▼────────────┐
        └───────┬─────────┘  └─────┬────────────┘     │ Gemini LLM          │
                └────────┬─────────┘                  │ answer + citations  │
                ┌────────▼─────────┐                  └────────┬────────────┘
                │ MiniLM embedding │                           │
                │ 384-dim vectors  │                  ┌────────▼────────────┐
                └────────┬─────────┘                  │ UI: answer +        │
                ┌────────▼─────────┐                  │ "Sources" expander  │
                │ ChromaDB (disk)  │◄─── queried ─────┤ (doc, page, type)   │
                │ vector+text+meta │                  └─────────────────────┘
                └──────────────────┘
```

There are **two main flows**:

- **Offline / indexing flow** – happens once per PDF when you click "Process documents".
- **Online / query flow** – happens every time you ask a question.

The bridge between them is **ChromaDB**, and the key rule is: **the same embedding model must be used for both chunks and questions**, otherwise the vectors would live in different "spaces" and similarity would be meaningless.

---

## 5. Pipeline A: Ingestion

**Entry point:** `pipeline.ingest_pdf(file_buffer, document_name)` in `src/pipeline.py`, called from `app.py` when the user clicks **"Process documents"**.

### Step A1 — Read the PDF bytes

```python
pdf_bytes = file_buffer.read()
```

The uploaded file is read **once** into memory as bytes. Those bytes are reused by both pdfplumber and PyMuPDF (wrapped in `io.BytesIO` for pdfplumber), so the file is never written to disk.

### Step A2 — Extract text page by page (`src/pdf_utils.py`)

```python
with pdfplumber.open(file_path_or_buffer) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
```

- Output: a list like `[(1, "Introduction ..."), (2, "Figure 1 shows ..."), ...]`.
- Page numbers are **1-indexed** (human-readable), which is what we later show in citations.
- `or ""` protects against pages with no extractable text (e.g. a pure image / scanned page) — they become empty strings and are skipped later.

### Step A3 — Detect pages that probably contain visuals (`src/pdf_render.py → find_visual_pages`)

A page is flagged as "visual" if **either** condition is true:

1. **It contains an embedded raster image** – `doc[page].get_images(full=True)` is non-empty (photos, screenshots, bitmap charts).
2. **Its text contains a caption keyword** – any of `"figure"`, `"fig."`, `"table"`, `"chart"`, `"graph"`, `"diagram"` (case-insensitive).

**Why the keyword check?** Most charts in research papers (e.g. made with matplotlib) are **vector drawings**, not embedded images, so `get_images()` alone would miss them. A caption like "Figure 3:" is a cheap, reliable hint that a figure is on that page.

Then in `pipeline.py`:

```python
visual_page_numbers = pdf_render.find_visual_pages(pdf_bytes, pages)[:MAX_VISUAL_PAGES]   # MAX = 15
```

At most **15 pages** per document are sent to the vision model — this protects the free-tier API quota and keeps ingestion time reasonable.

### Step A4 — Render each visual page to an image (`render_page_image`)

```python
zoom = dpi / 72                         # PDF native unit is 72 points/inch → 150 dpi ≈ 2.08× zoom
pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
png_bytes = pix.tobytes("png")
```

The **entire page** is rendered as a PNG at 150 DPI. This approach (from spec §6) captures *everything* visible — vector charts, tables, diagrams, equations — without needing a separate figure-detection or cropping model. 150 DPI is a balance between legibility for the vision model and image size.

### Step A5 — Describe the image with Gemini Vision (`gemini_client.describe_image`)

The PNG bytes + a carefully written prompt are sent to Gemini:

```
Describe this document page image factually and specifically.
- chart/graph/histogram/plot → type, axes, trend, notable values/outliers
- table → columns & rows, best/worst value
- diagram → structure or flow
- nothing visual → say so in one sentence
Do not speculate beyond what is visible in the image.
```

The prompt is **structured by content type** so the descriptions contain the kind of facts users actually ask about ("which model is best?", "what trend?"). The last line reduces hallucination.

Result: a list of `(page_number, description_text)` tuples — exactly the same shape as the text pages from Step A2. **This is the core multimodal trick:** images are converted to text so they can go through the *same* text pipeline (spec §12, "Multimodal Indexing Strategy").

### Step A6 — Chunking (`src/chunking.py → chunk_pages`)

Called twice: once with `content_type="text"` and once with `content_type="visual"`.

```python
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")   # MiniLM's own tokenizer
offsets = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)["offset_mapping"]   # (char_start, char_end) per token
start = 0
while start < len(offsets):
    end = min(start + 200, len(offsets))
    chunk_text = text[offsets[start][0]:offsets[end - 1][1]]        # slice the ORIGINAL text
    ... append chunk ...
    if end == len(offsets): break
    start = end - 40                                                 # step back 40 tokens → overlap
```

- **Chunk size = 200 tokens, overlap = 40 tokens (20%).** A sliding window over the token list.
- **Tokens are counted with the embedding model's own tokenizer**, so the size is exactly what MiniLM sees. MiniLM reads at most **256 tokens**, including 2 special tokens (`[CLS]`, `[SEP]`). 200 + 2 = 202 tokens, which stays under the limit with room to spare, so **every word of every chunk is part of its embedding**.
- **Why slice by character offsets instead of decoding tokens?** The MiniLM tokenizer is *uncased* (it lowercases everything) and splits words into pieces like `"embed", "##ding"`. Decoding tokens back into text would give lowercased, oddly spaced text. Each token's character position in the original page is used instead, so the chunk text keeps its original casing and formatting, which matters because it is shown to the LLM.
- **Chunks never cross page boundaries** (each page is chunked separately). This guarantees every chunk maps to **exactly one page number**, which keeps citations accurate.
- Empty pages are skipped (`if not text.strip(): continue`).
- Each chunk is a dictionary:

```python
{
  "document_name": "paper.pdf",
  "page_number": 4,
  "chunk_id": "paper.pdf_text_p4_c7",     # unique: doc + type + page + running counter
  "content": "...chunk text...",
  "content_type": "text"                 # or "visual"
}
```

**Why overlap?** If an important sentence falls on the boundary between two chunks, the 40-token overlap ensures it appears complete in at least one chunk.

**Why tokens instead of characters/words?** Embedding models work in tokens and have a hard token limit, so token-based sizes are exact and predictable.

**Why 200 tokens?** MiniLM **silently truncates** any input longer than 256 of *its own* tokens. If a chunk were longer, its end would never be embedded, so that text could not be found by search even though the LLM would read it. A 200-token chunk plus the 2 special tokens is 202 tokens, so every chunk is embedded in full. Tokens are counted with MiniLM's tokenizer rather than a general one like tiktoken, because different tokenizers split the same text into different numbers of tokens (MiniLM's can produce up to ~1.25× more). Counting with the model's own tokenizer makes the limit exact.

### Step A7 — Embedding (`src/embeddings.py`)

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # shared with chunking.py
_model = SentenceTransformer(MODEL_NAME)                 # loaded once (lazy singleton)
vectors = model.encode(texts).tolist()
```

- All chunks (text + visual) are embedded **in one batch** for speed.
- `MODEL_NAME` is imported by `chunking.py`, so the chunker and the embedder are **guaranteed to use the same model's tokenizer**. Swapping the embedding model in one place updates both.
- Max input: **256 tokens**. Anything longer is truncated, which is why the chunks are 200 tokens.
- Each chunk becomes a **384-dimensional vector**. Texts with similar *meaning* get vectors that point in similar directions, even if they use different words (e.g. "car" and "automobile").
- The model is loaded **lazily** the first time it is needed and then reused (global `_model`), because loading it takes a few seconds.

### Step A8 — Store in ChromaDB (`src/vector_store.py → add_chunks`)

```python
collection.add(
    ids=[chunk_id ...],
    embeddings=[vector ...],
    documents=[chunk text ...],
    metadatas=[{"document_name", "page_number", "content_type"} ...],
)
```

- Uses `chromadb.PersistentClient(path="chroma_db")` → data is saved to disk and **survives app restarts**.
- One collection named `"documents"` holds chunks from **all** uploaded PDFs; the `document_name` metadata keeps them separable.
- For each chunk ChromaDB stores 4 things: **ID, vector, original text, metadata**. The text is stored so we can feed it to the LLM later; the metadata is what makes **citations** and **per-document filtering** possible.
- ChromaDB builds an **HNSW index** (Hierarchical Navigable Small World graph) for fast approximate nearest-neighbour search.
- `add_chunks` also **invalidates the BM25 keyword index**. It is rebuilt in memory from the chunks stored in ChromaDB on the next search (`collection.get()` → tokenize → `BM25Okapi`). Deleting a document or clearing the database does the same. Because BM25 is always rebuilt from ChromaDB's own copy of the text, the two indexes can never disagree about which chunks exist, and there is no extra file to keep in sync.

### Step A9 — Return a summary

```python
return {"text_chunks": ..., "visual_chunks": ..., "visual_pages_found": ...}
```

The UI shows: *"Indexed paper.pdf: 42 text chunks, 6 figure/table descriptions"*.

---

## 6. Pipeline B: Question Answering (RAG)

**Entry point:** `pipeline.answer_question(question, top_k=TOP_K, history=...)` (`TOP_K = 10`), called when the user types in the chat box. Retrieval (steps B1 to B4) lives in `pipeline.retrieve`, which comparison and the benchmark reuse.

### Step B1 — Embed the question

```python
query_vector = embeddings.embed_texts([question])[0]
```

Same MiniLM model as ingestion → the question vector lives in the same 384-dim space as the chunk vectors.

### Step B2 — Stage 1a: two candidate searches

```python
dense_hits = vector_store.query(query_vector, top_k=30)       # meaning
bm25_hits  = vector_store.bm25_query(question, top_k=30)      # exact words
```

- **Dense search** (ChromaDB, HNSW index) returns the 30 chunks whose vectors are closest to the question vector. It matches *meaning*, so "What dataset was used?" finds "The model was trained on WMT 2014".
- **BM25 search** returns the 30 chunks with the best keyword score. BM25 rewards chunks that contain the question's words, especially *rare* words (high IDF), with diminishing returns for repeats and a penalty for very long chunks. It catches exact terms that embeddings blur: codes like `XJ-4471-B`, numbers, rare names.
- **Tokenizer** for BM25: lowercase, then the regex `\b[\w-]+\b`, which keeps word-internal hyphens so a code stays one token instead of being split into pieces.
- Only chunks with a positive BM25 score are returned. Both searches cover **text chunks and figure descriptions** together.
- If nothing is indexed, retrieval returns nothing and the pipeline replies *"No documents indexed yet. Please upload a PDF first."* (no wasted API call).

### Step B3 — Stage 1b: Reciprocal Rank Fusion (`vector_store.rrf_merge`)

```python
score(chunk) = sum over the two lists of  1 / (60 + rank of chunk in that list)
```

- Ranks start at 1. A chunk ranked 1st by BM25 and 3rd by dense search scores `1/61 + 1/63`; a chunk found by only one search gets only one term.
- Duplicates (the same chunk ID in both lists) are merged, and the **best 25** are kept.
- **Why ranks and not scores?** BM25 scores (unbounded) and vector distances live on completely different scales, so adding them is meaningless. RRF only uses positions. The constant **k = 60** (the value from the original RRF paper) stops a single first place from dominating, so a chunk that both searches rank fairly high beats one that only one search ranks first.

### Step B4 — Stage 2: cross-encoder reranking (`src/reranker.py`)

```python
scores = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2").predict(
    [(question, chunk_text) for chunk_text in candidates])
hits = top 10 candidates by score
```

- A **bi-encoder** (MiniLM in dense search) embeds the question and the chunk *separately* and compares vectors. A **cross-encoder** reads the question and the chunk *together* in one pass, so attention can connect question words with passage words. It is more accurate, but it needs one model run per (question, chunk) pair, which is why it only sees the 25 fused candidates instead of every chunk.
- The model was trained on MS MARCO (real search queries with relevant passages). It is small (22M parameters, ~90 MB) and runs on CPU.
- The **top 10** reranked chunks go to the LLM: with 200-token chunks that is about 2,000 tokens of context.
- **Loaded once, safely:** the model is a lazy singleton created behind the same lock (`vector_store.INIT_LOCK`) as the ChromaDB client, because Streamlit runs each browser session on its own thread and two sessions must not load it at the same time.
- **Latency:** each stage is timed with `time.perf_counter()` (embedding, dense, BM25, RRF, reranking, and later the LLM call) and the times are returned with the answer. Reranking 25 candidates takes about **0.8 s** on the development laptop (Ryzen 5 5500U CPU); everything else in retrieval takes about 30 ms.

### Step B5 — Build the context with source labels (`gemini_client.generate_answer`)

Each retrieved chunk is prefixed with a label so the LLM knows where it came from:

```
[Source: paper.pdf, page 4]
...chunk text...

[Figure/Table description — paper.pdf, page 6]
...Gemini's description of the chart...
```

The label is different for visual chunks so the LLM (and user) can tell that the information came from a figure/table.

### Step B6 — Add conversation history

The previous chat turns are passed in as `history` (list of `(role, text)`) and written into the prompt as:

```
user: What dataset was used?
assistant: The WMT 2014 dataset ...
```

This lets the LLM understand follow-up questions such as *"How many sentence pairs does **it** have?"*.

### Step B7 — The prompt (grounding instructions)

```
You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.
Always cite the source document and page number for any claim you make.

Conversation history: ...
Context: ...
Question: ...
Answer (include page-number citations):
```

Three rules here are the heart of **reducing hallucination**:
1. **"ONLY the context"** → grounding: the model should not use its general knowledge.
2. **"say you don't know"** → gives the model a safe way out instead of inventing an answer.
3. **"Always cite"** → makes every claim verifiable by the user.

### Step B8 — Call Gemini with retry (`_generate_with_retry`)

```python
for attempt in range(max_retries + 1):     # up to 3 attempts total
    try:
        return client.models.generate_content(model="gemini-flash-lite-latest", contents=...)
    except errors.ServerError:
        if attempt == max_retries: raise
        time.sleep(2 * (attempt + 1))      # wait 2s, then 4s
```

If Gemini returns a temporary server error (5xx, e.g. "model overloaded"), the call is retried with **increasing wait times** (linear backoff). Other errors are raised immediately. The same helper is used for vision, answers, and comparison.

### Step B9 — Return answer, sources and timings

```python
sources = [{"document": ..., "page": ..., "type": "text" | "visual"} for h in hits]
return answer, sources, timings      # timings: ms per stage, including llm_ms
```

The UI **deduplicates** sources (two chunks from the same page show once) and shows them in a **"Sources" expander**, e.g. `- paper.pdf (page 6, figure/table)`, followed by a one-line **latency breakdown** for that question (embedding, BM25, dense, RRF, rerank, Gemini, total).

So citations appear in **two places**: inline in the LLM's answer (generated) **and** in the Sources list (taken directly from metadata, so it is always correct even if the LLM forgets to cite).

### RAG flow in one line

> **Question → BM25 top 30 + dense top 30 → RRF keeps 25 → cross-encoder picks 10 → paste chunks + rules into prompt → LLM writes grounded answer with page citations.**

---

## 7. Pipeline C: Document Comparison

**Entry point:** `pipeline.compare_documents(document_a, document_b, question=None, top_k=TOP_K)`. The "Compare Documents" panel only appears in the UI when **2 or more** documents are indexed.

### Steps

1. **Question:** if the user leaves the focus box empty, a default is used:
   *"Compare the methodology, approach, and key findings of these two documents."*
2. **Two filtered retrievals**, the key idea:
   ```python
   hits_a, timings_a = retrieve(question, top_k=10, document_name=document_a)
   hits_b, timings_b = retrieve(question, top_k=10, document_name=document_b)
   ```
   The whole two-stage retrieval runs once per document, and **every stage is filtered** to that document: dense search uses ChromaDB's `where={"document_name": ...}` metadata filter, BM25 skips chunks from other documents, so RRF and the reranker only ever see that document's chunks.
   **Why not one search?** If one paper is much longer or more similar to the question, a single top-k search could return 10 chunks from paper A and 0 from paper B. Separate searches **guarantee balanced evidence** from both documents.
3. If either side has no hits → returns *"One or both documents have no indexed content to compare."*
4. **Structured prompt** (`gemini_client.compare_documents`) forces this output format:
   ```
   <Doc A name>:
   <key points with page citations>

   <Doc B name>:
   <key points with page citations>

   Main Differences:
   <summary>
   ```
   Context from each document is given in **separate labelled sections** (`Context from Paper_A.pdf: ...`), so the LLM cannot mix up which fact came from which paper. It is also told to say so if the context is insufficient rather than guess.
5. Returns the answer, combined sources (20 chunks: 10 from each) and the summed stage timings, shown in a "Sources used in comparison" expander.

This implements the comparison feature described in `SPEC.md` §13.

---

## 8. Pipeline D: Evaluation

Located in `eval/` and `benchmarks/`, run from the command line. `eval/metrics.py` holds the metric functions both use.

### D1 — Build a test set (`eval/build_eval_set.py`)

1. Lists indexed documents; picks the one given on the command line (or the only one).
2. Fetches all **text** chunks of that document (visual chunks excluded) using `vector_store.get_document_chunks`.
3. **Randomly samples 15 chunks.**
4. For each chunk, asks Gemini: *"Based ONLY on the following text, write exactly one factual question … respond with JSON `{question, expected_answer}`"*.
5. Strips any ```` ```json ```` fences, parses the JSON, and records the chunk's **page number as the ground-truth `expected_page`**.
6. Saves to `eval/eval_set.json`. The script tells you to **review/edit** the questions manually before using them.

> Note: the spec suggested hand-writing ~15–20 Q&A pairs. The code **automates** the first draft with the LLM (faster and tied to a known page). Because generated questions reuse the source's wording, a second set of **12 hand-paraphrased questions** (`eval/eval_set_paraphrased.json`) was added. They ask about different facts in the same chapter, are deliberately worded differently from the text (e.g. "how big is the list of words a model chooses from" for "vocabulary of about 50,000 tokens"), and each page label was checked against the PDF text.

One label in the generated set was corrected: question 9 ("What is the title of section 1.3?") was generated from the table of contents (page 2), but section 1.3 actually starts on **page 9**, which is now the expected page.

### D2 — Run the evaluation (`eval/run_evaluation.py`)

For each question in both sets (27 in total):

1. Run the **real** pipeline: `pipeline.answer_question(question)`, two-stage retrieval with the top 10 going to Gemini.
2. Mark each retrieved chunk **relevant** if it comes from the expected document and page.
3. **Hit** = at least one relevant chunk in the top 10.
4. **Reciprocal rank** = 1 / (rank of the first relevant chunk), 0 if none.
5. **Precision@10** = relevant chunks / 10.

Then, averaged over questions:

- **Recall@10** (HitRate@10): "How often did the correct page appear anywhere in the top 10?"
- **MRR@10** (Mean Reciprocal Rank): "How high up was the first correct chunk?" 1.0 means always first; 0.5 means second on average.
- **Precision@10**: "What fraction of the 10 chunks came from the correct page?"

The script waits about 4.5 s between questions to stay under the Gemini free tier's 15 requests per minute.

Writes `eval/eval_report.md` with a per-question table and a blank **"Faithful? (Y/N)"** column for **manual** judgement of faithfulness / hallucination (spec §14 says manual scoring, no RAGAS dependency).

### Results (from `eval/eval_report.md`)

Setup: 200-token chunks, 40 overlap, two-stage retrieval (BM25 + dense → RRF 25 → rerank → top 10). All three PDFs are indexed, so chunks from the other two documents also compete for the top slots. The questions are about `Chapter01_Introduction_to_AI (2).pdf`.

| Question set | Questions | Recall@10 | MRR@10 | Precision@10 |
|---|---|---|---|---|
| **All questions** | **27** | **96%** (26 of 27) | **0.91** | 23% |
| Generated from chunks | 15 | 100% | 0.93 | 23% |
| Hand-paraphrased | 12 | 92% (11 of 12) | 0.88 | 22% |

- **The one miss** is a paraphrased question: "Roughly how big is the list of words a typical large language model chooses from each time it writes the next piece of text?" (answer: about 50,000 tokens, page 20). Page 20 is mostly figure captions, and the system retrieved chunks about the toy six-word vocabulary on page 22 instead. It is a fair miss: the question shares almost no words with the answer and the meaning match was weak.
- **MRR 0.91** means the first correct chunk is usually ranked first.

**Why precision is low:** each question has exactly **one** correct page, but the system always retrieves 10 chunks. Even if 2–3 chunks come from the correct page, precision is only 20–30%. With one relevant page, precision is **low by design**. The other chunks are often *related* pages, which still help the LLM. **Recall and MRR are the more meaningful metrics here.**

**Honest caveat:** the generated questions reuse the source's wording, which makes retrieval easier (100% recall). The paraphrased set is the stricter test (92%). Both sets cover one document only.

### D3 — Retrieval benchmark (`benchmarks/benchmark_retrieval.py`)

Runs the same 27 questions through four retrieval configurations, **without Gemini**, and reports HitRate@K, MRR@K and **NDCG@K** for K = 3, 5, 10, plus **p50/p95 latency**:

1. **Dense-only**: MiniLM + ChromaDB, top 10.
2. **Sparse-only**: BM25, top 10.
3. **Hybrid**: BM25 + dense fused with RRF.
4. **Two-Stage**: Hybrid + cross-encoder reranker (what the app uses).

**NDCG@K** (Normalized Discounted Cumulative Gain) rewards putting relevant chunks near the top: each relevant chunk at rank *i* adds `1 / log2(i + 1)`, and the total is divided by the best possible score, i.e. all of that page's chunks ranked first (capped at K). 1.0 is a perfect ranking. A warm-up query runs first so model loading isn't timed, and each question is timed 3 times.

Results (all 27 questions; latency is retrieval only, on a Ryzen 5 5500U CPU):

| Configuration | HitRate@3 | HitRate@10 | MRR@10 | NDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| Dense-only | 0.778 | 0.963 | 0.677 | 0.448 | 16 | 19 |
| Sparse-only (BM25) | 0.815 | 0.926 | 0.781 | 0.432 | 1 | 2 |
| Hybrid (BM25 + Dense, RRF) | 0.889 | 0.963 | 0.847 | 0.498 | 19 | 23 |
| **Two-Stage (Hybrid + Reranker)** | **0.963** | **0.963** | **0.907** | **0.554** | 828 | 909 |

What the benchmark shows:

- **All configurations find the right page in the top 10 about equally often** (0.93–0.96). The difference is **where** it ranks.
- **Hybrid beats either search alone** (MRR 0.847 vs 0.677 dense and 0.781 BM25): the two searches fail on different questions, and RRF combines their strengths.
- **The reranker moves the right page to the top:** HitRate@3 rises from 0.778 (dense) to 0.963, and MRR from 0.677 to 0.907. In 26 of 27 questions the correct page is in the top 3.
- **The cost is latency:** about 0.8 s per question for reranking 25 candidates on CPU, compared with ~20 ms for hybrid. Gemini's answer itself takes a few seconds, so the total wait grows by roughly a quarter.
- Reranking fewer candidates was also measured: 15 candidates gave p50 453 ms with NDCG@10 0.540; 10 candidates gave 342 ms but then the reranker only reorders the same 10 chunks Gemini would get anyway. 25 was kept for the best ranking quality.
- On the **generated** questions, BM25 alone scores a high MRR (0.84) because their wording copies the text; on the **paraphrased** questions BM25's MRR drops to 0.70 and NDCG to 0.37, while two-stage stays at 0.88 MRR. That is exactly why both sets are reported.

### The 15 generated test questions

Stored in `eval/eval_set.json`; per-question results in `eval/eval_report.md`. All questions are about `Chapter01_Introduction_to_AI (2).pdf` (a 35-page lecture chapter on AI). "Found?" means the correct page was among the top 10 retrieved chunks.

| # | Question | Expected answer | Correct page | Found? |
|---|---|---|---|---|
| 1 | When was the Transformer architecture introduced by Google researchers? | 2017, in "Attention Is All You Need" | 7 | ✅ |
| 2 | What discipline is described as an umbrella term for extracting knowledge and value from data? | Data Science | 9 | ✅ |
| 3 | Who will have the customer's coffee ready? | Barista Priya | 26 | ✅ |
| 4 | What is the accuracy percentage of the system described in the text? | 96.5% | 31 | ✅ |
| 5 | Why are vendors motivated to apply the "AI-powered" label to products? | It reliably increases sales ("AI hype") | 8 | ✅ |
| 6 | What question does diagnostic analytics answer? | "Why did it happen?" | 11 | ✅ |
| 7 | What are the two mechanisms by which an AI system's behaviour can come to exist? | Machine learning and rule-based AI | 5 | ✅ |
| 8 | According to Figure 1.5, what goes into classical programming to have the program compute answers? | Rules and data | 15 | ✅ |
| 9 | What is the title of section 1.3 in the provided text? | Data Science: An Umbrella Discipline | 9 | ✅ |
| 10 | How is intelligence commonly defined in the text? | The ability to acquire and apply knowledge | 4 | ✅ |
| 11 | Is machine learning a subset of AI, or is AI a subset of ML? | ML is a subset of AI | 33 | ✅ |
| 12 | Which adaptation mechanism continues training some or all of a model's parameters on a smaller, task-specific dataset? | Fine-tuning | 25 | ✅ |
| 13 | What is Artificial Narrow Intelligence (ANI)? | AI that performs a specific task or bounded range of tasks, without transferring to novel task types | 5 | ✅ |
| 14 | How is time-series data defined in the provided text? | Tabular data with an added ordering in time (e.g. daily sales) | 17 | ✅ |
| 15 | What is the upfront requirement-gathering step called in specification-driven development (SDD)? | The specification stage | 29 | ✅ |

### The 12 hand-paraphrased test questions

Stored in `eval/eval_set_paraphrased.json`. Same document, different facts, worded so that they share as few words as possible with the text.

| # | Question | Expected answer | Correct page | Found? |
|---|---|---|---|---|
| 1 | How long did it take OpenAI's free chat interface, launched in late 2022, to reach a hundred million people? | Within two months | 8 | ✅ |
| 2 | In the Bengaluru taxi-app story, how long were passengers in HSR Layout waiting for a car before and after the change? | 2.8 → 4.2 minutes | 13 | ✅ |
| 3 | How many temperature measurements were used to learn the conversion between the two temperature scales? | Six paired readings | 16 | ✅ |
| 4 | What do we call the number of different labels a non-numeric variable is able to take? | Cardinality | 18 | ✅ |
| 5 | Roughly how big is the list of words a typical large language model chooses from each time it writes the next piece of text? | About 50,000 tokens | 20 | ❌ |
| 6 | When the six-word toy model always picks its most likely next word, what does it end up writing, and how many times does it have to run? | "The cat sat on mat", four forward passes | 22 | ✅ |
| 7 | Which methods are suggested for catching suspicious card payments when almost no confirmed cheating cases are labelled? | GMM, Isolation Forest, PCA reconstruction error | 24 | ✅ |
| 8 | What share of shoppers say they would rather text a business than phone it? | Roughly 64% | 26 | ✅ |
| 9 | In the help-desk walkthrough, who handles a customer complaining that they were billed twice for recent purchases? | AI-assisted human agent (complex query) | 28 | ✅ |
| 10 | Of all the payments the fraud checker raised an alarm on, what fraction turned out to be genuinely fraudulent? | About 33.3% | 31 | ✅ |
| 11 | Give the chapter's example of a machine that counts as artificial intelligence but has nothing to do with data science. | A line-following factory robot | 10 | ✅ |
| 12 | How many hundreds did the batter in the one-day cricket example score over their career? | 43 centuries | 12 | ✅ |

**Notable questions in the generated set:**
- **Q8** asks about a figure ("Figure 1.5"), so it exercises the figure-description part of the system.
- **Q3** is a small detail buried in a worked example, which shows retrieval works even for minor facts.
- **Q11** tests a conceptual relationship, not just a definition.

---

## 9. The Streamlit User Interface

`app.py` — run with `streamlit run app.py`.

| Part | What happens |
|---|---|
| **Startup** | `load_dotenv()` loads `GEMINI_API_KEY`; page title and wide layout set. |
| **Session state** | `chat_history` (list of `(role, text)`) and `indexed_docs` (names processed this session). Streamlit re-runs the whole script on every interaction, so session state is how data persists between re-runs. |
| **Sidebar – Upload** | `st.file_uploader(type=["pdf"], accept_multiple_files=True)` + **"Process documents"** button. For each file not already processed this session → `pipeline.ingest_pdf` inside a spinner → success message with chunk counts. |
| **Compare panel** | Shown only if `vector_store.list_documents()` returns ≥ 2 docs (reads ChromaDB, so it also works for documents indexed in earlier sessions). Two dropdowns (Doc B list excludes Doc A), optional focus text, **Compare** button. |
| **Chat** | Replays `chat_history` as chat bubbles. `st.chat_input` takes a new question → `pipeline.answer_question(question, history=chat_history[:-1])` (history excludes the question just added). Errors are caught and shown as `"Error: ..."` instead of crashing the app. |
| **Sources** | Deduplicated by `(document, page, type)`, shown in an expander with a "text" or "figure/table" label, followed by `show_latency()`: one line with the time of each stage for that question (embedding, BM25, dense, RRF, rerank, Gemini) and the total. The comparison panel shows the same line. p50/p95 are not shown in the app, because a percentile needs many queries; they come from the benchmark. |

---

## 10. Worked Example

Suppose you upload **`attention.pdf`** (10 pages; page 3 has "Figure 1: The Transformer architecture"; page 8 has "Table 2: BLEU scores").

**Ingestion**

1. pdfplumber → 10 `(page, text)` tuples.
2. `find_visual_pages` → pages **3** and **8** (caption keywords "figure" and "table"), plus any page with an embedded image.
3. Pages 3 and 8 are rendered to PNG → Gemini returns e.g.
   *Page 8: "A table with columns Model, BLEU EN-DE, BLEU EN-FR, Training cost. The Transformer (big) achieves the highest BLEU of 28.4 on EN-DE…"*
4. Chunking: page texts → say **25 text chunks**; the 2 descriptions → **2 visual chunks**.
5. All 27 chunks → MiniLM → 27 vectors of length 384 → stored in ChromaDB with metadata like `{document_name: "attention.pdf", page_number: 8, content_type: "visual"}`.
6. UI: *"Indexed attention.pdf: 25 text chunks, 2 figure/table descriptions"*.

**Query:** *"Which model has the best BLEU score?"*

1. Question → vector.
2. Dense search and BM25 each return 30 candidates. BM25 ranks chunks containing the rare word "BLEU" highly; dense search adds chunks about translation quality. RRF keeps 25, and the cross-encoder puts the table description first: e.g. visual chunk (p8), two text chunks (p8), text chunk (p9), text chunk (p7), … (10 in total).
3. Prompt contains:
   `[Figure/Table description — attention.pdf, page 8] A table with columns ... Transformer (big) achieves 28.4 ...`
4. Gemini answers: *"The Transformer (big) model achieves the best BLEU score of 28.4 on English-German (attention.pdf, page 8)."*
5. Sources expander: `attention.pdf (page 8, figure/table)`, `attention.pdf (page 8, text)`, `attention.pdf (page 9, text)` …

Without the vision step, this answer might be impossible if the table's numbers were not cleanly extractable as text.

---

## 11. Key Design Decisions

| Decision | Reasoning |
|---|---|
| **RAG instead of fine-tuning** | No training needed, works on any new PDF instantly, answers are traceable to sources, cheaper, and reduces hallucination. |
| **Convert images → text descriptions** (instead of image embeddings like CLIP) | Lets visuals use the **same** text embedding + vector DB + retrieval pipeline. One index, one search, simpler system. Also, text descriptions can be read and cited. |
| **Render whole page, not crop figures** | PyMuPDF's image extraction misses vector charts and cannot find tables. Rendering the page captures everything without a layout-detection model. |
| **Heuristic visual-page detection (images + caption keywords)** | Avoids sending *every* page to the vision API (slow + quota limits). Cheap and works well for academic PDFs that caption their figures. |
| **Cap of 15 visual pages** | Protects free-tier API quota and keeps ingestion time bounded. |
| **200-token chunks, 40 overlap, counted with MiniLM's tokenizer** | Fits completely inside MiniLM's 256-token input, so no text is lost from embeddings; small chunks are focused, which helps precision. Overlap avoids cutting ideas at boundaries. |
| **Hybrid search (BM25 + dense)** | Dense search understands paraphrases; BM25 catches exact codes, numbers and rare names that embeddings blur. They fail on different questions, so combining them is more robust than either alone. |
| **Reciprocal Rank Fusion, k = 60** | Merges two result lists using only ranks, so BM25 scores and vector distances (different scales) never have to be normalised. k = 60 is the standard value and stops one first place from dominating. |
| **30 candidates per search, 25 after fusion** | Enough recall headroom for the reranker to rescue a relevant chunk ranked low by one search, while keeping reranking to one batch. |
| **Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`)** | Reads question and passage together, so ranking is more accurate. Small enough for CPU (~0.8 s for 25 candidates). A larger reranker such as `bge-reranker-base` (~1.1 GB) would be several times slower on this laptop. |
| **Top-K = 10 after reranking** | Each chunk is small (200 tokens), so 10 chunks give the LLM about 2,000 tokens of context: enough to answer well while staying focused. |
| **BM25 rebuilt from ChromaDB, not saved to a file** | ChromaDB already stores every chunk's text. Rebuilding BM25 from it (fast for a few hundred chunks) means the keyword and vector indexes can never drift apart, and there is no pickle file to manage. |
| **Hyphen-preserving BM25 tokenizer** | `\b[\w-]+\b` keeps codes such as `XJ-4471-B` as one token, so an exact code matches only chunks with that exact code. |
| **Chunks never cross pages** | Every chunk has exactly one page number → precise citations. |
| **all-MiniLM-L6-v2** | Free, local, fast on CPU, small, widely used baseline. |
| **ChromaDB persistent client** | Zero setup (no server), metadata filters, data persists across restarts. |
| **Separate per-document search in comparison** | Guarantees balanced context from both documents. |
| **Grounded prompt + "say you don't know"** | Main defence against hallucination. |
| **Citations from metadata (UI) + from LLM (inline)** | Metadata-based sources are always correct; inline citations make the answer readable. |
| **Lazy singletons** for models, Gemini client, Chroma client | Expensive objects are created once and reused across Streamlit re-runs within the process. The ChromaDB client and the reranker are created behind one shared lock, because Streamlit sessions run on separate threads. |
| **Retry with backoff on server errors** | Free-tier Gemini sometimes returns "overloaded" errors; retrying makes the app robust. |
| **API key in `.env`, `.env` in `.gitignore`** | Security: keys never committed to source control. |
| **Modular `src/` + orchestrator** | Each module is testable independently (spec §19). Swapping a component (e.g. a different embedding model or vector DB) changes only one file. |

---

## 12. Known Limitations

What the current design does not handle, and what would fix it.

1. **Small chunks can split context.** The trade-off of 200-token chunks: an idea spread over a long paragraph can be split across chunks, so the system relies on K = 10 to bring the neighbouring pieces back together. An advanced fix is **small-to-big retrieval**: search with small chunks, then give the LLM the larger surrounding text.
2. **No text cleaning step.** Text from pdfplumber is chunked as-is (headers, footers, hyphenation, extra whitespace are not removed).
3. **No `section` metadata.** The spec lists `section` as chunk metadata; the code stores document name, page number, content type, and chunk ID only.
4. **Visual-page heuristic can miss or over-select.** A vector chart without a caption keyword is missed; a page that merely *mentions* "table" in a sentence is selected. Only the first 15 flagged pages are described.
5. **Retrieval uses only the current question.** History is given to the LLM, but the *search* uses only the latest message. A vague follow-up like "explain more" may retrieve poor chunks. Fix: rewrite the follow-up into a standalone question before retrieval.
6. **Scanned PDFs** (images of text) have no extractable text; only pages flagged as visual would get a description. No OCR is applied.
7. **Re-uploading the same file name** in a new session will reuse the same chunk IDs; ChromaDB will not re-index those IDs. Two different files with the same name would also collide.
8. **Only server errors are retried**; rate-limit (HTTP 429) errors are not.
9. **Reranking latency** — reranking 25 candidates on a laptop CPU takes about 0.8 s per question (about 1.6 s in comparison mode, which retrieves twice). A GPU, a smaller candidate set or a distilled/quantised reranker would cut this.
    The **first** question after the app starts is slower still (about 10 s in testing), because MiniLM and the cross-encoder are loaded on first use.
10. **BM25 rebuild cost** — the keyword index is rebuilt from all chunks after every upload or deletion. That is fast for a few thousand chunks, but a large collection would need an incremental index.
11. **Evaluation** covers one document; 15 questions are LLM-generated from chunks (optimistic) and only 12 are paraphrased; faithfulness is judged manually.
12. **flash-lite** gives less detailed figure descriptions than larger models.

**Future work:** small-to-big retrieval; query rewriting for follow-ups; OCR for scanned PDFs; figure cropping with a layout model; the §15 experiments (MiniLM vs BGE-small vs E5-small, 200 vs 800 token chunks) using the existing evaluation scripts.

---

## 13. Testing Strategy

Each feature has its own test script in `tests/`, so every module can be checked independently:

| Script | What it checks | Question asked |
|---|---|---|
| `test_gemini_key.py`, `list_gemini_models.py` | API key works; which Gemini models are available. | — |
| `test_phase1_manual.py`, `test_full_pipeline.py` | Builds a small PDF on the fly with **fpdf** (e.g. "The dataset used was WMT 2014 English-German…"), ingests it, asks a question, checks the answer and citation. | "What dataset was used for training?" |
| `test_phase3_multimodal.py` | Draws a **bar chart** with PIL ("Model Accuracy Comparison", bars A–D), embeds it in a PDF page with caption "Figure 1…", checks that the page is detected as visual and that Gemini's description is indexed and retrievable. | "What does the chart in Figure 1 show?" |
| `test_phase4_comparison.py` | Creates **Paper_A.pdf** (CNN, 91%) and **Paper_B.pdf** (Transformer, self-attention, 94%) — the exact example from spec §13 — and runs the comparison. | "Compare the methodologies used in both papers." |
| `test_hybrid_retrieval.py` | Runs without the database or API: the BM25 tokenizer keeps codes whole; BM25 ranks a chunk containing `ZR-5519` first where dense search ranks it 3rd of 5; the BM25 document filter; RRF scores and deduplication against hand-computed values; the reranker scores 25 chunk-sized candidates on CPU within a 1 s budget (measured ~0.8 s). | "What does fault code ZR-5519 refer to?" |
| `cleanup_test_doc.py` | Removes test documents from ChromaDB. | — |

Generating test PDFs in code means the **ground truth is known exactly**, so the output can be checked.

---

## 14. How to Run

```bash
pip install -r requirements.txt
# create .env from .env.example and set GEMINI_API_KEY=...
streamlit run app.py
```

Evaluation (after indexing a PDF through the app):

```bash
python eval/build_eval_set.py "<document_name.pdf>"
python eval/run_evaluation.py
python benchmarks/benchmark_retrieval.py      # retrieval ablation, no API calls
python tests/test_hybrid_retrieval.py         # no database or API needed
```

---

## 15. Technical Interview Questions

Q1–Q26 cover how the system works and the results. Q27–Q32 are system-design and production questions; answers that describe work not done yet say so.

**Q1. What is RAG and why did you use it?**
Retrieval-Augmented Generation: first *retrieve* relevant pieces of your own documents, then give them to an LLM to *generate* the answer. The LLM doesn't need to have been trained on your PDF. It reduces hallucination, gives citations, and new documents work instantly without retraining.

**Q2. What is an embedding?**
A fixed-length list of numbers (here 384) that represents the meaning of a text. Texts with similar meaning have vectors close to each other, so we can search by meaning rather than exact keywords.

**Q3. How does retrieval work?**
In two stages. Stage 1 runs two searches: dense search (the question's MiniLM vector against ChromaDB's HNSW index, top 30) and BM25 keyword search (top 30). Reciprocal Rank Fusion merges them: each chunk scores 1 / (60 + rank) in each list, and the best 25 are kept. Stage 2 is a cross-encoder that reads the question with each of the 25 chunks and scores relevance; the top 10 go to Gemini.

**Q4. Why chunk documents at all?**
(a) LLM prompts have a size limit and cost; (b) a single embedding of a whole document would blur many topics together; (c) small chunks allow precise retrieval and precise page citations.

**Q5. Why 200 tokens with 40 overlap?**
MiniLM only reads 256 of its own tokens and silently truncates the rest, so anything longer would be partly unsearchable. I count tokens with MiniLM's own tokenizer and use 200 tokens (plus 2 special tokens = 202 < 256), so every chunk is embedded in full. Because chunks are small, I retrieve the top 10 so the LLM still gets enough context. The 40-token overlap prevents losing a sentence split across chunk boundaries.

**Q6. How does the multimodal part work? Did you train a vision model?**
No training. Pages likely to contain figures/tables are rendered to images and sent to Gemini Vision, which returns a text description. That description is chunked, embedded, and stored like normal text, so visual information becomes searchable.

**Q7. Why not use image embeddings like CLIP?**
CLIP vectors live in a different space from MiniLM text vectors, so we'd need two indexes and a merging strategy. CLIP is also weak at reading charts and tables. Text descriptions are simpler, searchable with the same model, and readable for citations.

**Q8. How do you reduce hallucination?**
Grounded prompt ("use ONLY the context"), explicit permission to say "I don't know", mandatory citations, vision prompt saying "do not speculate", and sources shown from metadata so the user can verify.

**Q9. How are citations produced?**
Every chunk carries `document_name` and `page_number` metadata. Chunks are labelled with that info in the prompt so the LLM cites them, and the UI separately lists the retrieved chunks' sources straight from metadata.

**Q10. How does document comparison work?**
The full two-stage retrieval runs twice, once per document, with every stage filtered to that document (ChromaDB's `where` filter for dense search, a document check for BM25), giving 10 chunks per document. A structured prompt asks the LLM for Doc A points, Doc B points, and Main Differences, each with citations.

**Q11. How did you evaluate the system?**
27 questions with known correct pages: 15 generated from the chapter's text and 12 hand-paraphrased. Through the full pipeline, Recall@10 = 96% (26 of 27) and MRR@10 = 0.91; on the paraphrased set alone, 92% and 0.88. A separate benchmark compares dense-only, BM25-only, hybrid and two-stage retrieval with HitRate, MRR and NDCG at K = 3, 5, 10 and p50/p95 latency. Faithfulness is checked manually.

**Q12. Why is precision so low if recall is so high?**
Precision divides by K (10), but only one page is the "correct" one, so even with perfect retrieval it stays low (typically 10–30%). The other retrieved chunks are usually neighbouring/related pages and still useful context. That's why MRR and NDCG, which measure ranking, are reported too.

**Q13. Why Gemini and why flash-lite?**
Gemini is multimodal (text + vision with one API key) and has a usable free tier. Standard flash was limited to ~20 requests/day; flash-lite has a much larger quota. Trade-off: less detailed figure descriptions.

**Q14. What happens if Gemini is overloaded?**
`_generate_with_retry` retries server errors up to 2 more times, waiting 2 s and then 4 s.

**Q15. Why both pdfplumber and PyMuPDF?**
pdfplumber gives clean page text; PyMuPDF is fast at rendering pages to images and detecting embedded images. Each is used for what it does best.

**Q16. How is the API key kept secure?**
Stored in `.env`, loaded with python-dotenv, `.env` is in `.gitignore`; `.env.example` shows the format without the real key.

**Q17. What would you improve next?**
Small-to-big retrieval (search small chunks, give the LLM larger surrounding text), query rewriting for follow-ups, a faster reranker (GPU or a distilled model) to cut the 0.8 s, OCR for scanned PDFs, a larger multi-document paraphrased test set, and the embedding comparison (MiniLM vs BGE-small vs E5-small) using the benchmark script.

**Q18. What's the difference between keyword search and semantic search, and why use both?**
Keyword search (BM25) matches exact words: great for codes, numbers and names, but it fails on synonyms and paraphrases. Semantic search compares meaning vectors, so "What dataset was used?" can match "The model was trained on WMT 2014", but it can blur exact codes. Using both and fusing with RRF covers each one's blind spot; in the benchmark, hybrid's MRR@10 (0.847) beats dense (0.677) and BM25 (0.781).

**Q19. Why does the same embedding model have to be used for chunks and queries?**
Different models produce vectors in different spaces (even different sizes); distances between them are meaningless.

**Q20. Is anything stored permanently?**
Yes — ChromaDB data in `chroma_db/` (vectors, text, metadata). Chat history and the "indexed this session" list live only in Streamlit session state and are lost on refresh.

**Q21. On which questions did you test the system?**
27 evaluation questions about *Chapter 1: Introduction to AI* (listed in Section 8), each with a known correct page: 15 generated from the text and 12 hand-paraphrased. The correct page was retrieved for 26 of 27 (Recall@10 = 96%). They mix definitions (Q6, Q13), small details (Q3), a figure question (Q8) and conceptual relationships (Q11). Separately, each feature has its own test with a PDF built in code (Section 13): "What dataset was used for training?", "What does the chart in Figure 1 show?", and "Compare the methodologies used in both papers."

**Q22. Who wrote the evaluation questions?**
For the first 15, a script picked random text passages from the chapter and had Gemini write one question per passage; each question is linked to the page its passage came from. Their wording is close to the source text, so 12 more questions were written in different words about other facts in the chapter, with each page checked against the PDF. Results are reported for both sets separately.

**Q23. What is Reciprocal Rank Fusion and why k = 60?**
A way to merge ranked lists using only positions: score = sum of 1 / (k + rank). It avoids normalising BM25 scores against vector distances, which have different scales. k = 60 is the standard value from the original paper; it flattens the curve so a chunk ranked well by both searches beats a chunk ranked first by only one.

**Q24. What is a cross-encoder, and why not use it for the whole search?**
It reads the question and a passage together and outputs one relevance score, so it can compare words across the two directly. That is more accurate than comparing two separately computed embeddings, but it needs one model run per pair. Running it on every chunk would be far too slow, so it only reranks the 25 fused candidates.

**Q25. What do MRR and NDCG measure?**
Both measure ranking, not just whether the answer was found. MRR is 1 / rank of the first relevant chunk, averaged over questions (1.0 = always first). NDCG gives each relevant chunk credit 1 / log2(rank + 1) and divides by the best possible score, so it rewards having all relevant chunks near the top.

**Q26. How fast is retrieval?**
Embedding, dense search, BM25 and RRF together take about 20–30 ms. The cross-encoder takes about 0.8 s for 25 candidates on the laptop CPU, so two-stage retrieval has a p50 of 828 ms and a p95 of 909 ms. Gemini's answer takes a few seconds on top of that. The app shows this breakdown under every answer.

**Q27. How would this scale to millions of documents?**
Not done yet; the current design targets hundreds to a few thousand chunks. Changes in order: (1) move from embedded ChromaDB to a served vector database (or ChromaDB in client/server mode) with sharding; (2) replace the in-memory BM25 rebuild with an incremental inverted index (Elasticsearch/OpenSearch, or a sparse-vector field in the vector DB), because rebuilding from every chunk after each upload is O(corpus); (3) move ingestion into a background job queue so uploads don't block the UI, and batch embedding on a GPU; (4) add metadata filters (tenant, document) before the vector search so each query searches a small slice. The two-stage shape stays the same: cheap recall over everything, expensive reranking over a fixed 25 candidates, so reranking cost does not grow with the corpus.

**Q28. What was the hardest bug?**
A `KeyError: 'chroma_db'` that appeared only when several browser sessions opened the app at once. Streamlit serves each session on its own thread, and ChromaDB's `PersistentClient` setup is not thread-safe, so two threads creating the client at the same moment raced. The fix is a lazy singleton with double-checked locking (`INIT_LOCK` in `vector_store.py`), and the cross-encoder reuses the same lock so the model is never loaded twice concurrently.

**Q29. How would you cut the latency?**
The reranker is about 0.8 of the ~0.85 s retrieval time, so that is where to start: run it on a GPU (a few tens of ms), rerank fewer candidates (15 candidates measured at 453 ms with NDCG@10 0.540 vs 0.554), or use a quantised/distilled reranker (an ONNX quantised version was tried and changed the rankings, so it would need re-evaluation). Beyond retrieval: cache answers for repeated questions, stream Gemini's answer token by token so the first words appear sooner, and load the models at startup instead of on the first question.

**Q30. How would you monitor quality in production?**
Not done yet. Log every query with its per-stage timings (already computed) and the retrieved chunk IDs; alert on p95 latency. Track user feedback (thumbs up/down) and sample answers for an LLM-as-judge faithfulness check against the retrieved context. Keep the 27-question set (and grow it with real failed queries) as a regression test that runs on every change to chunking, models or retrieval settings, the same way `benchmarks/benchmark_retrieval.py` compared the four configurations.

**Q31. What does a query cost?**
Retrieval runs locally (MiniLM, BM25, cross-encoder on CPU), so the only paid call is Gemini: one call per question with about 2,000 tokens of context (10 chunks × 200 tokens) plus history. Ingestion costs up to 15 vision calls per PDF, which is why the visual-page cap exists. Using flash-lite keeps this within the free tier for a demo.

**Q32. What if an uploaded PDF contains instructions aimed at the LLM (prompt injection)?**
Partly handled: the prompt tells Gemini to answer only from the context, and sources are shown from metadata, so a user can see where an answer came from. Not handled: a chunk saying "ignore previous instructions" is pasted into the prompt as-is. Mitigations would be clearly delimiting retrieved text as untrusted data in the prompt, filtering instruction-like text at ingestion, and never giving the LLM tools or actions that a document could trigger.

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation: retrieve relevant document chunks, then generate an answer from them. |
| **LLM** | Large Language Model (here Gemini) that generates text. |
| **VLM / Vision-Language Model** | A model that understands images and text together (Gemini Vision). |
| **Token** | A piece of a word that models process; ~4 English characters on average. |
| **Chunk** | A small piece of a document (200 tokens here) that is embedded and retrieved. |
| **Overlap** | Tokens shared between consecutive chunks (40 here) to avoid cutting ideas. |
| **Embedding** | A numeric vector representing the meaning of a text (384 numbers here). |
| **Vector database** | A database optimised to store vectors and find the nearest ones quickly (ChromaDB). |
| **HNSW** | The graph-based approximate nearest-neighbour index ChromaDB uses. |
| **Top-K** | Number of chunks sent to the LLM after reranking (K = 10). |
| **BM25** | A keyword ranking function: scores a chunk by how often it contains the query's words, weighting rare words more and long chunks less. |
| **Dense retrieval** | Search by comparing embedding vectors (meaning), as opposed to sparse keyword search. |
| **Hybrid search** | Combining keyword (sparse) and embedding (dense) search results. |
| **RRF** | Reciprocal Rank Fusion: merges ranked lists with score = sum of 1 / (k + rank), k = 60 here. |
| **Cross-encoder / reranker** | A model that reads a query and a passage together and outputs a relevance score; used to reorder candidates. |
| **Metadata filter** | Restricting search to chunks with certain metadata (e.g. one document). |
| **Grounding** | Forcing the LLM to answer from provided context only. |
| **Hallucination** | The LLM stating something not supported by the context or reality. |
| **Recall@K / HitRate@K** | Fraction of questions where a relevant chunk appears in the top K. |
| **MRR@K** | Mean Reciprocal Rank: average of 1 / rank of the first relevant chunk in the top K (0 if none). |
| **NDCG@K** | Normalized Discounted Cumulative Gain: ranking quality where relevant chunks near the top count more; 1.0 is a perfect ranking. |
| **p50 / p95 latency** | The median and 95th-percentile response time over many queries. |
| **Precision@K** | Fraction of the top K chunks that are relevant. |
| **Faithfulness** | Whether the answer is supported by the retrieved context. |
| **Session state** | Streamlit's way of keeping variables between script re-runs. |
