import io
import time

from . import chunking, embeddings, gemini_client, pdf_render, pdf_utils, reranker, vector_store

MAX_VISUAL_PAGES = 15

# Two-stage retrieval: BM25 and dense search each return CANDIDATES_PER_RETRIEVER
# chunks, Reciprocal Rank Fusion keeps the best FUSED_CANDIDATES, and the
# cross-encoder reranks those down to the TOP_K chunks sent to Gemini.
CANDIDATES_PER_RETRIEVER = 30
RRF_K = 60
FUSED_CANDIDATES = 25
TOP_K = 10


def ingest_pdf(file_buffer, document_name):
    pdf_bytes = file_buffer.read()

    pages = pdf_utils.extract_pages(io.BytesIO(pdf_bytes))
    text_chunks = chunking.chunk_pages(pages, document_name, content_type="text")

    visual_page_numbers = pdf_render.find_visual_pages(pdf_bytes, pages)[:MAX_VISUAL_PAGES]
    visual_pages = []
    for page_number in visual_page_numbers:
        image_bytes = pdf_render.render_page_image(pdf_bytes, page_number)
        description = gemini_client.describe_image(image_bytes, page_number)
        visual_pages.append((page_number, description))
    visual_chunks = chunking.chunk_pages(visual_pages, document_name, content_type="visual")

    all_chunks = text_chunks + visual_chunks
    if all_chunks:
        texts = [c["content"] for c in all_chunks]
        vectors = embeddings.embed_texts(texts)
        # Also marks the BM25 index for a rebuild from ChromaDB on the next search.
        vector_store.add_chunks(all_chunks, vectors)

    return {
        "text_chunks": len(text_chunks),
        "visual_chunks": len(visual_chunks),
        "visual_pages_found": len(visual_page_numbers),
    }


def retrieve(
    question: str, top_k: int = TOP_K, document_name: str | None = None
) -> tuple[list[dict], dict[str, float]]:
    """
    Two-stage hybrid retrieval: BM25 + dense search fused with RRF (stage 1),
    then cross-encoder reranking (stage 2). document_name restricts every stage
    to one document. Returns the top_k chunks and per-stage latency in ms.
    """
    timings = {}

    start = time.perf_counter()
    query_vector = embeddings.embed_texts([question])[0]
    timings["embed_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    dense_hits = vector_store.query(
        query_vector, top_k=CANDIDATES_PER_RETRIEVER, document_name=document_name
    )
    timings["dense_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    bm25_hits = vector_store.bm25_query(
        question, top_k=CANDIDATES_PER_RETRIEVER, document_name=document_name
    )
    timings["bm25_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    candidates = vector_store.rrf_merge(
        [bm25_hits, dense_hits], rrf_k=RRF_K, top_n=FUSED_CANDIDATES
    )
    timings["rrf_ms"] = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    hits = reranker.rerank(question, candidates, top_n=top_k)
    timings["rerank_ms"] = (time.perf_counter() - start) * 1000

    return hits, timings


def _sources(hits):
    return [
        {
            "document": h["metadata"]["document_name"],
            "page": h["metadata"]["page_number"],
            "type": h["metadata"].get("content_type", "text"),
        }
        for h in hits
    ]


def answer_question(question, top_k=TOP_K, history=None):
    """Returns (answer, sources, timings), with timings in ms per stage."""
    hits, timings = retrieve(question, top_k=top_k)
    if not hits:
        return "No documents indexed yet. Please upload a PDF first.", [], timings

    start = time.perf_counter()
    answer = gemini_client.generate_answer(question, hits, history=history)
    timings["llm_ms"] = (time.perf_counter() - start) * 1000
    return answer, _sources(hits), timings


DEFAULT_COMPARISON_QUESTION = "Compare the methodology, approach, and key findings of these two documents."


def compare_documents(document_a, document_b, question=None, top_k=TOP_K):
    """Returns (answer, sources, timings). Each document is retrieved separately."""
    question = question or DEFAULT_COMPARISON_QUESTION

    hits_a, timings_a = retrieve(question, top_k=top_k, document_name=document_a)
    hits_b, timings_b = retrieve(question, top_k=top_k, document_name=document_b)
    timings = {k: timings_a[k] + timings_b[k] for k in timings_a}

    if not hits_a or not hits_b:
        return "One or both documents have no indexed content to compare.", [], timings

    start = time.perf_counter()
    answer = gemini_client.compare_documents(question, document_a, hits_a, document_b, hits_b)
    timings["llm_ms"] = (time.perf_counter() - start) * 1000
    return answer, _sources(hits_a + hits_b), timings
