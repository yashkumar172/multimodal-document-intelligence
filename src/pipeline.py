import io

from . import chunking, embeddings, gemini_client, pdf_render, pdf_utils, vector_store

MAX_VISUAL_PAGES = 15
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
        vector_store.add_chunks(all_chunks, vectors)

    return {
        "text_chunks": len(text_chunks),
        "visual_chunks": len(visual_chunks),
        "visual_pages_found": len(visual_page_numbers),
    }


def answer_question(question, top_k=TOP_K, history=None):
    query_vector = embeddings.embed_texts([question])[0]
    hits = vector_store.query(query_vector, top_k=top_k)
    if not hits:
        return "No documents indexed yet. Please upload a PDF first.", []

    answer = gemini_client.generate_answer(question, hits, history=history)
    sources = [
        {
            "document": h["metadata"]["document_name"],
            "page": h["metadata"]["page_number"],
            "type": h["metadata"].get("content_type", "text"),
        }
        for h in hits
    ]
    return answer, sources


DEFAULT_COMPARISON_QUESTION = "Compare the methodology, approach, and key findings of these two documents."


def compare_documents(document_a, document_b, question=None, top_k=TOP_K):
    question = question or DEFAULT_COMPARISON_QUESTION
    query_vector = embeddings.embed_texts([question])[0]

    hits_a = vector_store.query(query_vector, top_k=top_k, document_name=document_a)
    hits_b = vector_store.query(query_vector, top_k=top_k, document_name=document_b)

    if not hits_a or not hits_b:
        return "One or both documents have no indexed content to compare.", []

    answer = gemini_client.compare_documents(question, document_a, hits_a, document_b, hits_b)
    sources = [
        {
            "document": h["metadata"]["document_name"],
            "page": h["metadata"]["page_number"],
            "type": h["metadata"].get("content_type", "text"),
        }
        for h in hits_a + hits_b
    ]
    return answer, sources
