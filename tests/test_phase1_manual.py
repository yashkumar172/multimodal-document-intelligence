import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpdf import FPDF

from src import chunking, embeddings, pdf_utils, vector_store


def build_test_pdf():
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)

    pdf.add_page()
    pdf.multi_cell(
        0,
        10,
        "This paper introduces the Transformer architecture. "
        "The Transformer uses self-attention instead of convolution or recurrence. "
        "It was evaluated on machine translation tasks.",
    )

    pdf.add_page()
    pdf.multi_cell(
        0,
        10,
        "The dataset used for training was the WMT 2014 English-German dataset. "
        "It contains about 4.5 million sentence pairs. "
        "The model achieved a BLEU score of 28.4.",
    )

    buf = io.BytesIO(bytes(pdf.output()))
    buf.seek(0)
    return buf


def main():
    print("Step 1: Building test PDF...")
    pdf_buffer = build_test_pdf()

    print("Step 2: Extracting text per page...")
    pages = pdf_utils.extract_pages(pdf_buffer)
    assert len(pages) == 2, f"Expected 2 pages, got {len(pages)}"
    for num, text in pages:
        print(f"  Page {num}: {text[:60]!r}...")
    assert "Transformer" in pages[0][1]
    assert "WMT 2014" in pages[1][1]
    print("  OK")

    print("Step 3: Chunking...")
    chunks = chunking.chunk_pages(pages, "test_paper.pdf")
    assert len(chunks) >= 2
    for c in chunks:
        assert set(c.keys()) == {"document_name", "page_number", "chunk_id", "content", "content_type"}
        assert c["content_type"] == "text"
    print(f"  Produced {len(chunks)} chunks")
    print("  OK")

    print("Step 4: Embedding chunks (downloads MiniLM model on first run)...")
    texts = [c["content"] for c in chunks]
    vectors = embeddings.embed_texts(texts)
    assert len(vectors) == len(chunks)
    assert len(vectors[0]) == 384, f"Expected 384-dim embeddings, got {len(vectors[0])}"
    print(f"  Got {len(vectors)} vectors of dim {len(vectors[0])}")
    print("  OK")

    print("Step 5: Storing in ChromaDB...")
    vector_store.add_chunks(chunks, vectors)
    docs = vector_store.list_documents()
    assert "test_paper.pdf" in docs
    print(f"  Indexed documents: {docs}")
    print("  OK")

    print("Step 6: Querying ChromaDB...")
    query_text = "What dataset was used for training?"
    query_vector = embeddings.embed_texts([query_text])[0]
    hits = vector_store.query(query_vector, top_k=2)
    assert len(hits) > 0
    top_hit = hits[0]
    print(f"  Top hit page: {top_hit['metadata']['page_number']}, distance: {top_hit['distance']:.4f}")
    print(f"  Top hit content: {top_hit['content'][:80]!r}")
    assert top_hit["metadata"]["page_number"] == 2, (
        f"Expected top hit to be page 2 (dataset info), got page "
        f"{top_hit['metadata']['page_number']}"
    )
    print("  OK - retrieval correctly matched the dataset question to page 2")

    print(
        "\nAll Phase 1 checks passed "
        "(PDF ingestion -> chunking -> embeddings -> ChromaDB retrieval)."
    )


if __name__ == "__main__":
    main()
