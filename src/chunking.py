from transformers import AutoTokenizer

from .embeddings import MODEL_NAME

# Chunks are measured with the embedding model's own WordPiece tokenizer, so the
# size limit is exact. all-MiniLM-L6-v2 truncates input at 256 tokens including
# the [CLS]/[SEP] special tokens, so a chunk must stay at or below 254.
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Whole pages are tokenized here only to find split points, never fed to
        # the model, so silence the "sequence longer than max length" warning.
        _tokenizer.model_max_length = 10**9
    return _tokenizer


def chunk_pages(pages, document_name, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, content_type="text"):
    """
    pages: list of (page_number, text)
    Returns a list of dicts: {document_name, page_number, chunk_id, content, content_type}
    """
    tokenizer = _get_tokenizer()
    chunks = []
    chunk_id = 0
    for page_number, text in pages:
        if not text.strip():
            continue
        # Character offsets let us slice the original text, keeping its casing and
        # spacing (the MiniLM tokenizer lowercases, so decoding tokens would not).
        offsets = tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        start = 0
        while start < len(offsets):
            end = min(start + chunk_size, len(offsets))
            chunk_text = text[offsets[start][0]:offsets[end - 1][1]]
            chunks.append({
                "document_name": document_name,
                "page_number": page_number,
                "chunk_id": f"{document_name}_{content_type}_p{page_number}_c{chunk_id}",
                "content": chunk_text,
                "content_type": content_type,
            })
            chunk_id += 1
            if end == len(offsets):
                break
            start = end - overlap
    return chunks
