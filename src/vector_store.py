import re
import threading

import chromadb
from rank_bm25 import BM25Okapi

_CLIENT = None
# Shared by every one-time setup that Streamlit's session threads could race on:
# the ChromaDB client here and the cross-encoder in reranker.py.
INIT_LOCK = threading.Lock()
_COLLECTION_NAME = "documents"
_PERSIST_DIR = "chroma_db"

# Keeps word-internal hyphens, so codes like "XJ-4471-B" stay one token.
_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")

_bm25_index = None
_BM25_LOCK = threading.Lock()


def get_collection():
    global _CLIENT
    if _CLIENT is None:
        # Streamlit serves each session on its own thread. ChromaDB's
        # PersistentClient setup is not thread-safe: concurrent creation can
        # fail with KeyError: 'chroma_db'. Create the shared client only once.
        with INIT_LOCK:
            if _CLIENT is None:
                _CLIENT = chromadb.PersistentClient(path=_PERSIST_DIR)
    return _CLIENT.get_or_create_collection(_COLLECTION_NAME)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory BM25Okapi index over chunks, with per-document filtering."""

    def __init__(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        # BM25Okapi cannot be built from an empty corpus.
        self._bm25 = BM25Okapi([tokenize(d) for d in documents]) if documents else None

    def search(self, query: str, top_k: int = 30, document_name: str | None = None) -> list[dict]:
        """Return up to top_k chunks with a positive BM25 score, best first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if document_name and self.metadatas[i]["document_name"] != document_name:
                continue
            hits.append({
                "id": self.ids[i],
                "content": self.documents[i],
                "metadata": self.metadatas[i],
                "bm25_score": float(scores[i]),
            })
            if len(hits) == top_k:
                break
        return hits


def _get_bm25_index() -> BM25Index:
    """Build the BM25 index from ChromaDB on first use, so the two never drift."""
    global _bm25_index
    if _bm25_index is None:
        with _BM25_LOCK:
            if _bm25_index is None:
                data = get_collection().get()
                _bm25_index = BM25Index(data["ids"], data["documents"], data["metadatas"])
    return _bm25_index


def _invalidate_bm25():
    # Rebuilt from ChromaDB on the next search.
    global _bm25_index
    with _BM25_LOCK:
        _bm25_index = None


def add_chunks(chunks, embeddings):
    collection = get_collection()
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in chunks],
        metadatas=[
            {
                "document_name": c["document_name"],
                "page_number": c["page_number"],
                "content_type": c.get("content_type", "text"),
            }
            for c in chunks
        ],
    )
    _invalidate_bm25()


def query(query_embedding, top_k=10, document_name=None):
    """Dense search: the top_k nearest chunks in ChromaDB's HNSW index."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    where = {"document_name": document_name} if document_name else None
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    hits = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"id": chunk_id, "content": doc, "metadata": meta, "distance": dist})
    return hits


def bm25_query(query_text: str, top_k: int = 30, document_name: str | None = None) -> list[dict]:
    """Sparse search: the top_k chunks by BM25 keyword score."""
    return _get_bm25_index().search(query_text, top_k=top_k, document_name=document_name)


def rrf_merge(ranked_lists: list[list[dict]], rrf_k: int = 60, top_n: int = 25) -> list[dict]:
    """
    Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (rrf_k + rank of d),
    with ranks starting at 1. Chunks are deduplicated by id; the top_n are returned
    best first, each with an "rrf_score".
    """
    scores = {}
    chunks = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank)
            chunks.setdefault(hit["id"], hit)
    best = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [{**chunks[i], "rrf_score": scores[i]} for i in best]


def get_document_chunks(document_name, content_type=None):
    collection = get_collection()
    where = {"document_name": document_name}
    if content_type:
        where = {"$and": [{"document_name": document_name}, {"content_type": content_type}]}
    results = collection.get(where=where)
    return [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(results["documents"], results["metadatas"])
    ]


def list_documents():
    collection = get_collection()
    if collection.count() == 0:
        return []
    metadatas = collection.get()["metadatas"]
    return sorted({m["document_name"] for m in metadatas})


def delete_document(document_name):
    get_collection().delete(where={"document_name": document_name})
    _invalidate_bm25()


def clear_all():
    global _CLIENT
    get_collection()
    _CLIENT.delete_collection(_COLLECTION_NAME)
    _invalidate_bm25()
