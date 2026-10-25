import threading

import chromadb

_CLIENT = None
_CLIENT_LOCK = threading.Lock()
_COLLECTION_NAME = "documents"
_PERSIST_DIR = "chroma_db"


def get_collection():
    global _CLIENT
    if _CLIENT is None:
        # Streamlit serves each session on its own thread. ChromaDB's
        # PersistentClient setup is not thread-safe: concurrent creation can
        # fail with KeyError: 'chroma_db'. Create the shared client only once.
        with _CLIENT_LOCK:
            if _CLIENT is None:
                _CLIENT = chromadb.PersistentClient(path=_PERSIST_DIR)
    return _CLIENT.get_or_create_collection(_COLLECTION_NAME)


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


def query(query_embedding, top_k=10, document_name=None):
    collection = get_collection()
    if collection.count() == 0:
        return []
    where = {"document_name": document_name} if document_name else None
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where)
    hits = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"content": doc, "metadata": meta, "distance": dist})
    return hits


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


def clear_all():
    global _CLIENT
    get_collection()
    _CLIENT.delete_collection(_COLLECTION_NAME)
