from sentence_transformers import CrossEncoder

from .vector_store import INIT_LOCK

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model = None


def get_model() -> CrossEncoder:
    """Load the cross-encoder once, on CPU, and reuse it across requests."""
    global _model
    if _model is None:
        # Same lock as the ChromaDB client: Streamlit sessions run on separate
        # threads and must not load the model twice at the same time.
        with INIT_LOCK:
            if _model is None:
                _model = CrossEncoder(MODEL_NAME, device="cpu", max_length=512)
    return _model


def rerank(query: str, candidates: list[dict], top_n: int = 10) -> list[dict]:
    """
    Score each (query, chunk text) pair with the cross-encoder and return the
    top_n candidates, best first, each with a "rerank_score".
    """
    if not candidates:
        return []
    scores = get_model().predict(
        [(query, c["content"]) for c in candidates], batch_size=32, show_progress_bar=False
    )
    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_n]]
