import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import embeddings, reranker
from src.vector_store import BM25Index, rrf_merge, tokenize

# A ticket code mentioned once in an unrelated status update. The question's
# wording ("fault code") matches the generic chunks better in meaning, so dense
# search ranks them first; BM25 matches the exact code token.
CHUNKS = [
    "The error code table lists every fault code with its meaning. Look up the fault code shown on the display.",
    "Fault codes beginning with E indicate electrical faults; codes beginning with M indicate mechanical faults.",
    "If a fault code appears, restart the device and check whether the code persists before calling support.",
    "Weekly report: the team finished the database migration. Ticket ZR-5519 is closed.",
    "Firmware updates are released monthly and include security patches.",
]
TARGET_INDEX = 3
QUERY = "What does fault code ZR-5519 refer to?"
IDS = [f"chunk_{i}" for i in range(len(CHUNKS))]
METAS = [{"document_name": "manual.pdf", "page_number": i + 1} for i in range(len(CHUNKS))]


def check_tokenizer():
    tokens = tokenize("Unit XJ-4471-B passed; see Fig. 2.")
    assert "xj-4471-b" in tokens, tokens
    print(f"PASS tokenizer keeps hyphenated codes whole: {tokens}")


def check_bm25_exact_code():
    target_id = IDS[TARGET_INDEX]
    bm25_hits = BM25Index(IDS, CHUNKS, METAS).search(QUERY, top_k=len(CHUNKS))
    assert bm25_hits[0]["id"] == target_id, [h["id"] for h in bm25_hits]

    vectors = np.array(embeddings.embed_texts(CHUNKS + [QUERY]))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = vectors[:-1] @ vectors[-1]
    dense_ranking = [IDS[i] for i in np.argsort(-similarity)]
    dense_rank = dense_ranking.index(target_id) + 1
    assert dense_rank > 1, f"dense search ranked the code chunk #{dense_rank}"
    print(
        f"PASS BM25 ranks the chunk with code ZR-5519 first; "
        f"dense search ranks it #{dense_rank} of {len(CHUNKS)}"
    )


def check_bm25_document_filter():
    # BM25Okapi gives no weight to words found in most chunks, so "solar" is
    # kept to 3 of 10 chunks, in both documents.
    texts = [
        "solar panel efficiency report", "solar panel installation guide",
        "wind turbine maintenance", "battery storage costs", "solar inverter faults",
        "grid frequency control", "hydro dam output", "nuclear plant safety",
        "coal phase-out timeline", "heat pump adoption",
    ]
    ids = [f"t{i}" for i in range(len(texts))]
    metas = [{"document_name": "a.pdf" if i % 2 else "b.pdf", "page_number": 1} for i in range(len(texts))]
    index = BM25Index(ids, texts, metas)
    unfiltered = index.search("solar panel", top_k=10)
    hits = index.search("solar panel", top_k=10, document_name="a.pdf")
    assert {h["metadata"]["document_name"] for h in unfiltered} == {"a.pdf", "b.pdf"}
    assert hits and all(h["metadata"]["document_name"] == "a.pdf" for h in hits)
    assert BM25Index([], [], []).search("anything") == []
    print(f"PASS BM25 document filter returns only a.pdf ({len(hits)} hits); empty index returns []")


def check_rrf():
    def hit(i):
        return {"id": i, "content": i, "metadata": {}}

    bm25 = [hit("a"), hit("b"), hit("c")]
    dense = [hit("c"), hit("a"), hit("d")]
    merged = rrf_merge([bm25, dense], rrf_k=60, top_n=25)

    ids = [m["id"] for m in merged]
    assert len(ids) == len(set(ids)) == 4, ids
    expected = {
        "a": 1 / 61 + 1 / 62,
        "c": 1 / 63 + 1 / 61,
        "b": 1 / 62,
        "d": 1 / 63,
    }
    for m in merged:
        assert abs(m["rrf_score"] - expected[m["id"]]) < 1e-12, m
    assert ids == ["a", "c", "b", "d"], ids
    assert len(rrf_merge([bm25, dense], top_n=2)) == 2
    print(f"PASS RRF dedupes 6 hits to 4 and orders them {ids} with the expected scores")


def check_reranker_latency(limit_ms=1000, runs=5):
    # 25 candidates of roughly chunk size (~200 tokens each). Measured at about
    # 800 ms on a Ryzen 5 5500U, so the budget is 1 s.
    filler = (
        "The model is trained on the WMT 2014 English-German dataset using "
        "multi-head self-attention, residual connections and layer normalisation. "
    )
    candidates = [
        {"id": str(i), "content": f"Passage {i}. " + filler * 8, "metadata": {}}
        for i in range(25)
    ]
    query = "What dataset was the model trained on?"
    reranker.rerank(query, candidates)  # warm-up: loads the model

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        top = reranker.rerank(query, candidates, top_n=10)
        times.append((time.perf_counter() - start) * 1000)
    assert len(top) == 10
    assert all(top[i]["rerank_score"] >= top[i + 1]["rerank_score"] for i in range(9))
    median = sorted(times)[len(times) // 2]
    tokens = len(reranker.get_model().tokenizer(query, candidates[0]["content"])["input_ids"])
    print(
        f"Cross-encoder on CPU, 25 candidates (~{tokens} tokens per pair): "
        f"median {median:.0f} ms, runs {[round(t) for t in times]}"
    )
    assert median < limit_ms, f"median {median:.0f} ms is over {limit_ms} ms"
    print(f"PASS reranking 25 candidates takes under {limit_ms} ms")


if __name__ == "__main__":
    check_tokenizer()
    check_rrf()
    check_bm25_document_filter()
    check_bm25_exact_code()
    check_reranker_latency()
    print("\nAll hybrid retrieval checks passed.")
