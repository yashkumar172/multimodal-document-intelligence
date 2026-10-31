"""
Retrieval metrics for one question. `relevant` is the ranked list of retrieved
chunks marked True/False: a chunk is relevant if it comes from the question's
expected document and page.
"""

import json
import math
import os

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SETS = {
    "generated": os.path.join(EVAL_DIR, "eval_set.json"),
    "paraphrased": os.path.join(EVAL_DIR, "eval_set_paraphrased.json"),
}


def load_eval_sets() -> list[dict]:
    """All questions from both eval sets, each tagged with its "set" name."""
    items = []
    for name, path in EVAL_SETS.items():
        with open(path, "r", encoding="utf-8") as f:
            items.extend({**item, "set": name} for item in json.load(f))
    return items


def is_relevant(metadata: dict, item: dict) -> bool:
    return (
        metadata["document_name"] == item["document_name"]
        and metadata["page_number"] == item["expected_page"]
    )


def hit_rate_at_k(relevant: list[bool], k: int) -> float:
    """1 if any of the top k is relevant, else 0."""
    return 1.0 if any(relevant[:k]) else 0.0


def mrr_at_k(relevant: list[bool], k: int) -> float:
    """1 / rank of the first relevant chunk within the top k, else 0."""
    for rank, rel in enumerate(relevant[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def precision_at_k(relevant: list[bool], k: int) -> float:
    """Share of the top k that is relevant (divides by k even if fewer were retrieved)."""
    return sum(relevant[:k]) / k if k else 0.0


def ndcg_at_k(relevant: list[bool], k: int, total_relevant: int) -> float:
    """
    Binary-relevance NDCG. The ideal ranking puts all min(total_relevant, k)
    relevant chunks first; total_relevant is how many chunks the index holds for
    the expected page. Returns 0 when there is nothing relevant to find.
    """
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(relevant[:k], start=1) if rel)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (pct in 0-100), matching numpy's default."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
