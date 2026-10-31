import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from metrics import hit_rate_at_k, is_relevant, load_eval_sets, mrr_at_k, precision_at_k

from src import pipeline

REPORT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.md")
K = pipeline.TOP_K
# The Gemini free tier allows 15 requests per minute; stay under it.
SECONDS_PER_QUESTION = 4.5


def summarize(results):
    n = len(results)
    if not n:
        return {"n": 0, "hit": 0.0, "mrr": 0.0, "precision": 0.0}
    return {
        "n": n,
        "hit": sum(r["hit"] for r in results) / n,
        "mrr": sum(r["mrr"] for r in results) / n,
        "precision": sum(r["precision"] for r in results) / n,
    }


def main():
    eval_set = load_eval_sets()
    results = []

    for i, item in enumerate(eval_set, start=1):
        print(f"[{i}/{len(eval_set)}] ({item['set']}) {item['question']}")
        started = time.monotonic()
        answer, sources, _timings = pipeline.answer_question(item["question"])
        relevant = [
            is_relevant({"document_name": s["document"], "page_number": s["page"]}, item)
            for s in sources
        ]
        retrieved_pages = [s["page"] for s in sources if s["document"] == item["document_name"]]
        results.append({
            **item,
            "generated_answer": answer,
            "retrieved_pages": retrieved_pages,
            "hit": hit_rate_at_k(relevant, K),
            "mrr": mrr_at_k(relevant, K),
            "precision": precision_at_k(relevant, K),
        })
        time.sleep(max(0.0, SECONDS_PER_QUESTION - (time.monotonic() - started)))

    rows = [
        ("All questions", summarize(results)),
        ("Generated from chunks", summarize([r for r in results if r["set"] == "generated"])),
        ("Hand-paraphrased", summarize([r for r in results if r["set"] == "paraphrased"])),
    ]
    for label, s in rows:
        print(f"{label}: Recall@{K} {s['hit']:.0%}, MRR@{K} {s['mrr']:.2f}, Precision@{K} {s['precision']:.0%}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Evaluation Report\n\n")
        f.write(
            "Retrieval: BM25 + dense (top 30 each) fused with RRF (k = 60, top 25), "
            f"then reranked by the `{pipeline.reranker.MODEL_NAME}` cross-encoder; "
            f"the top {K} chunks go to Gemini.\n\n"
            "A retrieved chunk counts as relevant if it comes from the question's expected page.\n\n"
        )
        f.write(f"| Question set | Questions | Recall@{K} | MRR@{K} | Precision@{K} |\n")
        f.write("|---|---|---|---|---|\n")
        for label, s in rows:
            f.write(f"| {label} | {s['n']} | {s['hit']:.0%} | {s['mrr']:.2f} | {s['precision']:.0%} |\n")
        f.write(
            "\nFaithfulness/Hallucination column is left blank for manual review "
            "(compare the generated answer against the expected answer and mark Y/N).\n\n"
        )
        f.write("| # | Set | Question | Expected Page | Retrieved Pages | Hit | Generated Answer | Faithful? (Y/N) |\n")
        f.write("|---|-----|----------|---------------|-----------------|-----|------------------|-----------------|\n")
        for i, r in enumerate(results, start=1):
            answer_short = r["generated_answer"].replace("\n", " ").replace("|", "/")[:200]
            f.write(
                f"| {i} | {r['set']} | {r['question']} | {r['expected_page']} | "
                f"{r['retrieved_pages']} | {'Yes' if r['hit'] else 'No'} | {answer_short} | |\n"
            )

    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
