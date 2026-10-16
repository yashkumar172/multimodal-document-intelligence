import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src import pipeline

EVAL_SET_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "eval_report.md")


def main():
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        eval_set = json.load(f)

    results = []
    hits = 0
    precisions = []

    for i, item in enumerate(eval_set, start=1):
        print(f"[{i}/{len(eval_set)}] {item['question']}")
        answer, sources = pipeline.answer_question(item["question"])
        doc_sources = [s for s in sources if s["document"] == item["document_name"]]
        retrieved_pages = [s["page"] for s in doc_sources]
        relevant = [p for p in retrieved_pages if p == item["expected_page"]]
        hit = len(relevant) > 0
        precision = len(relevant) / len(sources) if sources else 0.0
        hits += int(hit)
        precisions.append(precision)

        results.append({
            **item,
            "generated_answer": answer,
            "retrieved_pages": retrieved_pages,
            "hit": hit,
            "precision_at_k": round(precision, 2),
        })

    recall_at_k = hits / len(eval_set) if eval_set else 0.0
    avg_precision = sum(precisions) / len(precisions) if precisions else 0.0

    print(f"\nRecall@K (correct page retrieved): {recall_at_k:.0%}")
    print(f"Average Precision@K: {avg_precision:.0%}")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Phase 2 Evaluation Report\n\n")
        f.write(f"- Questions evaluated: {len(eval_set)}\n")
        f.write(f"- Recall@K (correct page retrieved): {recall_at_k:.0%}\n")
        f.write(f"- Average Precision@K: {avg_precision:.0%}\n\n")
        f.write(
            "Faithfulness/Hallucination column is left blank for manual review "
            "(compare the generated answer against the expected answer and mark Y/N).\n\n"
        )
        f.write("| # | Question | Expected Page | Retrieved Pages | Hit | Generated Answer | Faithful? (Y/N) |\n")
        f.write("|---|----------|---------------|------------------|-----|-------------------|------------------|\n")
        for i, r in enumerate(results, start=1):
            answer_short = r["generated_answer"].replace("\n", " ").replace("|", "/")[:200]
            f.write(
                f"| {i} | {r['question']} | {r['expected_page']} | "
                f"{r['retrieved_pages']} | {'Yes' if r['hit'] else 'No'} | {answer_short} | |\n"
            )

    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
