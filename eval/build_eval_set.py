import json
import os
import random
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src import gemini_client, vector_store

NUM_QUESTIONS = 15
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "eval_set.json")

QUESTION_GEN_PROMPT_TEMPLATE = """Based ONLY on the following text, write exactly one factual
question that can be answered using this text, along with its correct answer.
The question should be answerable in a sentence or two.

Text:
\"\"\"
{content}
\"\"\"

Respond with ONLY valid JSON in this exact format, no markdown fences, no extra text:
{{"question": "...", "expected_answer": "..."}}
"""


def generate_qa_for_chunk(chunk):
    prompt = QUESTION_GEN_PROMPT_TEMPLATE.format(content=chunk["content"])
    text = gemini_client.generate_text(prompt).strip()
    text = re.sub(r"^```json|^```|```$", "", text, flags=re.MULTILINE).strip()
    data = json.loads(text)
    return {
        "question": data["question"],
        "expected_answer": data["expected_answer"],
        "expected_page": chunk["metadata"]["page_number"],
        "document_name": chunk["metadata"]["document_name"],
    }


def main():
    documents = vector_store.list_documents()
    if not documents:
        print("No documents indexed yet. Upload a PDF via the app first.")
        sys.exit(1)

    if len(sys.argv) > 1:
        document_name = sys.argv[1]
    elif len(documents) == 1:
        document_name = documents[0]
    else:
        print("Multiple documents indexed:", documents)
        print('Specify which one: python eval/build_eval_set.py "<document_name>"')
        sys.exit(1)

    all_chunks = vector_store.get_document_chunks(document_name)
    chunks = [c for c in all_chunks if c["metadata"].get("content_type", "text") != "visual"]
    if not chunks:
        print(f"No text chunks found for '{document_name}'.")
        sys.exit(1)

    sample_size = min(NUM_QUESTIONS, len(chunks))
    sampled = random.sample(chunks, sample_size)

    eval_set = []
    for i, chunk in enumerate(sampled, start=1):
        page = chunk["metadata"]["page_number"]
        print(f"Generating question {i}/{sample_size} (page {page})...")
        try:
            qa = generate_qa_for_chunk(chunk)
            eval_set.append(qa)
        except Exception as e:
            print(f"  Skipped (error: {e})")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)

    print(f"\nSaved {len(eval_set)} Q&A pairs to {OUTPUT_PATH}")
    print("Review/edit this file if any questions look off before running the evaluation.")


if __name__ == "__main__":
    main()
