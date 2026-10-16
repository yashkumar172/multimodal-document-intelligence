import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src import gemini_client


def main():
    print("Testing Gemini API key...")
    try:
        answer = gemini_client.generate_answer(
            question="Reply with exactly one word: OK",
            context_chunks=[
                {
                    "content": "This is a connectivity test.",
                    "metadata": {"document_name": "test", "page_number": 1},
                }
            ],
        )
        print("SUCCESS - Gemini responded:")
        print(answer)
    except Exception as e:
        print("FAILED -", type(e).__name__, str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
