import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

from src import pipeline


def build_test_pdf():
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    pdf.add_page()
    pdf.multi_cell(
        0,
        10,
        "The dataset used for training was the WMT 2014 English-German dataset. "
        "It contains about 4.5 million sentence pairs.",
    )
    buf = io.BytesIO(bytes(pdf.output()))
    buf.seek(0)
    return buf


def main():
    print("Ingesting test PDF...")
    n = pipeline.ingest_pdf(build_test_pdf(), "integration_test.pdf")
    print(f"  Indexed {n} chunks")

    print("Asking a question via the full RAG pipeline (real Gemini call)...")
    answer, sources = pipeline.answer_question("What dataset was used for training?")
    print("Answer:", answer)
    print("Sources:", sources)

    assert sources, "Expected at least one cited source"
    assert sources[0]["document"] == "integration_test.pdf"
    print("\nFull pipeline (ingest -> retrieve -> Gemini answer -> citations) works end-to-end.")


if __name__ == "__main__":
    main()
