import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()

from src import pipeline


def build_pdf(text):
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    pdf.add_page()
    pdf.multi_cell(0, 10, text)
    buf = io.BytesIO(bytes(pdf.output()))
    buf.seek(0)
    return buf


def main():
    print("Ingesting Paper_A.pdf (CNN-based)...")
    paper_a = build_pdf(
        "This paper proposes a CNN based model for image classification. "
        "The architecture uses convolution layers to extract spatial features. "
        "It achieves 91% accuracy on the benchmark dataset."
    )
    pipeline.ingest_pdf(paper_a, "Paper_A.pdf")

    print("Ingesting Paper_B.pdf (Transformer-based)...")
    paper_b = build_pdf(
        "This paper proposes a Transformer based model for image classification. "
        "The architecture uses self-attention instead of convolution to model "
        "relationships between image patches. It achieves 94% accuracy on the "
        "same benchmark dataset."
    )
    pipeline.ingest_pdf(paper_b, "Paper_B.pdf")

    print("Comparing methodologies via the full pipeline...")
    answer, sources = pipeline.compare_documents(
        "Paper_A.pdf", "Paper_B.pdf", question="Compare the methodologies used in both papers."
    )
    print("\n--- Comparison Answer ---")
    print(answer)
    print("\n--- Sources ---")
    print(sources)

    assert any(s["document"] == "Paper_A.pdf" for s in sources)
    assert any(s["document"] == "Paper_B.pdf" for s in sources)
    assert "CNN" in answer or "convolution" in answer.lower()
    assert "transformer" in answer.lower() or "attention" in answer.lower()

    print("\nPhase 4 document comparison works end-to-end.")


if __name__ == "__main__":
    main()
