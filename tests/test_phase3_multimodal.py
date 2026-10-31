import io
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image, ImageDraw

load_dotenv()

from src import pdf_render, pdf_utils, pipeline


def build_chart_image():
    img = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(img)
    bars = [50, 120, 90, 160]
    labels = ["A", "B", "C", "D"]
    bar_width = 60
    x = 40
    for value, label in zip(bars, labels):
        top = 250 - value
        draw.rectangle([x, top, x + bar_width, 250], fill="steelblue")
        draw.text((x + 10, 255), label, fill="black")
        x += bar_width + 20
    draw.text((100, 10), "Model Accuracy Comparison", fill="black")
    return img


def build_test_pdf_with_chart():
    chart_img = build_chart_image()

    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)

    pdf.add_page()
    pdf.multi_cell(0, 10, "Introduction. This paper compares four models: A, B, C, and D.")

    pdf.add_page()
    pdf.multi_cell(0, 10, "Figure 1 shows the accuracy comparison across models.")
    pdf.image(chart_img, x=30, y=40, w=150)

    buf = io.BytesIO(bytes(pdf.output()))
    buf.seek(0)
    return buf


def main():
    print("Building test PDF with an embedded chart on page 2...")
    pdf_buffer = build_test_pdf_with_chart()
    pdf_bytes = pdf_buffer.read()
    pdf_buffer.seek(0)

    print("Testing visual page detection...")
    pages = pdf_utils.extract_pages(io.BytesIO(pdf_bytes))
    visual_pages = pdf_render.find_visual_pages(pdf_bytes, pages)
    print("  Detected visual pages:", visual_pages)
    assert 2 in visual_pages, "Expected page 2 (with the chart) to be detected as visual"
    print("  OK")

    print("Testing page image rendering...")
    image_bytes = pdf_render.render_page_image(pdf_bytes, 2)
    assert image_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Expected valid PNG bytes"
    print(f"  Rendered {len(image_bytes)} bytes of PNG")
    print("  OK")

    print("Ingesting PDF (calls Gemini vision once for the chart page)...")
    pdf_buffer.seek(0)
    result = pipeline.ingest_pdf(pdf_buffer, "multimodal_test.pdf")
    print("  Result:", result)
    assert result["visual_chunks"] >= 1, "Expected at least one visual chunk"
    print("  OK")

    print("Asking a question about the chart via the full pipeline...")
    answer, sources, _timings = pipeline.answer_question("What does the chart in Figure 1 show?")
    print("Answer:", answer)
    print("Sources:", sources)
    assert any(s["type"] == "visual" for s in sources), "Expected a visual source to be cited"

    print("\nPhase 3 multimodal pipeline works end-to-end.")


if __name__ == "__main__":
    main()
