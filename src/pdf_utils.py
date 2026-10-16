import pdfplumber


def extract_pages(file_path_or_buffer):
    """Extract text from a PDF. Returns a list of (page_number, text) tuples, 1-indexed."""
    pages = []
    with pdfplumber.open(file_path_or_buffer) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages
