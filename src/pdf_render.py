import pymupdf

FIGURE_KEYWORDS = ("figure", "fig.", "table", "chart", "graph", "diagram")


def find_visual_pages(pdf_bytes, pages):
    """
    pages: list of (page_number, text) from pdf_utils.extract_pages
    Returns 1-indexed page numbers likely to contain a figure, graph, table, or diagram,
    based on embedded raster images or caption keywords in the page text.
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    visual_pages = []
    for page_number, text in pages:
        has_embedded_image = len(doc[page_number - 1].get_images(full=True)) > 0
        has_caption_keyword = any(kw in text.lower() for kw in FIGURE_KEYWORDS)
        if has_embedded_image or has_caption_keyword:
            visual_pages.append(page_number)
    doc.close()
    return visual_pages


def render_page_image(pdf_bytes, page_number, dpi=150):
    """page_number is 1-indexed. Returns PNG bytes of the rendered page."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    zoom = dpi / 72
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes
