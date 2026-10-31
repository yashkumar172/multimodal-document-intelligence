import os
import time

from google import genai
from google.genai import errors, types

_MODEL_NAME = "gemini-flash-lite-latest"
_client = None

_IMAGE_DESCRIPTION_PROMPT = """Describe this document page image factually and specifically.

If it contains a chart, graph, histogram, or plot: state the type, the axes, the
trend or pattern shown, and any notable values or outliers.

If it contains a table: describe the columns and rows, and identify the
best/worst value if relevant.

If it contains a diagram or architecture figure: explain the structure or flow shown.

If the page has no chart, table, or diagram, say so briefly in one sentence.

Do not speculate beyond what is visible in the image."""


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _format_source_label(metadata):
    if metadata.get("content_type") == "visual":
        return f"[Figure/Table description — {metadata['document_name']}, page {metadata['page_number']}]"
    return f"[Source: {metadata['document_name']}, page {metadata['page_number']}]"


def _generate_with_retry(contents, max_retries=2):
    client = _get_client()
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(model=_MODEL_NAME, contents=contents)
        except errors.ServerError:
            if attempt == max_retries:
                raise
            time.sleep(2 * (attempt + 1))


def generate_text(prompt):
    response = _generate_with_retry(prompt)
    return response.text


def describe_image(image_bytes, page_number=None):
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    response = _generate_with_retry([_IMAGE_DESCRIPTION_PROMPT, image_part])
    return response.text


def generate_answer(question, context_chunks, history=None):
    """
    context_chunks: list of {"content", "metadata"}: the reranked chunks from pipeline.retrieve
    history: list of (role, text) tuples for prior turns, optional
    """
    context_text = "\n\n".join(
        f"{_format_source_label(c['metadata'])}\n{c['content']}"
        for c in context_chunks
    )

    history_text = ""
    if history:
        history_text = "\n".join(f"{role}: {text}" for role, text in history)

    prompt = f"""You are a document assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.
Always cite the source document and page number for any claim you make.

Conversation history:
{history_text}

Context:
{context_text}

Question: {question}

Answer (include page-number citations):"""

    response = _generate_with_retry(prompt)
    return response.text


def compare_documents(question, doc_a_name, chunks_a, doc_b_name, chunks_b):
    """
    chunks_a, chunks_b: lists of {"content", "metadata"} scoped to each document
    """
    context_a = "\n\n".join(
        f"(page {c['metadata']['page_number']}) {c['content']}" for c in chunks_a
    )
    context_b = "\n\n".join(
        f"(page {c['metadata']['page_number']}) {c['content']}" for c in chunks_b
    )

    prompt = f"""You are a document comparison assistant. Compare the two documents below
using ONLY the provided context. Cite page numbers for claims. If the context is
insufficient to answer part of the question, say so rather than guessing.

Comparison focus: {question}

Structure your answer exactly like this:

{doc_a_name}:
<key points from this document relevant to the comparison, with page citations>

{doc_b_name}:
<key points from this document relevant to the comparison, with page citations>

Main Differences:
<summary of the key differences>

Context from {doc_a_name}:
{context_a}

Context from {doc_b_name}:
{context_b}
"""

    response = _generate_with_retry(prompt)
    return response.text
