import os
import sys

import streamlit as st
from dotenv import load_dotenv

sys.path.append(os.path.dirname(__file__))
load_dotenv()

from src import pipeline, vector_store

st.set_page_config(page_title="Document Intelligence Assistant", layout="wide")
st.title("Multimodal Document Intelligence Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (role, text)
if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )
    if uploaded_files and st.button("Process documents"):
        for f in uploaded_files:
            if f.name in vector_store.list_documents():
                continue
            with st.spinner(f"Processing {f.name} (text + figures/tables)..."):
                result = pipeline.ingest_pdf(f, f.name)
            st.session_state.indexed_docs.append(f.name)
            st.success(
                f"Indexed {f.name}: {result['text_chunks']} text chunks, "
                f"{result['visual_chunks']} figure/table descriptions"
            )

    stored_docs = vector_store.list_documents()
    if stored_docs:
        st.subheader("Indexed documents")
        for name in stored_docs:
            c1, c2 = st.columns([4, 1])
            c1.write(f"- {name}")
            if c2.button("✕", key=f"del_{name}", help=f"Remove {name}"):
                vector_store.delete_document(name)
                if name in st.session_state.indexed_docs:
                    st.session_state.indexed_docs.remove(name)
                st.rerun()
        if st.button("Clear all documents"):
            vector_store.clear_all()
            st.session_state.indexed_docs = []
            st.session_state.chat_history = []
            st.rerun()

available_docs = vector_store.list_documents()
if len(available_docs) >= 2:
    with st.expander("Compare Documents"):
        col1, col2 = st.columns(2)
        with col1:
            doc_a = st.selectbox("Document A", available_docs, key="compare_doc_a")
        with col2:
            doc_b = st.selectbox(
                "Document B",
                [d for d in available_docs if d != doc_a],
                key="compare_doc_b",
            )
        comparison_question = st.text_input(
            "Comparison focus (optional)",
            placeholder=pipeline.DEFAULT_COMPARISON_QUESTION,
        )
        if st.button("Compare"):
            with st.spinner(f"Comparing {doc_a} vs {doc_b}..."):
                comparison_answer, comparison_sources = pipeline.compare_documents(
                    doc_a, doc_b, question=comparison_question or None
                )
            st.markdown(comparison_answer)
            if comparison_sources:
                unique = list(
                    {(s["document"], s["page"], s["type"]): s for s in comparison_sources}.values()
                )
                with st.expander("Sources used in comparison"):
                    for s in unique:
                        label = "figure/table" if s["type"] == "visual" else "text"
                        st.write(f"- {s['document']} (page {s['page']}, {label})")

st.subheader("Ask a question")

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

question = st.chat_input("Ask something about your uploaded documents...")
if question:
    st.session_state.chat_history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = pipeline.answer_question(
                    question, history=st.session_state.chat_history[:-1]
                )
            except Exception as e:
                answer = f"Error: {e}"
                sources = []
            st.markdown(answer)
            if sources:
                unique_sources = list({(s["document"], s["page"], s["type"]): s for s in sources}.values())
                with st.expander("Sources"):
                    for s in unique_sources:
                        label = "figure/table" if s["type"] == "visual" else "text"
                        st.write(f"- {s['document']} (page {s['page']}, {label})")
    st.session_state.chat_history.append(("assistant", answer))
