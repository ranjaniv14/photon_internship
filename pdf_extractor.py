import fitz  # PyMuPDF
import streamlit as st
import tempfile
import os
from typing import List, Dict


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    pages = []

    for i, page in enumerate(doc):
        page_text = page.get_text()
        pages.append({
            "filename": filename,
            "page_number": i + 1,
            "text": page_text.strip()
        })

    return pages


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap

    return chunks


def extract_pdf_chunks(pdf_path: str, chunk_size=500, overlap=100) -> List[Dict]:
    page_data = extract_text_by_page(pdf_path)
    all_chunks = []

    for page in page_data:
        chunks = chunk_text(page["text"], chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": page["filename"],
                "page_number": page["page_number"],
                "chunk_id": f"{page['page_number']}_{i + 1}",
                "text": chunk
            })

    return all_chunks


# ---------------- Streamlit UI ---------------- #

st.title("📄 PDF Reader")

# Sidebar controls
st.sidebar.header("🧩 Chunk Settings")
chunk_size = st.sidebar.slider("Chunk size (words)", min_value=100, max_value=1000, value=500, step=50)
overlap = st.sidebar.slider("Overlap size (words)", min_value=0, max_value=500, value=100, step=25)

# File upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")
    st.info(f"Extracting and chunking text (chunk size: {chunk_size}, overlap: {overlap})...")

    try:
        chunks = extract_pdf_chunks(tmp_path, chunk_size, overlap)
        st.success(f"✅ Extracted {len(chunks)} chunks from {uploaded_file.name}.")

        # Show preview
        st.subheader("Preview of Extracted Chunks:")
        for chunk in chunks[:5]:
            st.markdown(f"**Page {chunk['page_number']} | Chunk {chunk['chunk_id']}**")
            st.text(chunk['text'])
            st.divider()

    except Exception as e:
        st.error(f"❌ Error: {e}")
