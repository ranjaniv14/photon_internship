import os
import fitz
import streamlit as st
import tempfile
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
from collections import Counter
import requests
import re

# ---------------- Settings ---------------- #
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
DB_URL = os.getenv("DB_URL") or "postgresql://ranjanivenkataraman:yourpassword@localhost:5432/pdfdb"
engine = create_engine(DB_URL)

# ---------------- Functions ---------------- #

def extract_text_by_page(pdf_path: str) -> List[Dict]:
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    title = doc.metadata.get("title") or filename
    pages = []
    for i, page in enumerate(doc):
        page_text = page.get_text()
        pages.append({
            "filename": filename,
            "title": title.strip(),
            "page_number": i + 1,
            "text": page_text.strip()
        })
    return pages

def chunk_text(page_text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    words = page_text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    return chunks

def embed_and_store_chunks(pdf_path: str, chunk_size=500, overlap=100):
    page_data = extract_text_by_page(pdf_path)
    all_data = []
    with engine.connect() as conn:
        for page in page_data:
            chunks = chunk_text(page["text"], chunk_size, overlap)
            for i, chunk in enumerate(chunks):
                emb = embedding_model.encode(chunk).tolist()
                insert_sql = text("""
                    INSERT INTO pdf_chunks (filename, page_number, chunk_id, text, embedding)
                    VALUES (:filename, :page_number, :chunk_id, :text, :embedding)
                """)
                conn.execute(insert_sql, {
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "chunk_id": f"{page['page_number']}_{i+1}",
                    "text": chunk,
                    "embedding": emb
                })
                all_data.append((page["page_number"], f"{page['page_number']}_{i+1}", chunk, page["title"]))
        conn.commit()
    return all_data

def search_similar_chunks(query: str, top_k: int = 5):
    query_emb = embedding_model.encode(query).tolist()
    vector_str = f"ARRAY{query_emb}::vector"
    sql = text(f"""
        SELECT filename, page_number, chunk_id, text,
               embedding <-> {vector_str} AS distance
        FROM pdf_chunks
        ORDER BY embedding <-> {vector_str}
        LIMIT :top_k
    """)
    with engine.connect() as conn:
        results = conn.execute(sql, {"query_emb": query_emb, "top_k": top_k})
        return list(results)

def get_ollama_response(prompt: str, model: str = "llama3") -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Error contacting Ollama: {str(e)}"
    
# def chunk_list(lst, n):
#     """Split list into chunks of size n"""
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# def hierarchical_summarize(all_chunks, group_size=5):
#     # Summarize each group of chunks
#     intermediate_summaries = []
#     for group in chunk_list(all_chunks, group_size):
#         context = "\n\n".join([chunk[2] for chunk in group])
#         prompt = f"""Summarize the main content of the following document chunk:

# {context}

# Provide a concise summary.
# """
#         summary = get_ollama_response(prompt)
#         intermediate_summaries.append(summary)

#     # Summarize the summaries to get the final summary
#     final_context = "\n\n".join(intermediate_summaries)
#     final_prompt = f"""Summarize the following summaries into one concise summary:

# {final_context}

# Provide a concise summary.
# """
#     final_summary = get_ollama_response(final_prompt)
#     return final_summary


# def get_frequent_terms(text: str, top_n: int = 10):
#     words = re.findall(r'\b\w+\b', text.lower())
#     stopwords = set([
#         "the", "and", "a", "of", "in", "to", "is", "with", "on", "for", "that",
#         "by", "this", "it", "as", "at", "an", "be", "are", "from", "or"
#     ])
#     filtered = [w for w in words if w not in stopwords and len(w) > 2]
#     counter = Counter(filtered)
#     return counter.most_common(top_n)

# ---------------- Streamlit UI ---------------- #

st.title("📄 PDF Reader")

st.sidebar.header("🧩 Chunking")
chunk_size = st.sidebar.slider("Chunk size (words)", 100, 1000, 500, step=50)
overlap = st.sidebar.slider("Overlap (words)", 0, 300, 100, step=25)

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("🔄 Processing and embedding PDF chunks...")
    chunks = embed_and_store_chunks(tmp_path, chunk_size, overlap)
    st.success(f"✅ Stored {len(chunks)} chunks from {uploaded_file.name}.")

    #title = chunks[0][3] if chunks else "Unknown Title"
    full_text = " ".join(chunk[2] for chunk in chunks)
    total_words = len(full_text.split())
    #frequent_terms = get_frequent_terms(full_text)

    st.subheader("📊 Document Summary")
    #st.markdown(f"- **Title:** {title}")
    st.markdown(f"- **Total Chunks:** {len(chunks)}")
    st.markdown(f"- **Estimated Word Count:** {total_words}")

    #st.markdown("**🔑 Top Frequent Terms:**")
    #st.write({term: count for term, count in frequent_terms})

    #st.markdown("📝 **AI-Generated Summary:**")
    #summary = hierarchical_summarize(chunks[:5])  # Use top 5 chunks for summary
    #st.markdown(summary)

    st.subheader("Sample Extracted Chunks:")
    for page_num, chunk_id, chunk_text_content, _ in chunks[:5]:
        st.markdown(f"**Page {page_num} | Chunk {chunk_id}**")
        st.text(chunk_text_content)
        st.divider()

st.subheader("🔍 Ask a question about the PDF")
query = st.text_input("Enter your question:")

if query:
    top_chunks = search_similar_chunks(query)
    if not top_chunks:
        st.warning("No relevant chunks found.")
    else:
        context = "\n\n".join([row.text for row in top_chunks])
        prompt = f"""Answer the question using the following context:

{context}

Question: {query}
Answer:"""
        answer = get_ollama_response(prompt)

        st.subheader("💬 Answer")
        st.markdown(answer)

        st.subheader("📚 Sources")
        shown = set()
        for row in top_chunks:
            if row.text in shown:
                continue
            shown.add(row.text)
            st.markdown(f"- **Page {row.page_number} | File: {row.filename}**")
