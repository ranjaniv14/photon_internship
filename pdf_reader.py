import os
import fitz
import streamlit as st
import tempfile
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector
import requests

from collections import Counter
import re

def summarize_document(chunks: List[tuple]):
    all_text = " ".join(chunk[2] for chunk in chunks)  # chunk[2] is the text
    words = re.findall(r'\b\w+\b', all_text.lower())
    stopwords = set([
        "the", "and", "to", "of", "a", "in", "for", "on", "with", "is", "that", "by", "this",
        "as", "are", "at", "an", "be", "from", "or", "it", "which", "but", "has", "have"
    ])
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    word_counts = Counter(filtered_words)
    top_terms = word_counts.most_common(10)

    prompt = f"Summarize this document in 5 sentences:\n{all_text[:3000]}"
    summary = get_ollama_response(prompt)
    return top_terms, summary


# ---------------- Settings ---------------- #
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim
DB_URL = os.getenv("DB_URL") or "postgresql://ranjanivenkataraman:yourpassword@localhost:5432/pdfdb"
engine = create_engine(DB_URL)

# ---------------- Functions ---------------- #

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
                all_data.append((page["page_number"], f"{page['page_number']}_{i+1}", chunk))
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

def build_context_prompt(query: str, chunks: List, personality:str) -> str:
    context = "\n\n".join([row.text for row in chunks])
    if personality == "Sarcastic":
        persona_intro = "You are a witty, sarcastic AI assistant who answers questions with humor, but still provides accurate and well-researched information. Think roast comedian meets high school history teacher."
    elif personality == "Academic":
        persona_intro = "You're a formal, scholarly AI assistant who provides concise and citation-style explanations. Think academic and professional"
    else:
        persona_intro = "You're a relaxed, relatable AI who chats like a helpful friend explaining things casually."

    prompt = f"""{persona_intro}
Use the following context to answer the question as accurately as possible.
    
Context:
{context}

Question: {query}

Answer:"""
    return prompt

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

# ---------------- Streamlit UI ---------------- #

st.title("📄 PDF Reader")

st.sidebar.header("🧩 Chunking")
chunk_size = st.sidebar.slider("Chunk size (words)", 100, 1000, 500, step=50)
overlap = st.sidebar.slider("Overlap (words)", 0, 300, 100, step=25)

st.sidebar.header("🧠 Chatbot Personality")
personality = st.sidebar.selectbox(
    "Choose a personality:",
    [
        "Sarcastic",
        "Academic",
        "Casual"
    ]
)


uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("Processing and embedding PDF chunks...")
    chunks = embed_and_store_chunks(tmp_path, chunk_size, overlap)
    top_terms, summary = summarize_document(chunks)

    st.subheader("🧠 PDF Summary")
    st.markdown(summary)

    st.subheader("🔤 Most Frequent Terms")
    for term, freq in top_terms:
        st.markdown(f"- **{term}**: {freq} times")

    st.success(f"✅ Stored {len(chunks)} chunks from {uploaded_file.name} into PostgreSQL.")

    # st.subheader("Sample Extracted Chunks:")
    # for page_num, chunk_id, chunk_text_content in chunks[:5]:
    #     st.markdown(f"**Page {page_num} | Chunk {chunk_id}**")
    #     st.text(chunk_text_content)
    #     st.divider()

# Search + RAG Section
st.subheader("🔍 Ask a question about the PDF")

query = st.text_input("Enter your question:")

if query:
    top_chunks = search_similar_chunks(query)
    if not top_chunks:
        st.warning("No relevant chunks found.")
    else:
        prompt = build_context_prompt(query, top_chunks, personality)
        st.info("Searching...")
        answer = get_ollama_response(prompt)

        st.subheader("💬 Answer")
        st.markdown(answer)

        st.subheader("📚 Sources")
        shown_texts = set()
        for row in top_chunks:
            if row.text in shown_texts:
                continue
            shown_texts.add(row.text)
            st.markdown(f"- **Page {row.page_number} | File: {row.filename}**")
