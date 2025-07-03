import fitz  # PyMuPDF
import os
from typing import List, Dict


def extract_text_by_page(pdf_path: str) -> List[Dict]:
    """
    Extracts text from a PDF, returns a list of dicts:
    [{page_num, text, filename}]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
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
    """
    Splits text into overlapping chunks (for embeddings).
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap  # move forward with overlap

    return chunks


def extract_pdf_chunks(pdf_path: str, chunk_size=500, overlap=100) -> List[Dict]:
    """
    Main function: Extracts, chunks, and adds metadata.
    Returns: List with filename, page, chunk_id, and text.
    """
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

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor_backend.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    chunks = extract_pdf_chunks(pdf_path)

    # Print a preview (first 2 chunks)
    print(json.dumps(chunks[:2], indent=2))
