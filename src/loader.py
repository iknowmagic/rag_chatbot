# src/loader.py
from typing import List

import fitz  # PyMuPDF


def load_pdf_chunks(path: str, chunk_size=500, overlap=50) -> List[str]:
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunks.append(full_text[start:end])
        start += chunk_size - overlap
    return chunks
