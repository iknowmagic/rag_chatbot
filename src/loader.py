# src/loader.py
from typing import List, Tuple

import fitz  # PyMuPDF


def load_pdf_chunks(path: str, chunk_size=500, overlap=50) -> List[Tuple[str, int]]:
    doc = fitz.open(path)
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        start = 0
        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end]
            chunks.append((chunk, page_num))
            start += chunk_size - overlap
    doc.close()
    return chunks
