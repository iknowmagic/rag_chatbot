# src/embedder.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, chunks_with_pages):
        self.pdf_name = "Cognitive-Biases_V4.pdf"
        self.texts = [c[0] for c in chunks_with_pages]
        self.pages = [c[1] for c in chunks_with_pages]
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self.model.encode(self.texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query, k=5):
        query_vec = self.model.encode([query])
        _, indices = self.index.search(query_vec, k)
        results = []
        for i in indices[0]:
            results.append({
                "text": self.texts[i],
                "page": self.pages[i],
                "pdf": self.pdf_name
            })
        return results
