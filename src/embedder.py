# src/embedder.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, texts):
        self.texts = texts
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def search(self, query, k=5):
        query_vec = self.model.encode([query])
        _, indices = self.index.search(query_vec, k)
        return [self.texts[i] for i in indices[0]]
