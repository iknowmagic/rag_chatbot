# src/rag.py
import google.generativeai as genai

from src.config import PDF_PATH, get_google_api_key
from src.embedder import Embedder
from src.loader import load_pdf_chunks


class RAGChatbot:
    def __init__(self, model_name):
        print(f"Loading PDF and building index for model {model_name}...")
        # Now load chunks with page info
        chunks_with_pages = load_pdf_chunks(PDF_PATH)
        self.embedder = Embedder(chunks_with_pages)

        genai.configure(api_key=get_google_api_key())
        self.model = genai.GenerativeModel(model_name)
        print("Ready.")

    def ask(self, query, mode="tutor"):
        # Retrieve with metadata
        retrieved = self.embedder.search(query)
        context = "\n\n---\n\n".join(r["text"] for r in retrieved)

        prompt = f"""You are a helpful assistant answering questions about cognitive biases.
Use ONLY the context below to answer.

Context:
{context}

Question: {query}
"""
        answer = self.model.generate_content(prompt).text
        return answer, retrieved
