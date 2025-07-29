# src/rag.py
import google.generativeai as genai

from src.config import PDF_PATH, get_google_api_key
from src.embedder import Embedder
from src.loader import load_pdf_chunks


class RAGChatbot:
    def __init__(self, model_name):
        print(f"Loading PDF and building index for model {model_name}...")
        self.texts = load_pdf_chunks(PDF_PATH)
        self.embedder = Embedder(self.texts)

        genai.configure(api_key=get_google_api_key())
        self.model = genai.GenerativeModel(model_name)
        print("Ready.")

    def ask(self, query, mode="tutor"):
        context = "\n\n---\n\n".join(self.embedder.search(query))
        prompt = f"""You are a helpful assistant answering questions about cognitive biases.
Answer this question using the following context:

{context}

Question: {query}
"""
        return self.model.generate_content(prompt).text
