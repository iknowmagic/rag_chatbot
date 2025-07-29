# Cognitive Biases RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about cognitive biases using content from a reference PDF (`Cognitive-Biases_V4.pdf`).  
It combines semantic search with a large language model (Google Gemini) to generate accurate, source-backed answers.

## Features

- **PDF-Powered Knowledge** — Answers are based on a preloaded PDF, not just the base LLM.
- **Source Transparency** — References section shows the PDF name, page numbers, and exact matched snippets.
- **Model Selection** — Choose any available Google Generative AI model.
- **Evaluation Metrics**
  - **BLEU score** — measures word overlap with reference answers.
  - **Semantic similarity** — measures meaning similarity using SBERT embeddings.
- **Interactive UI** — Built with [Gradio](https://gradio.app/) for an easy web-based interface.
- **Sample Questions** — Quickly test with preloaded example queries.

## 📂 Project Structure
