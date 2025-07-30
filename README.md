# Cognitive Biases RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about cognitive biases using content from a reference PDF (`Cognitive-Biases_V4.pdf`).  
It combines semantic search with a large language model (Google Gemini) to generate accurate, source-backed answers.

![RAG Chatbot Diagram](RAG_Chatbot.png)

## How It Works

The chatbot follows a **Retriever → Generator** pipeline, with each stage implemented in its own Python file.

```
+-----------------------+
| PDF Loading & Chunking|
| loader.py             |
| Split into 500-char   |
| chunks                |
+----------+------------+
           |
           v
+-----------------------+
| Embedding & Indexing  |
| embedder.py           |
| all-MiniLM-L6-v2      |
| embeddings + FAISS    |
+----------+------------+
           |
           v
+-----------------------+
| Retriever             |
| rag.py                |
| Query embedding,      |
| return top chunks     |
+----------+------------+
           |
           v
+-----------------------+
| Prompt Construction   |
| rag.py                |
| Concatenate retrieved |
| chunks into context   |
+----------+------------+
           |
           v
+-----------------------+
| Generator             |
| Google Gemini         |
| Generates answer      |
+----------+------------+
           |
           v
+-----------------------+
| References + Metrics  |
| app.py                |
| Show sources, BLEU,   |
| semantic similarity   |
+-----------------------+
```

### 1. PDF Loading & Chunking — `loader.py`
- Loads the PDF with PyMuPDF.
- Splits text into **500-character chunks** with **50-character overlaps**.
- Stores the **page number** with each chunk so references can cite the exact source.

### 2. Embedding & Indexing — `embedder.py`
- Converts chunks into vector embeddings using `all-MiniLM-L6-v2` from SentenceTransformers.
- Stores embeddings in a **FAISS** index for fast semantic search.

### 3. Retriever — `rag.py`
- Encodes the user’s question into an embedding.
- Searches FAISS for the **top 5 most similar chunks**.
- Returns both text and metadata (page number, PDF name).

### 4. Prompt Construction — `rag.py`
- Concatenates the retrieved chunks into a `Context` section.
- Appends the question with instructions: *"Use ONLY the context below to answer"* to reduce hallucination.

### 5. Generator — Google Gemini
- Sends the prompt to the selected Gemini model (`list_available_models()` from `models.py`).
- Generates the answer based only on the provided context.

### 6. References + Metrics — `app.py`
- Displays retrieved sources in a collapsible **References** section (PDF name, page, snippet).
- Optional metrics:
  - **BLEU score** — Exact wording overlap with a reference answer using NLTK.
  - **Semantic similarity** — Meaning similarity using cosine similarity between embeddings.

---

## Features

- **PDF-Powered Knowledge** — Answers come from a preloaded PDF, not just the base LLM.
- **Source Transparency** — Shows exactly which part of the PDF was used.
- **Model Selection** — Choose any available Gemini model.
- **Evaluation Metrics** — BLEU and semantic similarity to assess accuracy.
- **Interactive UI** — Built with [Gradio](https://gradio.app/) for ease of use.
- **Sample Questions** — Quickly test with preloaded queries.

---

## Running Locally

1. **Install dependencies**  
```bash
pip install -r requirements.txt
```

2. **Set up your Google API key**  
Create a `.env` file in the project root with the following content:
```
GOOGLE_API_KEY=your_api_key_here
```

Alternatively, you can export it as an environment variable:
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

3. **Run the app**  
```bash
bash run.sh
```

4. **Access in browser**  
Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to use the chatbot.

---

## Acknowledgments

Built as part of [The Build Fellowship](https://www.buildfellowship.com/) projects under the guidance of [Denis Lusson](https://www.linkedin.com/in/denis-lusson/).
