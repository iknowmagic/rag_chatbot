# src/app.py
import gradio as gr
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util

from src.models import list_available_models
from src.rag import RAGChatbot

# Ensure required NLTK data is available
nltk.download("punkt", quiet=True)

# Load SBERT model once for semantic similarity
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# Example Q/A pairs from your PDF for evaluation
reference_data = {
    "What is confirmation bias?": [
        "the tendency to search for, interpret, or remember information in a way that confirms one's preconceptions"
    ],
    "What is anchoring bias?": [
        "relying too heavily on the first piece of information encountered when making decisions"
    ],
    "What is availability heuristic?": [
        "estimating the likelihood of events based on how easily examples come to mind"
    ],
}

sample_questions = list(reference_data.keys())
smoothie = SmoothingFunction().method1  # BLEU smoothing


def main():
    bot = None

    def load_bot(selected_model):
        nonlocal bot
        bot = RAGChatbot(model_name=selected_model)
        return f"Loaded model: {selected_model}"

    def insert_sample(choice):
        return choice

    def chat_with_bot(message, history, compute_bleu, compute_semantic):
        if not bot:
            return "Please select a model first", history, None, None, ""

        try:
            answer, sources = bot.ask(message)
        except Exception as e:
            return f"Error: {str(e)}", history, None, None, ""

        # Append messages for type="messages"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})

        # Format sources into markdown for the accordion
        refs_markdown = "\n".join(
            [f"**{src['pdf']} - Page {src['page']}**\n> {src['text']}" for src in sources]
        )

        # BLEU and Semantic scores
        bleu_score = None
        semantic_score = None
        if message in reference_data:
            ref_text = reference_data[message][0]
            if compute_bleu:
                reference = [ref_text.split()]
                candidate = answer.split()
                bleu_score = round(
                    sentence_bleu(reference, candidate, smoothing_function=smoothie), 3
                )
            if compute_semantic:
                emb_ref = semantic_model.encode(ref_text, convert_to_tensor=True)
                emb_resp = semantic_model.encode(answer, convert_to_tensor=True)
                semantic_score = round(float(util.cos_sim(emb_ref, emb_resp)), 3)

        return "", history, bleu_score, semantic_score, refs_markdown

    with gr.Blocks(css=".gradio-container {font-family: Arial, sans-serif;}") as demo:
        gr.Markdown("## 🧠 Cognitive Biases Chatbot\nAsk about cognitive biases and check accuracy.")

        # Model selection
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list_available_models(), label="Select Model"
            )
            load_button = gr.Button("Load Model", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

        # Sample questions
        with gr.Row():
            sample_q = gr.Dropdown(sample_questions, label="Sample Questions")
            insert_button = gr.Button("Insert Question")

        # Chat and evaluation controls
        chatbot = gr.Chatbot(label="Chat History", height=300, type="messages")
        msg = gr.Textbox(label="Your message")

        with gr.Row():
            bleu_toggle = gr.Checkbox(label="Compute BLEU Accuracy", value=False)
            semantic_toggle = gr.Checkbox(label="Compute Semantic Similarity", value=False)

        with gr.Row():
            bleu_output = gr.Textbox(label="BLEU Score", interactive=False)
            semantic_output = gr.Textbox(label="Semantic Score", interactive=False)

        # References accordion declared here
        with gr.Accordion("References", open=False):
            refs_box = gr.Markdown("")

        clear = gr.Button("Clear Chat", variant="secondary")

        # Bindings
        load_button.click(load_bot, inputs=[model_dropdown], outputs=[status])
        insert_button.click(insert_sample, inputs=[sample_q], outputs=[msg])
        msg.submit(
            chat_with_bot,
            inputs=[msg, chatbot, bleu_toggle, semantic_toggle],
            outputs=[msg, chatbot, bleu_output, semantic_output, refs_box],
        )
        clear.click(
            lambda: (None, None, None, None, ""),
            None,
            [msg, chatbot, bleu_output, semantic_output, refs_box],
        )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
