# src/app.py
import gradio as gr
import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from src.models import list_available_models
from src.rag import RAGChatbot

# Ensure required NLTK data is available
nltk.download("punkt", quiet=True)

# Example Q/A pairs from your PDF for BLEU evaluation
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
    # Add more here for richer BLEU checks
}

sample_questions = list(reference_data.keys())
smoothie = SmoothingFunction().method1  # Smoothing for BLEU


def main():
    bot = None

    def load_bot(selected_model):
        """Load the RAG chatbot with the selected model."""
        nonlocal bot
        bot = RAGChatbot(model_name=selected_model)
        return f"Loaded model: {selected_model}"

    def insert_sample(choice):
        """Insert a sample question into the message box."""
        return choice

    def chat_with_bot(message, history, compute_bleu):
        """Handle sending a message to the bot and optionally compute BLEU."""
        if not bot:
            return "Please select a model first", history, None

        try:
            response = bot.ask(message)
        except Exception as e:
            return f"Error: {str(e)}", history, None

        # Append messages in the correct format for type="messages"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})

        bleu_score = None
        if compute_bleu and message in reference_data:
            reference = [reference_data[message][0].split()]
            candidate = response.split()
            bleu_score = sentence_bleu(
                reference, candidate, smoothing_function=smoothie
            )
            bleu_score = round(bleu_score, 3)

        return "", history, bleu_score

    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Cognitive Biases Chatbot")

        # Model selection
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list_available_models(),
                label="Select Model",
                value=None
            )
            load_button = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        # Sample questions
        with gr.Row():
            sample_q = gr.Dropdown(sample_questions, label="Sample Questions")
            insert_button = gr.Button("Insert Question")

        # Chat + BLEU
        chatbot = gr.Chatbot(label="Chat History", height=300, type="messages")
        msg = gr.Textbox(label="Your message")
        with gr.Row():
            bleu_toggle = gr.Checkbox(label="Compute BLEU Accuracy", value=False)
            bleu_output = gr.Textbox(label="BLEU Score", interactive=False)
        clear = gr.Button("Clear Chat")

        # Bindings
        load_button.click(load_bot, inputs=[model_dropdown], outputs=[status])
        insert_button.click(insert_sample, inputs=[sample_q], outputs=[msg])
        msg.submit(
            chat_with_bot,
            inputs=[msg, chatbot, bleu_toggle],
            outputs=[msg, chatbot, bleu_output],
        )
        clear.click(
            lambda: (None, None, None),
            None,
            [msg, chatbot, bleu_output],
        )

    # You can set share=True if you want a public link
    demo.launch(share=False)


if __name__ == "__main__":
    main()
