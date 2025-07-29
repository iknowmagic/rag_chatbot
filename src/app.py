# src/app.py
import gradio as gr

from src.models import list_available_models
from src.rag import RAGChatbot


def main():
    bot = None

    def load_bot(selected_model):
        nonlocal bot
        bot = RAGChatbot(model_name=selected_model)
        return f"Loaded model: {selected_model}"

    def chat_with_bot(message, history):
        if not bot:
            return "Please select a model first", history
        response = bot.ask(message)
        history.append((message, response))
        return "", history

    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Cognitive Biases Chatbot")

        model_dropdown = gr.Dropdown(choices=list_available_models(), label="Select Model")
        load_button = gr.Button("Load Model")
        status = gr.Textbox(label="Status", interactive=False)

        chatbot = gr.Chatbot(label="Chat History", height=500, type="messages")
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear Chat")

        load_button.click(load_bot, inputs=[model_dropdown], outputs=[status])
        msg.submit(chat_with_bot, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: None, None, chatbot)

    demo.launch()

if __name__ == "__main__":
    main()
