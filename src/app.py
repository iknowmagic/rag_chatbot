# src/app.py
import gradio as gr

from src.rag import RAGChatbot


def main():
    bot = RAGChatbot()

    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Cognitive Biases Chatbot")
        query = gr.Textbox(label="Ask about cognitive biases:")
        output = gr.Textbox(label="Answer")
        submit = gr.Button("Ask")

        def respond(q):
            return bot.ask(q)

        submit.click(respond, inputs=[query], outputs=[output])

    demo.launch()

if __name__ == "__main__":
    main()
