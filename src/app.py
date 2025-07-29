# src/app.py
import gradio as gr

from src.rag import RAGChatbot


def main():
    bot = RAGChatbot()

    def chat_with_bot(message, history):
        # history is a list of [user, bot] pairs
        response = bot.ask(message)
        history.append((message, response))
        return "", history

    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 Cognitive Biases Chatbot")

        chatbot = gr.Chatbot(label="Chat History", height=500)
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear Chat")

        msg.submit(chat_with_bot, inputs=[msg, chatbot], outputs=[msg, chatbot])
        clear.click(lambda: None, None, chatbot)

    demo.launch()

if __name__ == "__main__":
    main()
