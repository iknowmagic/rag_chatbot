# src/models.py
import google.generativeai as genai

from src.config import get_google_api_key


def list_available_models():
    genai.configure(api_key=get_google_api_key())
    models = [
        m.name for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    return models

if __name__ == "__main__":
    print("Available models:")
    for name in list_available_models():
        print("-", name)
