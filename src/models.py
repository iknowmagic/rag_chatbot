import google.generativeai as genai

from src.config import get_google_api_key

genai.configure(api_key=get_google_api_key())

for m in genai.list_models():
    print(m.name, "→", m.supported_generation_methods)
