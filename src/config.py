# src/config.py
import os

from dotenv import load_dotenv

load_dotenv()

PDF_PATH = "data/Cognitive-Biases_V4.pdf"

def get_google_api_key():
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError("GOOGLE_API_KEY not found in .env file or environment")
    return key
