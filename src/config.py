# src/config.py
import json

PDF_PATH = "data/Cognitive-Biases_V4.pdf"

# In Colab, userdata.json exists
def get_google_api_key():
    try:
        with open("/content/userdata.json") as f:
            userdata = json.load(f)
        return userdata.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        return None  # or os.getenv("GOOGLE_API_KEY")
