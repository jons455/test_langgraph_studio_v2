from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from langchain_core.messages import HumanMessage
import requests

from dotenv import load_dotenv
import os

load_dotenv()


# === Konfiguration ===
API_URL = os.getenv("SIEMENS_API_ENDPOINT")  # Ersetze mit deiner URL
API_KEY = os.getenv("SIEMENS_API_KEY")  # Ersetze mit deinem tatsächlichen API-Key

# === Richtige Header ===
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

# === Anfrage-Daten ===
data = {
    "model": "GPT-4o",  # oder das interne Modell, das euch zur Verfügung steht
    "messages": [
        {"role": "user", "content": "Hello, world! Antworte mit 123"}
    ],
    "temperature": 0.7
}

# === Anfrage senden ===
try:
    response = requests.post(API_URL, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    print("Antwort des Modells:", result['choices'][0]['message']['content'])
except requests.exceptions.HTTPError as err:
    print("HTTP-Fehler:", err)
    print("Antwort:", response.text)
except Exception as e:
    print("Allgemeiner Fehler:", e)


