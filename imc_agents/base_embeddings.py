import os

import requests

from dotenv import load_dotenv

load_dotenv()


API_URL = os.getenv("SIEMENS_API_ENDPOINT_EMBEDDINGS_ADA")
API_KEY = os.getenv("SIEMENS_API_KEY")

def call_embedding(texts: list[str], model: str = "text-embedding-ada-002") -> list[list[float]]:
    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY
    }

    data = {
        "model": model,
        "input": texts
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        embeddings = response.json()["data"]
        return [item["embedding"] for item in embeddings]
    except Exception as e:
        raise RuntimeError(f"[Embedding Error] {str(e)}")