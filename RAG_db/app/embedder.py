# app/embedder.py
import requests


class Embedder:
    def __init__(self, model_name="mxbai-embed-large"):
        self.model = model_name

    def embed(self, text: str):
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={
                "model": self.model,
                "prompt": text
            }
        )

        data = response.json()

        if "embedding" not in data:
            raise ValueError(f"Invalid embedding response: {data}")

        return data["embedding"]
