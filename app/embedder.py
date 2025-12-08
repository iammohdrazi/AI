# app/embedder.py
import requests

class Embedder:
    def __init__(self, model_name="mxbai-embed-large"):
        # You can change to "nomic-embed-text" if you prefer
        self.model = model_name

    def embed(self, text: str):
        """Send text to Ollama for embedding."""
        response = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": self.model, "prompt": text}
        )

        data = response.json()

        if "embedding" not in data:
            raise ValueError(f"Invalid Ollama embedding response: {data}")

        return data["embedding"]


# Global embedder instance
_embedder = Embedder()  # default: "mxbai-embed-large"

def get_embedding(text: str):
    """Helper used by build_index.py"""
    return _embedder.embed(text)
