# app/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        emb = self.model.encode(text)
        return emb.tolist() if isinstance(emb, np.ndarray) else emb
