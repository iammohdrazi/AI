# app/rag_query.py
import json
import numpy as np
import requests
from app.embedder import Embedder


class RAGQuery:
    def __init__(self, index_path="index/vector_index.json"):
        self.index_path = index_path

        # ✅ Embedder ONLY for query
        self.embedder = Embedder()

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

        # ✅ Pre-load embeddings into NumPy once
        self.embeddings = np.array(
            [doc["embedding"] for doc in self.index],
            dtype=np.float32
        )

    def search(self, query, top_k=3):
        q_emb = np.array(self.embedder.embed(query), dtype=np.float32)

        # Cosine similarity (vectorized)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(q_emb)
        scores = np.dot(self.embeddings, q_emb) / norms

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(scores[i], self.index[i]) for i in top_indices]

    def ask(self, query):
        results = self.search(query)

        context = "\n\n".join(doc["text"] for _, doc in results)

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt},
            stream=True
        )

        answer = ""
        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode())
                answer += data.get("response", "")
            except:
                pass

        return answer.strip()
