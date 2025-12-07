# app/rag_query.py
import json
import numpy as np
from app.embedder import Embedder

class RAGQuery:
    def __init__(self, index_path="index.json"):
        self.index_path = index_path
        self.embedder = Embedder()

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

    def search(self, query, top_k=3):
        q_emb = self.embedder.embed(query)
        q_emb = np.array(q_emb)

        scored = []
        for item in self.index["documents"]:
            doc_emb = np.array(item["embedding"])
            score = float(np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)))
            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def ask(self, query):
        results = self.search(query)

        context = "\n\n".join([doc["text"] for _, doc in results])

        return {
            "query": query,
            "context": context,
            "answer": f"(Your LLM should answer here using context)"
        }
