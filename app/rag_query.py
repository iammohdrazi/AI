# app/rag_query.py
import json
import numpy as np
import requests
from app.embedder import Embedder


class RAGQuery:
    def __init__(self, index_path="index.json"):
        self.index_path = index_path
        self.embedder = Embedder()

        with open(index_path, "r", encoding="utf-8") as f:
            self.index = json.load(f)

    def search(self, query, top_k=3):
        q_emb = np.array(self.embedder.embed(query))

        scored = []
        for doc in self.index:
            doc_emb = np.array(doc["embedding"])
            score = np.dot(q_emb, doc_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(doc_emb)) #cosiene similarity
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def ask(self, query):
        results = self.search(query)
        context = "\n\n".join([doc["text"] for _, doc in results])

        prompt = f"""
    Use the context below to answer the question clearly and accurately.

    Context:
    {context}

    Question: {query}

    Answer:
    """

        # STREAMING RESPONSE FIX
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt},
            stream=True
        )

        answer = ""
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        answer += data["response"]
                except:
                    continue

        return {
            "query": query,
            "context": context,
            "answer": answer.strip()
        }

