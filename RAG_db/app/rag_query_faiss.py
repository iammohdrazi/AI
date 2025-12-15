# app/rag_query_faiss.py
import json
import faiss
import numpy as np
import requests
from app.embedder import Embedder

class RAGQueryFAISS:
    def __init__(self):
        self.embedder = Embedder()

        # Load FAISS index
        self.index = faiss.read_index("index/faiss.index")

        # Load metadata
        with open("index/faiss_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query, top_k=3):
        q_emb = np.array(
            [self.embedder.embed(query)],
            dtype="float32"
        )

        # Normalize for cosine similarity
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.metadata[idx]
            results.append((score, doc))

        return results

    def ask(self, query):
        results = self.search(query)

        context = "\n\n".join(doc["text"] for _, doc in results)

        prompt = f"""
Use the context below to answer the question clearly.

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

        return {
            "answer": answer.strip(),
            "context": context
        }
