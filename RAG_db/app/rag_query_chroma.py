import chromadb
import requests
from app.embedder import Embedder

COLLECTION = "rag_collection"


class RAGQueryChroma:
    def __init__(self, db_path="db/chroma", llm_model="llama3.2:1b"):
        self.embedder = Embedder()
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION
        )
        self.llm_model = llm_model

    def search(self, query, top_k=3):
        if not query.strip():
            return []

        query_embedding = self.embedder.embed(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents"]
        )

        if not results or not results.get("documents"):
            return []

        return results["documents"][0]

    def _call_llm(self, prompt: str) -> str:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            },
            timeout=120
        )

        data = response.json()

        # ✅ Correct extraction for chat API
        try:
            return data["message"]["content"].strip()
        except Exception:
            return f"[LLM ERROR] Unexpected response:\n{data}"

    def ask(self, query, top_k=3):
        docs = self.search(query, top_k=top_k)

        if not docs:
            return "No relevant information found."

        context = "\n\n".join(docs)

        prompt = f"""
    You are a helpful assistant.

    Based on the context below, answer the user's question clearly and concisely.
    Do NOT mention missing information.
    Do NOT add disclaimers.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

        return self._call_llm(prompt)

