# scripts/build_chroma_db.py
import json
import os
import chromadb

INDEX_PATH = "index/vector_index.json"
CHROMA_DIR = "db/chroma"
COLLECTION = "rag_collection"

def build_chroma_db():
    print("🔨 Building Chroma DB...")

    os.makedirs(CHROMA_DIR, exist_ok=True)

    # ✅ NEW API
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(name=COLLECTION)

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("vector_index.json not found")

    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("vector_index.json is empty")

    collection.add(
        ids=[f"doc_{i}" for i in range(len(data))],
        documents=[d["text"] for d in data],
        embeddings=[d["embedding"] for d in data],
        metadatas=[
            {"file": d["file"], "chunk_id": d["chunk_id"]}
            for d in data
        ]
    )

    print("📦 Total documents in collection:", collection.count())
    print("✅ Chroma DB created successfully")

if __name__ == "__main__":
    build_chroma_db()
