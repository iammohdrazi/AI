# scripts/build_faiss_db.py
import os
import json
import faiss
import numpy as np

JSON_INDEX = "index/vector_index.json"
FAISS_INDEX_PATH = "index/faiss.index"
META_PATH = "index/faiss_metadata.json"


def build_faiss_db():
    if os.path.exists(FAISS_INDEX_PATH):
        print("✅ FAISS DB already exists. Skipping creation.")
        return

    print("🔨 Creating FAISS DB from existing embeddings...")

    with open(JSON_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)

    embeddings = np.array(
        [item["embedding"] for item in data],
        dtype="float32"
    )

    dim = embeddings.shape[1]

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save metadata separately
    metadata = [
        {
            "file": item["file"],
            "chunk_id": item["chunk_id"],
            "text": item["text"]
        }
        for item in data
    ]

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("✅ FAISS DB created successfully")


if __name__ == "__main__":
    build_faiss_db()
