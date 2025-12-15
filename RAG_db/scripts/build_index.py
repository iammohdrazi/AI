# scripts/build_index.py
import os
import json

from app.pdf_loader import load_file
from app.embedder import Embedder

DATA_DIR = "data"
INDEX_PATH = "index/vector_index.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap

    return chunks


def build_index():
    index = []
    embedder = Embedder()  # ✅ Used ONLY here

    print("\n=== Building RAG Index (Embeddings Generated ONCE) ===\n")

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        print(f"[+] Loading: {filename}")

        text = load_file(file_path)
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            embedding = embedder.embed(chunk)

            index.append({
                "file": filename,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\n✅ Index saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
