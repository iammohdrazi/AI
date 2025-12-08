# scripts/build_index.py
import os
import json

from app.pdf_loader import load_file
from app.embedder import get_embedding


DATA_DIR = "data"
INDEX_PATH = "index/vector_index.json"


def build_index():
    index = []

    print("\n=== Building RAG Index ===\n")

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        # Skip non-files
        if not os.path.isfile(file_path):
            continue

        print(f"[+] Loading: {filename}")

        try:
            text = load_file(file_path)
        except Exception as e:
            print(f"[!] Failed to load {filename}: {e}")
            continue

        if not text.strip():
            print(f"[!] Empty content skipped: {filename}")
            continue

        # Compute embedding using Ollama
        emb = get_embedding(text)

        if not isinstance(emb, list):
            raise ValueError("Embedding must be a Python list!")

        index.append({
            "file": filename,
            "text": text,
            "embedding": emb
        })

    # Save index
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    print("\nIndex built successfully!")
    print(f"Saved to: {INDEX_PATH}\n")


if __name__ == "__main__":
    build_index()
