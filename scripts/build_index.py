# scripts/build_index.py
import os
import json

from app.pdf_loader import load_file
from app.embedder import get_embedding

DATA_DIR = "data"
INDEX_PATH = "index/vector_index.json"

# --- Chunking settings ---
CHUNK_SIZE = 500      # Approx number of characters per chunk
CHUNK_OVERLAP = 50    # Overlap characters between chunks for context

# CHUNK_SIZE = 100
# CHUNK_OVERLAP = 20

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + size, text_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap  # move start forward but keep overlap
    return chunks


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

        # --- Chunk the text ---
        text_chunks = chunk_text(text)

        for i, chunk in enumerate(text_chunks):
            # Compute embedding using Ollama
            emb = get_embedding(chunk)

            if not isinstance(emb, list):
                raise ValueError("Embedding must be a Python list!")

            index.append({
                "file": filename,
                "chunk_id": i,
                "text": chunk,
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
