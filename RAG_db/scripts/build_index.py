import os
import json

from app.pdf_loader import load_file
from app.embedder import Embedder

from app.utils.logger import setup_logger
from app.utils.progress import progress_bar
from app.utils.loader import Loader

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
    logger = setup_logger("RAG-Indexer")
    loader = Loader(logger)

    loader.start("Starting RAG index build")

    embedder = Embedder()
    index = []

    files = [
        f for f in os.listdir(DATA_DIR)
        if os.path.isfile(os.path.join(DATA_DIR, f))
    ]

    if not files:
        logger.warning("No files found in data directory")
        return

    for filename in files:
        loader.step(f"Processing file: {filename}")

        file_path = os.path.join(DATA_DIR, filename)
        text = load_file(file_path)
        chunks = chunk_text(text)

        logger.info(f"Chunks created: {len(chunks)}")

        for i, chunk in enumerate(
            progress_bar(
                chunks,
                desc=f"Embedding {filename}",
                unit="chunk"
            )
        ):
            embedding = embedder.embed(chunk)

            index.append({
                "file": filename,
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            })

    loader.step("Writing index to disk")

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    loader.done(f"Index successfully saved to {INDEX_PATH}")


if __name__ == "__main__":
    build_index()
