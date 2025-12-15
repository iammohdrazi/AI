# scripts/build_faiss_db.py
import os
import json
import faiss
import numpy as np

from app.utils.logger import setup_logger
from app.utils.loader import Loader
from app.utils.progress import progress_bar

JSON_INDEX = "index/vector_index.json"
FAISS_INDEX_PATH = "index/faiss.index"
META_PATH = "index/faiss_metadata.json"


def build_faiss_db():
    logger = setup_logger("FAISS-Builder")
    loader = Loader(logger)

    if os.path.exists(FAISS_INDEX_PATH):
        logger.info("FAISS index already exists. Skipping creation")
        return

    loader.start("Starting FAISS database creation")

    if not os.path.exists(JSON_INDEX):
        logger.error(f"Vector index not found: {JSON_INDEX}")
        return

    loader.step("Loading vector index JSON")

    with open(JSON_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        logger.warning("Vector index is empty. Nothing to index")
        return

    loader.step("Preparing embeddings matrix")

    embeddings = np.array(
        [
            item["embedding"]
            for item in progress_bar(
                data,
                desc="Loading embeddings",
                unit="vector"
            )
        ],
        dtype="float32"
    )

    dim = embeddings.shape[1]
    logger.info(f"Embedding dimension: {dim}")
    logger.info(f"Total vectors: {embeddings.shape[0]}")

    loader.step("Normalizing embeddings for cosine similarity")
    faiss.normalize_L2(embeddings)

    loader.step("Creating FAISS index")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)

    loader.step("Saving FAISS metadata")

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

    loader.done("FAISS database created successfully")
    logger.info(f"FAISS index path: {FAISS_INDEX_PATH}")
    logger.info(f"Metadata path: {META_PATH}")


if __name__ == "__main__":
    build_faiss_db()
