# scripts/build_index.py
import json
import os
import glob
from app.pdf_loader import load_document as load_pdf
from app.embedder import Embedder

embedder = Embedder()

def load_data(folder):
    documents = []

    for file in glob.glob(folder + "/*.pdf"):
        text = load_pdf(file)
        emb = embedder.embed(text)
        documents.append({
            "text": text,
            "embedding": emb,
            "source": file
        })

    for file in glob.glob(folder + "/*.txt"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        emb = embedder.embed(text)
        documents.append({
            "text": text,
            "embedding": emb,
            "source": file
        })

    return documents


def build_index():
    docs = load_data("data")

    index = {
        "documents": docs
    }

    with open("index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=4)

    print("Index built successfully!")

if __name__ == "__main__":
    build_index()
