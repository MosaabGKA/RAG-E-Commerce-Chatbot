"""Build the local FAISS vector store over the Bitext customer-support KB.

Chunk design: each KB entry is stored as chunk = "instruction" (the customer
question) for retrieval; the paired gold "response" is kept as metadata so it can
be injected into the generation prompt as grounding context.
"""
import pickle
import time

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import BITEXT_DATASET, EMBEDDING_MODEL, RAG_DIR

RETRIEVAL_COL = "instruction"   # embed the customer question for similarity search


def main():
    t0 = time.time()
    print("Loading Bitext KB ...")
    ds = load_dataset(BITEXT_DATASET)
    df = ds["train"].to_pandas()[["instruction", "response", "intent", "category"]].dropna()

    chunks = df[RETRIEVAL_COL].tolist()
    metadata = df[["instruction", "response", "intent", "category"]].to_dict("records")
    print(f"KB entries = {len(chunks)}")

    print(f"Loading embedding model {EMBEDDING_MODEL} ...")
    embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    print("Embedding chunks ...")
    batch_size = 256
    vectors = []
    for i in range(0, len(chunks), batch_size):
        vectors.append(embedder.encode(chunks[i:i + batch_size], show_progress_bar=True))
    matrix = np.vstack(vectors).astype("float32")

    # normalize -> inner product == cosine similarity
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

    import faiss
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    print(f"FAISS index vectors = {index.ntotal}, dim = {index.d}")

    faiss.write_index(index, str(RAG_DIR / "faiss.index"))
    with open(RAG_DIR / "metadata.pkl", "wb") as fh:
        pickle.dump({"metadata": metadata, "order": df.index.tolist()}, fh)
    with open(RAG_DIR / "embedder_path.txt", "w") as fh:
        fh.write(EMBEDDING_MODEL)
    print(f"Saved index + metadata to {RAG_DIR} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()