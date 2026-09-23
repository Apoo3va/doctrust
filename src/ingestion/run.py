"""
run.py
End-to-end ingestion: load sources -> chunk -> embed -> store in ChromaDB.
Run with: python src/ingestion/run.py --path data/
"""

import argparse
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from loaders import load_all_sources
from chunker import chunk_documents

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "doctrust_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_index(data_root: str = "data"):
    print(f"Loading documents from '{data_root}'...")
    documents = load_all_sources(data_root)
    print(f"  Loaded {len(documents)} raw documents.")

    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Produced {len(chunks)} chunks.")

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Storing in ChromaDB...")
    project_root = Path(__file__).resolve().parents[2]
    chroma_path = str(project_root / CHROMA_PATH)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=[
            {"source_type": c["source_type"], "source_name": c["source_name"]}
            for c in chunks
        ],
    )

    print(f"Done. Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data", help="Root data folder")
    args = parser.parse_args()

    build_index(args.path)