"""
run.py
End-to-end ingestion: load sources -> chunk -> extract entities -> embed -> store in ChromaDB.
Run with: python src/ingestion/run.py --path data/
"""

import argparse
from pathlib import Path
import sys
import chromadb
from sentence_transformers import SentenceTransformer

from loaders import load_all_sources
from chunker import chunk_documents

sys.path.append(str(Path(__file__).resolve().parents[1] / "knowledge"))
from extractor import extract_entities_for_chunks

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

    print("Extracting entities for each chunk...")
    chunks = extract_entities_for_chunks(chunks)

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Computing embeddings...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    print("Storing in ChromaDB...")
    project_root = Path(__file__).resolve().parents[2]
    chroma_path = str(project_root / CHROMA_PATH)
    client = chromadb.PersistentClient(path=chroma_path)

    # Fresh collection each run, so re-running ingestion doesn't duplicate old chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(COLLECTION_NAME)

    metadatas = []
    for c in chunks:
        meta = {"source_type": c["source_type"], "source_name": c["source_name"]}
        # Flatten entities into top-level metadata, replacing None with empty string
        # since ChromaDB metadata values must be str, int, float, or bool -- not None.
        for key, value in c["entities"].items():
            meta[key] = value if value is not None else ""
        metadatas.append(meta)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done. Indexed {len(chunks)} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data", help="Root data folder")
    args = parser.parse_args()

    build_index(args.path)