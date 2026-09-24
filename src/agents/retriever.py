"""
retriever.py
Hybrid retrieval: combines vector similarity search with optional
metadata filtering (department/category) derived from the query router.
Uses the SAME embedding model as ingestion for consistent similarity scoring.
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1] / "routing"))
from router import route_query

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "doctrust_chunks"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_project_root = Path(__file__).resolve().parents[2]
_chroma_path = str(_project_root / CHROMA_PATH)

_client = chromadb.PersistentClient(path=_chroma_path)
_collection = _client.get_collection(COLLECTION_NAME)
_embedder = SentenceTransformer(EMBEDDING_MODEL)


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieves the top_k most relevant chunks for a query.
    Applies a metadata filter (department/category) if the router detects one;
    otherwise falls back to pure vector similarity search.
    """
    query_embedding = _embedder.encode([query]).tolist()
    where_filter = route_query(query)

    query_kwargs = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = _collection.query(**query_kwargs)

    # If a filtered search returns nothing (filter too narrow), fall back to unfiltered search
    if where_filter and not results["documents"][0]:
        query_kwargs.pop("where")
        results = _collection.query(**query_kwargs)

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "metadata": meta,
            "similarity_score": 1 - dist,  # convert distance to a similarity-style score
        })

    return retrieved


if __name__ == "__main__":
    test_queries = [
        "What is the HR leave policy?",
        "How do I reset my VPN password?",
        "What are the expense reimbursement rules?",
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r['metadata']['source_type']}] score={r['similarity_score']:.3f} "
                  f"dept={r['metadata'].get('department')} cat={r['metadata'].get('category')}")
            print(f"    {r['text'][:80]!r}...")