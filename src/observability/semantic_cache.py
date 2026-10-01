"""
semantic_cache.py
Caches query -> answer pairs, keyed by embedding similarity rather than exact
text match, so near-duplicate questions ("What's the leave policy?" vs
"What is the leave policy?") can reuse a prior answer without another LLM call.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

CACHE_PATH = Path(__file__).resolve().parents[2] / "semantic_cache.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.92  # cosine similarity; higher = stricter match required
CACHE_TTL_SECONDS = 24 * 60 * 60  # cache entries expire after 24 hours

_embedder = SentenceTransformer(EMBEDDING_MODEL)


def _load_cache() -> list[dict]:
    if not CACHE_PATH.exists():
        return []
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_cache(entries: list[dict]):
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    a, b = np.array(vec_a), np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_cached_answer(query: str) -> dict | None:
    """
    Returns a cached result dict if a sufficiently similar, non-expired query
    exists in the cache; otherwise returns None.
    """
    entries = _load_cache()
    if not entries:
        return None

    query_embedding = _embedder.encode([query])[0].tolist()
    now = time.time()

    best_match = None
    best_score = 0.0

    for entry in entries:
        if now - entry["timestamp"] > CACHE_TTL_SECONDS:
            continue  # expired
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score > best_score:
            best_score = score
            best_match = entry

    if best_match and best_score >= SIMILARITY_THRESHOLD:
        print(f"[semantic_cache] Cache HIT (similarity={best_score:.3f}) for: {query!r}")
        return best_match["result"]

    return None


def store_cached_answer(query: str, result: dict):
    """Stores a new query -> result pair in the cache."""
    entries = _load_cache()
    query_embedding = _embedder.encode([query])[0].tolist()

    entries.append({
        "query": query,
        "embedding": query_embedding,
        "result": result,
        "timestamp": time.time(),
    })

    # Keep the cache from growing unbounded -- drop oldest entries beyond 200
    if len(entries) > 200:
        entries = sorted(entries, key=lambda e: e["timestamp"])[-200:]

    _save_cache(entries)


if __name__ == "__main__":
    # Quick manual test
    store_cached_answer("What is the leave policy?", {"final_answer": "Test answer", "guardrail_allowed": True})
    print(get_cached_answer("What is the leave policy?"))       # exact match -> hit
    print(get_cached_answer("What's the leave policy?"))         # near match -> likely hit
    print(get_cached_answer("How do I reset my VPN password?"))  # unrelated -> miss