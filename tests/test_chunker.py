"""
test_chunker.py
Unit tests for src/ingestion/chunker.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "ingestion"))
from chunker import chunk_text, chunk_documents


def test_chunk_text_short_text_returns_single_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long_text_splits_into_multiple_chunks():
    text = " ".join(["word"] * 1200)  # 1200 words, well over chunk_size
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_chunk_text_respects_overlap():
    words = [f"word{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=500, overlap=50)

    # The last 50 words of chunk 1 should appear at the start of chunk 2
    chunk1_words = chunks[0].split()
    chunk2_words = chunks[1].split()
    assert chunk1_words[-50:] == chunk2_words[:50]


def test_chunk_documents_produces_unique_chunk_ids():
    documents = [
        {"text": "Doc one content.", "source_type": "pdf", "source_name": "doc1.pdf"},
        {"text": "Doc two content.", "source_type": "wiki", "source_name": "doc2.md"},
    ]
    chunks = chunk_documents(documents)

    chunk_ids = [c["chunk_id"] for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"


def test_chunk_documents_preserves_metadata():
    documents = [
        {"text": "Some policy text here.", "source_type": "pdf", "source_name": "policy.pdf"},
    ]
    chunks = chunk_documents(documents)

    assert chunks[0]["source_type"] == "pdf"
    assert chunks[0]["source_name"] == "policy.pdf"


def test_chunk_documents_empty_list_returns_empty():
    assert chunk_documents([]) == []