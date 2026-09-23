"""
chunker.py
Splits loaded documents into smaller overlapping chunks suitable for embedding.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Simple word-based sliding window chunker.
    chunk_size and overlap are measured in words, not characters,
    to keep chunks semantically coherent.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Takes loader output (list of {"text", "source_type", "source_name"})
    and returns a flat list of chunk dicts, each with a unique chunk_id.
    """
    chunked = []
    chunk_counter = 0

    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for piece in pieces:
            chunked.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": piece,
                "source_type": doc["source_type"],
                "source_name": doc["source_name"],
            })
            chunk_counter += 1

    return chunked


if __name__ == "__main__":
    from pathlib import Path
    from loaders import load_all_sources

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data"

    docs = load_all_sources(str(data_path))
    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks from {len(docs)} documents:")
    for c in chunks[:5]:
        print(f"  {c['chunk_id']} [{c['source_type']}] {c['source_name']}")