"""
extractor.py
Uses Groq to extract structured entities (policy_name, department, date, category)
from each text chunk, for use as ChromaDB metadata.
"""

import os
import json
import time
from dotenv import load_dotenv
from groq import Groq

from schema import ChunkEntities

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME = "openai/gpt-oss-20b"  # small, fast model -- entity extraction is a simple task

EXTRACTION_PROMPT = """You extract structured metadata from a piece of enterprise document text.

Given the text below, identify:
- policy_name: the specific policy or topic name, if any (e.g. "Leave Policy", "Data Security Policy"). Use null if not applicable.
- department: the most relevant department (e.g. "HR", "IT", "Finance", "Security"). Use null if unclear.
- date: any specific date mentioned, in YYYY-MM-DD format. Use null if none.
- category: a short lowercase category label (e.g. "leave", "expenses", "remote_work", "security", "onboarding"). Use null if unclear.

Respond ONLY with valid JSON matching this exact structure, nothing else:
{{"policy_name": "...", "department": "...", "date": "...", "category": "..."}}

Text:
\"\"\"
{text}
\"\"\"
"""


def extract_entities(text: str, retries: int = 2) -> ChunkEntities:
    """Call Groq to extract structured entities from a single chunk of text."""
    prompt = EXTRACTION_PROMPT.format(text=text[:2000])  # cap input length for speed/cost

    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
                reasoning_effort="low",
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message
            raw = (message.content or "").strip()

            # gpt-oss models sometimes put the actual answer in a 'reasoning' field
            # instead of 'content' -- fall back to it if content came back empty.
            if not raw:
                raw = (getattr(message, "reasoning", None) or "").strip()

            # Strip markdown code fences if the model adds them despite instructions
            raw = raw.replace("```json", "").replace("```", "").strip()

            if not raw:
                raise ValueError("Empty response from model")

            data = json.loads(raw)
            return ChunkEntities(**data)

        except (json.JSONDecodeError, Exception) as e:
            if attempt < retries:
                time.sleep(1)
                continue
            print(f"  Warning: entity extraction failed after {retries + 1} attempts ({e}). Using empty entities.")
            return ChunkEntities()


def extract_entities_for_chunks(chunks: list[dict]) -> list[dict]:
    """
    Runs entity extraction over a list of chunk dicts (from chunker.py),
    attaching extracted entities to each chunk under the 'entities' key.
    """
    enriched = []
    for i, chunk in enumerate(chunks):
        print(f"  Extracting entities for chunk {i + 1}/{len(chunks)}...")
        entities = extract_entities(chunk["text"])
        enriched_chunk = dict(chunk)
        enriched_chunk["entities"] = entities.model_dump()
        enriched.append(enriched_chunk)
    return enriched


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1] / "ingestion"))
    from loaders import load_all_sources
    from chunker import chunk_documents

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data"

    docs = load_all_sources(str(data_path))
    chunks = chunk_documents(docs)

    print(f"Extracting entities for {len(chunks)} chunks...\n")
    enriched = extract_entities_for_chunks(chunks)

    print("\nSample results:")
    for c in enriched[:5]:
        print(f"  {c['chunk_id']} [{c['source_type']}] -> {c['entities']}")