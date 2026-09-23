"""
loaders.py
Loads raw content from PDF, Markdown, and CSV sources.
Each loader returns a list of dicts: {"text": str, "source_type": str, "source_name": str}
"""

from pathlib import Path
from pypdf import PdfReader
import pandas as pd


def load_pdfs(folder: str) -> list[dict]:
    """Extract text from every PDF in the given folder, one dict per page."""
    documents = []
    for pdf_path in Path(folder).glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append({
                    "text": text,
                    "source_type": "pdf",
                    "source_name": f"{pdf_path.name} (page {page_num + 1})"
                })
    return documents


def load_markdown(folder: str) -> list[dict]:
    """Load each Markdown file as one document (chunking happens later)."""
    documents = []
    for md_path in Path(folder).glob("*.md"):
        text = md_path.read_text(encoding="utf-8")
        if text.strip():
            documents.append({
                "text": text,
                "source_type": "wiki",
                "source_name": md_path.name
            })
    return documents


def load_csv_records(folder: str) -> list[dict]:
    """
    Load CSV records, turning each row into a single text blob.
    Assumes columns like question/answer/department, but works generically
    by concatenating all column values.
    """
    documents = []
    for csv_path in Path(folder).glob("*.csv"):
        df = pd.read_csv(csv_path)
        for idx, row in df.iterrows():
            row_text = "\n".join(f"{col}: {row[col]}" for col in df.columns)
            documents.append({
                "text": row_text,
                "source_type": "csv",
                "source_name": f"{csv_path.name} (row {idx + 1})"
            })
    return documents


def load_all_sources(data_root: str = "data") -> list[dict]:
    """Load PDF, Markdown, and CSV sources from their respective subfolders."""
    root = Path(data_root)
    documents = []
    documents.extend(load_pdfs(str(root / "pdf")))
    documents.extend(load_markdown(str(root / "wiki")))
    documents.extend(load_csv_records(str(root / "records")))
    return documents


if __name__ == "__main__":
    docs = load_all_sources()
    print(f"Loaded {len(docs)} raw documents:")
    for d in docs[:5]:
        print(f"  [{d['source_type']}] {d['source_name']} -- {d['text'][:60]!r}...")