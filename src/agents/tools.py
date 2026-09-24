"""
tools.py
Wraps our hybrid retriever (vector + entity metadata) as a CrewAI tool
so the Retriever Agent can call it.
"""

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from retriever import retrieve


class RetrieverToolInput(BaseModel):
    query: str = Field(..., description="The user's question to search for relevant document chunks")


class DocumentRetrieverTool(BaseTool):
    name: str = "document_retriever"
    description: str = (
        "Searches DocTrust's knowledge base (PDFs, wikis, CSV records) for chunks "
        "relevant to a query. Returns retrieved text along with source name, "
        "department, and category metadata for each chunk."
    )
    args_schema: type[BaseModel] = RetrieverToolInput

    def _run(self, query: str) -> str:
        results = retrieve(query, top_k=3)
        if not results:
            return "No relevant documents found in the knowledge base."

        formatted = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            formatted.append(
                f"[Chunk {i}] Source: {meta.get('source_name')} ({meta.get('source_type')})\n"
                f"Department: {meta.get('department') or 'N/A'} | Category: {meta.get('category') or 'N/A'}\n"
                f"Content: {r['text']}"
            )
        return "\n\n---\n\n".join(formatted)