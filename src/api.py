"""
api.py
FastAPI wrapper exposing DocTrust's multi-agent RAG pipeline as an HTTP API.
Run with: uvicorn src.api:app --reload
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent / "agents"))
from crew import run_query

app = FastAPI(
    title="DocTrust API",
    description="Enterprise RAG Copilot with Guardrails & Observability",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    allowed: bool
    guardrail_reason: str | None = None
    confidence: float | None = None
    citations: list[str] = []


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    result = run_query(request.query)

    return QueryResponse(
        query=result["query"],
        answer=result["final_answer"],
        allowed=result["guardrail_allowed"],
        guardrail_reason=result.get("guardrail_reason"),
        confidence=result.get("synthesized", {}).get("confidence"),
        citations=result.get("synthesized", {}).get("citations", []),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)