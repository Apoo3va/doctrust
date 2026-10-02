"""
api.py
FastAPI wrapper exposing DocTrust's multi-agent RAG pipeline as an HTTP API.
Protected by a simple API key header for the /query endpoint.
Run with: uvicorn src.api:app --reload
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Security, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent / "agents"))
sys.path.append(str(Path(__file__).resolve().parent / "observability"))
from crew import run_query
from logger import get_logger

load_dotenv()
log = get_logger("api")

API_KEY = os.getenv("DOCTRUST_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

app = FastAPI(
    title="DocTrust API",
    description="Enterprise RAG Copilot with Guardrails & Observability",
    version="1.0.0",
)


def verify_api_key(provided_key: str = Security(api_key_header)):
    if not API_KEY:
        log.warning("DOCTRUST_API_KEY is not set in the environment -- rejecting all requests")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server is not configured with an API key.",
        )
    if provided_key != API_KEY:
        log.warning("Rejected request with invalid or missing API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Include it in the 'X-API-Key' header.",
        )
    return provided_key


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
def query_endpoint(request: QueryRequest, api_key: str = Security(verify_api_key)):
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