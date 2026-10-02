"""
api.py
FastAPI wrapper exposing DocTrust's multi-agent RAG pipeline as an HTTP API.
Protected by a simple API key header. Includes a /feedback loop for tracking
answer helpfulness over time.
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
from feedback import store_feedback, get_feedback_summary

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


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    helpful: bool
    comment: str = ""


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str


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


@app.post("/feedback", response_model=FeedbackResponse)
def feedback_endpoint(request: FeedbackRequest, api_key: str = Security(verify_api_key)):
    feedback_id = store_feedback(
        query=request.query,
        answer=request.answer,
        helpful=request.helpful,
        comment=request.comment,
    )
    log.info(f"Feedback received: helpful={request.helpful} id={feedback_id}")
    return FeedbackResponse(feedback_id=feedback_id, message="Feedback recorded. Thank you.")


@app.get("/feedback/summary")
def feedback_summary_endpoint(api_key: str = Security(verify_api_key)):
    return get_feedback_summary()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)