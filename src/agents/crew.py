"""
crew.py
Defines the DocTrust multi-agent pipeline:
  Semantic Cache -> Retriever Agent -> Synthesizer Agent -> Validator Agent -> Guardrails
Instrumented with OpenTelemetry tracing, cost-aware model routing, semantic caching,
and structured logging. Supports multi-turn conversation memory via query contextualization.
Run with: python src/agents/crew.py
"""
import os
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from tools import DocumentRetrieverTool
from patches import apply_patch
from model_router import choose_model
from contextualizer import contextualize_query

sys.path.append(str(Path(__file__).resolve().parents[1] / "guardrails"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "observability"))
from guardrails import apply_guardrails
from pii import detect_pii
from tracing import traced_span, log_query_metrics
from semantic_cache import get_cached_answer, store_cached_answer
from logger import get_logger

load_dotenv()
apply_patch()

log = get_logger("crew")

retriever_tool = DocumentRetrieverTool()


def build_agents(model: str):
    retriever_agent = Agent(
        role="Document Retriever",
        goal="Retrieve the most relevant document chunks for a given question.",
        backstory=(
            "You are a precise retrieval specialist. You call the document_retriever "
            "tool exactly once with the user's question and return whatever it gives you, "
            "without rewording the query or second-guessing the results."
        ),
        tools=[retriever_tool],
        llm=model,
        max_iter=2,
        verbose=True,
    )

    synthesizer_agent = Agent(
        role="Answer Synthesizer",
        goal="Write a clear, accurate answer using only the retrieved context.",
        backstory=(
            "You are a careful technical writer. You never add information that isn't "
            "present in the retrieved context, and you always cite your sources. You "
            "estimate your own confidence honestly based on how directly the context "
            "supports your answer."
        ),
        llm=model,
        verbose=True,
    )

    validator_agent = Agent(
        role="Answer Validator",
        goal="Check whether the synthesized answer is fully grounded in the retrieved context.",
        backstory=(
            "You are a skeptical fact-checker. You flag any claim in the answer that "
            "isn't directly supported by the retrieved context, and you are not afraid "
            "to fail an answer that sounds plausible but isn't backed by evidence."
        ),
        llm=model,
        verbose=True,
    )

    return retriever_agent, synthesizer_agent, validator_agent


def _parse_json_output(raw_text: str) -> dict:
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(f"Failed to parse JSON output from agent: {cleaned[:200]!r}")
        return {}


def run_query(query: str, history: list[dict] | None = None) -> dict:
    start_time = time.time()
    history = history or []
    log.info(f"Received query: {query!r} (history_turns={len(history)})")

    original_query = query
    if history:
        query = contextualize_query(query, history)
        if query != original_query:
            log.info(f"Contextualized follow-up: {original_query!r} -> {query!r}")

    # Fail fast: PII check before expensive crew execution
    query_pii = detect_pii(query)
    if query_pii:
        log.warning(f"Blocked query containing PII types: {query_pii}")
        return {
            "query": original_query,
            "synthesized": {},
            "validation": {},
            "guardrail_allowed": False,
            "guardrail_reason": f"Query contains PII: {query_pii}",
            "final_answer": (
                "I can't process questions that include personal information like "
                "emails, phone numbers, or ID numbers. Please rephrase without that data."
            ),
        }

    # Semantic cache (skipped when history present)
    if not history:
        cached = get_cached_answer(query)
        if cached is not None:
            log.info(f"Served from semantic cache: {query!r}")
            return cached

    model = choose_model(query)
    log.info(f"Model router selected: {model}")
    retriever_agent, synthesizer_agent, validator_agent = build_agents(model)

    with traced_span("retrieve_task", query=query, model=model):
        retrieve_task = Task(
            description=(
                f"Call the document_retriever tool EXACTLY ONCE with this exact question "
                f"as the query: '{query}'. Do not reword the query or call the tool more "
                f"than once, even if the results seem imperfect. Return whatever the tool "
                f"gives you."
            ),
            expected_output="The raw retrieved document chunks returned by the tool.",
            agent=retriever_agent,
        )

    synthesize_task = Task(
        description=(
            f"Using ONLY the retrieved context from the previous task, answer this "
            f"question: '{query}'. Do not use any outside knowledge. If the context does "
            f"not contain enough information to answer, say so honestly instead of "
            f"guessing. Cite the source of each fact you use (file name or section). "
            f"Respond ONLY with a valid JSON object in this exact format, no extra text "
            f"before or after it: "
            f'{{"answer": "...", "citations": ["..."], "confidence": 0.0}} '
            f"where confidence is a number between 0.0 and 1.0 reflecting how directly "
            f"the retrieved context supports your answer."
        ),
        expected_output="A JSON object with answer, citations, and confidence fields.",
        context=[retrieve_task],
        agent=synthesizer_agent,
    )

    validate_task = Task(
        description=(
            "Review the synthesized answer from the previous task against the retrieved "
            "context. Check whether every claim in the answer is directly supported by "
            "the context. Respond ONLY with a valid JSON object in this exact format, no "
            "extra text before or after it: "
            '{"grounded": true, "issues": [], "verdict": "pass"} '
            "where verdict is either 'pass' or 'fail', and issues is a list of strings "
            "describing any unsupported claims (empty list if none)."
        ),
        expected_output="A JSON object with grounded, issues, and verdict fields.",
        context=[retrieve_task, synthesize_task],
        agent=validator_agent,
    )

    crew = Crew(
        agents=[retriever_agent, synthesizer_agent, validator_agent],
        tasks=[retrieve_task, synthesize_task, validate_task],
        process=Process.sequential,
        verbose=True,
    )

    log.info("Starting crew execution")
    try:
        with traced_span("crew_pipeline", query=query, model=model):
            crew.kickoff()
        log.info("Crew execution completed")
    except Exception as e:
        log.error(f"Crew execution failed: {e}")
        error_message = (
            "The system is currently handling a high volume of requests and hit a rate "
            "limit. Please wait a moment and try again."
        ) if "rate_limit" in str(e).lower() or "RateLimitError" in str(e) else (
            "Something went wrong while processing your question. Please try again."
        )
        return {
            "query": original_query,
            "synthesized": {},
            "validation": {},
            "guardrail_allowed": False,
            "guardrail_reason": f"Pipeline error: {e}",
            "final_answer": error_message,
        }

    synthesized = _parse_json_output(synthesize_task.output.raw)
    validation = _parse_json_output(validate_task.output.raw)
    log.info(f"Synthesized output: {synthesized}")
    log.info(f"Validation output: {validation}")
    guard_result = apply_guardrails(query, synthesized, validation)

    total_latency = time.time() - start_time

    usage = getattr(crew, "usage_metrics", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

    log_query_metrics(
        query=query,
        model=model,
        latency=total_latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        allowed=guard_result.allowed,
    )
    log.info(f"Query completed in {total_latency:.2f}s")

    result = {
        "query": original_query,
        "synthesized": synthesized,
        "validation": validation,
        "guardrail_allowed": guard_result.allowed,
        "guardrail_reason": guard_result.reason,
        "final_answer": guard_result.final_answer,
    }
    if guard_result.allowed and not history:
        store_cached_answer(query, result)
    return result


if __name__ == "__main__":
    query = "What is the HR leave policy?"
    log.info(f"Running DocTrust pipeline for: {query!r}")
    result = run_query(query)
    print("\n=== FINAL RESULT ===")
    print(f"Allowed: {result['guardrail_allowed']}")
    if result["guardrail_reason"]:
        print(f"Guardrail reason: {result['guardrail_reason']}")
    print(f"\nFinal answer:\n{result['final_answer']}")