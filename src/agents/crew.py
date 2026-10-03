"""
crew.py
Defines the DocTrust multi-agent pipeline:
  Semantic Cache -> Retriever Agent -> Synthesizer Agent -> Validator Agent -> Guardrails
Instrumented with OpenTelemetry tracing, cost-aware model routing, semantic caching,
structured logging, and multi-turn conversation memory via query contextualization.
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
    """Builds fresh agent instances using the given model (cheap or strong)."""

    retriever_agent = Agent(
        role="Document Retriever",
        goal=(
            "Find the most relevant document chunks in DocTrust's knowledge base "
            "to answer the user's question"
        ),
        backstory=(
            "You are an expert at searching enterprise documents across PDFs, wikis, "
            "and structured records to find exactly the right context for a question."
        ),
        tools=[retriever_tool],
        llm=model,
        max_iter=1,
        verbose=True,
    )

    synthesizer_agent = Agent(
        role="Answer Synthesizer",
        goal=(
            "Draft a clear, accurate answer strictly grounded in the retrieved "
            "context, with citations"
        ),
        backstory=(
            "You are a careful writer who never states anything not explicitly "
            "supported by the provided context, and always cites your sources by name."
        ),
        llm=model,
        verbose=True,
    )

    validator_agent = Agent(
        role="Answer Validator",
        goal=(
            "Check whether the drafted answer is fully supported by the retrieved "
            "context, flagging any unsupported claims"
        ),
        backstory=(
            "You are a skeptical fact-checker who cross-references every claim in an "
            "answer against the source context before approving it."
        ),
        llm=model,
        verbose=True,
    )

    return retriever_agent, synthesizer_agent, validator_agent


def _parse_json_output(raw_text: str) -> dict:
    """Best-effort parse of a task's raw text output into a JSON dict."""

    cleaned = (
        raw_text
        .strip()
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning(
            f"Failed to parse JSON output from agent: {cleaned[:200]!r}"
        )
        return {}


def run_query(query: str, history: list[dict] | None = None) -> dict:
    start_time = time.time()
    history = history or []

    log.info(
        f"Received query: {query!r} "
        f"(history_turns={len(history)})"
    )

    original_query = query

    # Contextualize follow-up questions when conversation history exists
    if history:
        query = contextualize_query(query, history)

        if query != original_query:
            log.info(
                f"Contextualized follow-up: "
                f"{original_query!r} -> {query!r}"
            )

    # Fail fast: check the raw query for PII before running
    # the expensive crew at all
    query_pii = detect_pii(query)

    if query_pii:
        log.warning(
            f"Query blocked for PII "
            f"({', '.join(query_pii)}): {query!r}"
        )

        return {
            "query": original_query,
            "synthesized": {},
            "validation": {},
            "guardrail_allowed": False,
            "guardrail_reason": (
                f"Query contains personal information "
                f"({', '.join(query_pii)})."
            ),
            "final_answer": (
                "I can't process questions that include personal information "
                "like emails, phone numbers, or ID numbers. "
                "Please rephrase without that data."
            ),
        }

    # Check semantic cache before running the expensive pipeline.
    # Skip caching for multi-turn queries because a cached single-turn
    # answer may not fit the conversational context.
    if not history:
        cached = get_cached_answer(query)

        if cached is not None:
            log.info(
                f"Served from semantic cache: {query!r}"
            )
            return cached

    # Cost-aware model routing: pick cheap or strong model
    # based on query complexity
    model = choose_model(query)

    log.info(
        f"Model router selected: {model}"
    )

    retriever_agent, synthesizer_agent, validator_agent = build_agents(model)

    # ---------------------------------------------------------
    # RETRIEVER TASK
    # ---------------------------------------------------------

    with traced_span(
        "retrieve_task",
        query=query,
        model=model,
    ):
        retrieve_task = Task(
            description=(
                f"Call the document_retriever tool EXACTLY ONCE "
                f"with this exact question as the query: "
                f"'{query}'. "
                "Do not reword the query or call the tool more than once, "
                "even if the results seem imperfect. "
                "Return whatever the tool gives you."
            ),
            expected_output=(
                "The retrieved document chunks with their source names, "
                "departments, and categories."
            ),
            agent=retriever_agent,
        )

    # ---------------------------------------------------------
    # SYNTHESIZER TASK
    # ---------------------------------------------------------

    synthesize_task = Task(
        description=(
            f"Using ONLY the retrieved context from the previous task, "
            f"answer this question: '{query}'. "
            "Do not use any outside knowledge or make anything up. "
            "If the context does not contain enough information to answer, "
            "say so explicitly instead of guessing.\n\n"
            "Respond ONLY with a valid JSON object in this exact format, "
            "nothing else:\n"
            '{"answer": "...", '
            '"citations": ["source_name_1", "source_name_2"], '
            '"confidence": 0.0}'
        ),
        expected_output=(
            'A JSON object with "answer", "citations", '
            'and "confidence" fields.'
        ),
        agent=synthesizer_agent,
        context=[retrieve_task],
    )

    # ---------------------------------------------------------
    # VALIDATOR TASK
    # ---------------------------------------------------------

    validate_task = Task(
        description=(
            "Review the synthesized answer from the previous task "
            "against the originally retrieved context. "
            "Check whether every claim in the answer is actually "
            "supported by the context. "
            "Be skeptical -- flag anything that seems inferred "
            "or not explicitly stated.\n\n"
            "Respond ONLY with a valid JSON object in this exact format, "
            "nothing else:\n"
            '{"grounded": true, "issues": [], "verdict": "pass"}'
        ),
        expected_output=(
            'A JSON object with "grounded", "issues", '
            'and "verdict" fields.'
        ),
        agent=validator_agent,
        context=[retrieve_task, synthesize_task],
    )

    # ---------------------------------------------------------
    # CREW
    # ---------------------------------------------------------

    crew = Crew(
        agents=[
            retriever_agent,
            synthesizer_agent,
            validator_agent,
        ],
        tasks=[
            retrieve_task,
            synthesize_task,
            validate_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    # FIXED: correct indentation
    log.info("Starting crew execution")

    try:
        with traced_span(
            "crew_pipeline",
            query=query,
            model=model,
        ):
            crew.kickoff()

        log.info("Crew execution completed")

    except Exception as e:
        log.error(
            f"Crew execution failed: {e}"
        )

        error_message = (
            "The system is currently handling a high volume of requests "
            "and hit a rate limit. Please wait a moment and try again."
        ) if (
            "rate_limit" in str(e).lower()
            or "RateLimitError" in str(e)
        ) else (
            "Something went wrong while processing your question. "
            "Please try again."
        )

        return {
            "query": original_query,
            "synthesized": {},
            "validation": {},
            "guardrail_allowed": False,
            "guardrail_reason": f"Pipeline error: {e}",
            "final_answer": error_message,
        }

    # ---------------------------------------------------------
    # PARSE AGENT OUTPUT
    # ---------------------------------------------------------

    synthesized = _parse_json_output(
        synthesize_task.output.raw
    )

    validation = _parse_json_output(
        validate_task.output.raw
    )

    # ---------------------------------------------------------
    # GUARDRAILS
    # ---------------------------------------------------------

    guard_result = apply_guardrails(
        query,
        synthesized,
        validation,
    )

    if not guard_result.allowed:
        log.warning(
            f"Guardrail blocked answer: "
            f"{guard_result.reason}"
        )
    else:
        log.info(
            f"Guardrail passed. "
            f"Confidence={synthesized.get('confidence')}"
        )

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    total_latency = time.time() - start_time

    # Pull token usage from CrewAI's usage metrics if available
    prompt_tokens = 0
    completion_tokens = 0

    try:
        usage = crew.usage_metrics

        prompt_tokens = getattr(
            usage,
            "prompt_tokens",
            0,
        )

        completion_tokens = getattr(
            usage,
            "completion_tokens",
            0,
        )

    except Exception as e:
        log.debug(
            f"Could not read usage_metrics: {e}"
        )

    log_query_metrics(
        query=query,
        model=model,
        latency=total_latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        allowed=guard_result.allowed,
    )

    log.info(
        f"Query completed in {total_latency:.2f}s"
    )

    # ---------------------------------------------------------
    # FINAL RESULT
    # ---------------------------------------------------------

    result = {
        "query": original_query,
        "synthesized": synthesized,
        "validation": validation,
        "guardrail_allowed": guard_result.allowed,
        "guardrail_reason": guard_result.reason,
        "final_answer": guard_result.final_answer,
    }

    # Only cache single-turn answers that passed guardrails
    if guard_result.allowed and not history:
        store_cached_answer(
            query,
            result,
        )

    return result


# -------------------------------------------------------------
# DIRECT EXECUTION
# -------------------------------------------------------------

if __name__ == "__main__":
    query = "What is the HR leave policy?"

    log.info(
        f"Running DocTrust pipeline for: {query!r}"
    )

    result = run_query(query)

    print("\n=== FINAL RESULT ===")
    print(
        f"Allowed: {result['guardrail_allowed']}"
    )

    if result["guardrail_reason"]:
        print(
            f"Guardrail reason: "
            f"{result['guardrail_reason']}"
        )

    print(
        f"\nFinal answer:\n"
        f"{result['final_answer']}"
    )