"""
crew.py
Defines the DocTrust multi-agent pipeline:
  Retriever Agent -> Synthesizer Agent -> Validator Agent -> Guardrails
Run with: python src/agents/crew.py
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from tools import DocumentRetrieverTool
from patches import apply_patch

sys.path.append(str(Path(__file__).resolve().parents[1] / "guardrails"))
from guardrails import apply_guardrails
from pii import detect_pii

load_dotenv()
apply_patch()

LLM_MODEL = "groq/openai/gpt-oss-20b"

retriever_tool = DocumentRetrieverTool()

retriever_agent = Agent(
    role="Document Retriever",
    goal="Find the most relevant document chunks in DocTrust's knowledge base to answer the user's question",
    backstory=(
        "You are an expert at searching enterprise documents across PDFs, wikis, "
        "and structured records to find exactly the right context for a question."
    ),
    tools=[retriever_tool],
    llm=LLM_MODEL,
    max_iter=1,
    verbose=True,
)

synthesizer_agent = Agent(
    role="Answer Synthesizer",
    goal="Draft a clear, accurate answer strictly grounded in the retrieved context, with citations",
    backstory=(
        "You are a careful writer who never states anything not explicitly supported "
        "by the provided context, and always cites your sources by name."
    ),
    llm=LLM_MODEL,
    verbose=True,
)

validator_agent = Agent(
    role="Answer Validator",
    goal="Check whether the drafted answer is fully supported by the retrieved context, flagging any unsupported claims",
    backstory=(
        "You are a skeptical fact-checker who cross-references every claim in an "
        "answer against the source context before approving it."
    ),
    llm=LLM_MODEL,
    verbose=True,
)


def _parse_json_output(raw_text: str) -> dict:
    """Best-effort parse of a task's raw text output into a JSON dict."""
    cleaned = raw_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def run_query(query: str) -> dict:
    # Fail fast: check the raw query for PII before running the (expensive) crew at all
    query_pii = detect_pii(query)
    if query_pii:
        return {
            "query": query,
            "synthesized": {},
            "validation": {},
            "guardrail_allowed": False,
            "guardrail_reason": f"Query contains personal information ({', '.join(query_pii)}).",
            "final_answer": (
                "I can't process questions that include personal information like "
                "emails, phone numbers, or ID numbers. Please rephrase without that data."
            ),
        }
    retrieve_task = Task(
    description=(
        f"Call the document_retriever tool EXACTLY ONCE with this exact question as the query: "
        f"'{query}'. Do not reword the query or call the tool more than once, even if the "
        "results seem imperfect. Return whatever the tool gives you."
    ),
    expected_output="The retrieved document chunks with their source names, departments, and categories.",
    agent=retriever_agent,
)

    synthesize_task = Task(
        description=(
            f"Using ONLY the retrieved context from the previous task, answer this question: '{query}'. "
            "Do not use any outside knowledge or make anything up. If the context does not contain "
            "enough information to answer, say so explicitly instead of guessing.\n\n"
            "Respond ONLY with a valid JSON object in this exact format, nothing else:\n"
            '{"answer": "...", "citations": ["source_name_1", "source_name_2"], "confidence": 0.0}'
        ),
        expected_output='A JSON object with "answer", "citations", and "confidence" fields.',
        agent=synthesizer_agent,
        context=[retrieve_task],
    )

    validate_task = Task(
        description=(
            "Review the synthesized answer from the previous task against the originally "
            "retrieved context. Check whether every claim in the answer is actually supported "
            "by the context. Be skeptical -- flag anything that seems inferred or not explicitly stated.\n\n"
            "Respond ONLY with a valid JSON object in this exact format, nothing else:\n"
            '{"grounded": true, "issues": [], "verdict": "pass"}'
        ),
        expected_output='A JSON object with "grounded", "issues", and "verdict" fields.',
        agent=validator_agent,
        context=[retrieve_task, synthesize_task],
    )

    crew = Crew(
        agents=[retriever_agent, synthesizer_agent, validator_agent],
        tasks=[retrieve_task, synthesize_task, validate_task],
        process=Process.sequential,
        verbose=True,
    )

    crew.kickoff()

    synthesized = _parse_json_output(synthesize_task.output.raw)
    validation = _parse_json_output(validate_task.output.raw)

    guard_result = apply_guardrails(query, synthesized, validation)

    return {
        "query": query,
        "synthesized": synthesized,
        "validation": validation,
        "guardrail_allowed": guard_result.allowed,
        "guardrail_reason": guard_result.reason,
        "final_answer": guard_result.final_answer,
    }


if __name__ == "__main__":
    query = "What is the HR leave policy?"
    print(f"\nRunning DocTrust pipeline for: {query!r}\n")
    result = run_query(query)

    print("\n=== FINAL RESULT ===")
    print(f"Allowed: {result['guardrail_allowed']}")
    if result["guardrail_reason"]:
        print(f"Guardrail reason: {result['guardrail_reason']}")
    print(f"\nFinal answer:\n{result['final_answer']}")