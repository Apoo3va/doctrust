"""
crew.py
Defines the DocTrust multi-agent pipeline:
  Retriever Agent -> Synthesizer Agent -> Validator Agent
Run with: python src/agents/crew.py
"""

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

from tools import DocumentRetrieverTool
from patches import apply_patch

load_dotenv()
apply_patch()

# LiteLLM (used internally by CrewAI) reads GROQ_API_KEY from the environment automatically.
# Using a stronger model here since synthesis/validation benefit from better reasoning
# than the small model we used for entity extraction.
LLM_MODEL = "groq/openai/gpt-oss-120b"

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


def run_query(query: str):
    retrieve_task = Task(
        description=(
            f"Search the knowledge base for content relevant to this question: '{query}'. "
            "Use the document_retriever tool. Return the retrieved chunks with their sources."
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

    return crew.kickoff()


if __name__ == "__main__":
    query = "What is the HR leave policy?"
    print(f"\nRunning DocTrust pipeline for: {query!r}\n")
    result = run_query(query)
    print("\n=== FINAL OUTPUT ===")
    print(result)