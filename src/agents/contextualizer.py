"""
contextualizer.py
Rewrites a follow-up question into a standalone question using prior
conversation history, so the existing single-turn retrieval pipeline
can handle multi-turn conversations without any other changes.

Example:
  History: [{"query": "What is the leave policy?", "answer": "Employees accrue..."}]
  Follow-up: "And what about remote work?"
  Rewritten: "What is the remote work policy?"
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"  # cheap model -- this is a simple rewriting task

CONTEXTUALIZE_PROMPT = """Given the conversation history and a follow-up question, rewrite the \
follow-up question to be a standalone question that includes all necessary context, so it can \
be understood without the history.

If the follow-up question is already standalone (does not depend on the history), return it unchanged.

Respond with ONLY the rewritten question, nothing else -- no explanation, no quotes.

Conversation history:
{history}

Follow-up question: {query}

Standalone question:"""


def format_history(history: list[dict]) -> str:
    if not history:
        return "(no prior conversation)"
    lines = []
    for turn in history[-3:]:  # only use the last 3 turns to keep the prompt small
        lines.append(f"Q: {turn['query']}")
        lines.append(f"A: {turn['answer']}")
    return "\n".join(lines)


def contextualize_query(query: str, history: list[dict]) -> str:
    """
    If history is empty, returns the query unchanged (no LLM call needed).
    Otherwise, asks the LLM to rewrite the query as a standalone question.
    """
    if not history:
        return query

    prompt = CONTEXTUALIZE_PROMPT.format(history=format_history(history), query=query)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150,
        )
        rewritten = response.choices[0].message.content.strip().strip('"')
        return rewritten if rewritten else query
    except Exception:
        # If contextualization fails for any reason, fall back to the original query
        # rather than breaking the whole pipeline.
        return query


if __name__ == "__main__":
    history = [
        {"query": "What is the leave policy?",
         "answer": "Employees accrue 1.5 days of paid leave per month, capped at 18 days per year."}
    ]
    follow_ups = [
        "And what about remote work?",
        "What is the probation period?",  # standalone, should stay unchanged
    ]
    for q in follow_ups:
        rewritten = contextualize_query(q, history)
        print(f"{q!r} -> {rewritten!r}")