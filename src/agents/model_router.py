"""
model_router.py
Picks a cheaper or stronger Groq model based on a simple heuristic on query
complexity, to reduce cost on straightforward questions.
"""

CHEAP_MODEL = "groq/openai/gpt-oss-20b"
STRONG_MODEL = "groq/openai/gpt-oss-120b"

# Simple heuristic: short, single-fact questions go to the cheap model.
# Longer or multi-part questions (likely needing more reasoning) go to the strong model.
WORD_COUNT_THRESHOLD = 12
MULTI_PART_INDICATORS = [" and ", " or ", ";", " also ", " as well as "]


def choose_model(query: str) -> str:
    word_count = len(query.split())
    has_multi_part = any(indicator in query.lower() for indicator in MULTI_PART_INDICATORS)

    if word_count > WORD_COUNT_THRESHOLD or has_multi_part:
        return STRONG_MODEL
    return CHEAP_MODEL


if __name__ == "__main__":
    test_queries = [
        "What is the leave policy?",
        "How do I reset my VPN password?",
        "What is the leave policy and how does it interact with the remote work policy for employees who also travel frequently for client meetings?",
    ]
    for q in test_queries:
        print(f"{q!r} -> {choose_model(q)}")