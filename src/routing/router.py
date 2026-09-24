"""
router.py
Lightweight keyword-based router: scans a user query for known department
or category names and builds a ChromaDB metadata filter if a match is found.
Falls back to no filter (pure vector search) if nothing matches.
"""

# Known values -- keep these in sync with what the entity extractor tends to produce.
KNOWN_DEPARTMENTS = ["HR", "IT", "Finance", "Security", "Engineering", "Ethics and Compliance"]

# Map each category label to a list of keywords/synonyms that might appear in a query.
# This avoids brittle exact-substring matches like "expenses" failing on "expense".
CATEGORY_KEYWORDS = {
    "leave": ["leave", "vacation", "time off", "pto"],
    "expenses": ["expense", "reimbursement", "reimburse"],
    "remote_work": ["remote work", "work from home", "wfh", "hybrid"],
    "security": ["security", "vpn", "password", "data breach", "encryption"],
    "onboarding": ["onboarding", "new hire", "orientation"],
    "compliance": ["compliance", "code of conduct", "ethics", "harassment"],
}


def route_query(query: str) -> dict | None:
    """
    Inspects the query text for department or category keywords.
    Returns a ChromaDB 'where' filter dict, or None if no keyword matched
    (meaning: fall back to pure vector search with no metadata filter).
    """
    query_lower = query.lower()

    matched_department = next(
        (dept for dept in KNOWN_DEPARTMENTS if dept.lower() in query_lower),
        None
    )
    matched_category = next(
        (cat for cat, keywords in CATEGORY_KEYWORDS.items()
         if any(kw in query_lower for kw in keywords)),
        None
    )

    conditions = []
    if matched_department:
        conditions.append({"department": matched_department})
    if matched_category:
        conditions.append({"category": matched_category})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$or": conditions}


if __name__ == "__main__":
    test_queries = [
        "What is the HR leave policy?",
        "How do I reset my VPN password?",
        "What are the expense reimbursement rules?",
        "Tell me something random about the company",
    ]
    for q in test_queries:
        print(f"{q!r} -> {route_query(q)}")