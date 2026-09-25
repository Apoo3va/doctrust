"""
test_router.py
Unit tests for src/routing/router.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "routing"))
from router import route_query


def test_route_query_no_match_returns_none():
    result = route_query("Tell me something random about the company")
    assert result is None


def test_route_query_matches_department():
    result = route_query("What is the HR policy on something?")
    assert result is not None
    assert "department" in str(result)


def test_route_query_matches_category_security():
    result = route_query("How do I reset my VPN password?")
    assert result == {"category": "security"}


def test_route_query_matches_category_expenses():
    result = route_query("What are the expense reimbursement rules?")
    assert result == {"category": "expenses"}


def test_route_query_matches_both_department_and_category():
    result = route_query("What is the HR leave policy?")
    assert result is not None
    assert "$or" in result


def test_route_query_case_insensitive():
    result_lower = route_query("what is the hr leave policy?")
    result_upper = route_query("WHAT IS THE HR LEAVE POLICY?")
    assert result_lower == result_upper