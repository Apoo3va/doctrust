"""
test_guardrails.py
Unit tests for src/guardrails/guardrails.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "guardrails"))
from guardrails import apply_guardrails


def test_apply_guardrails_blocks_pii_in_query():
    result = apply_guardrails(
        query="My email is test@test.com, what is the leave policy?",
        synthesized={"answer": "Some answer", "confidence": 1.0},
        validation={"grounded": True, "verdict": "pass"},
    )
    assert result.allowed is False
    assert "personal information" in result.reason.lower()


def test_apply_guardrails_blocks_pii_in_answer():
    result = apply_guardrails(
        query="What is the leave policy?",
        synthesized={"answer": "Contact john@example.com for details", "confidence": 1.0},
        validation={"grounded": True, "verdict": "pass"},
    )
    assert result.allowed is False


def test_apply_guardrails_blocks_ungrounded_answer():
    result = apply_guardrails(
        query="What is the leave policy?",
        synthesized={"answer": "Some answer", "confidence": 1.0},
        validation={"grounded": False, "verdict": "fail", "issues": ["not supported"]},
    )
    assert result.allowed is False
    assert "validation" in result.reason.lower()


def test_apply_guardrails_blocks_low_confidence():
    result = apply_guardrails(
        query="What is the leave policy?",
        synthesized={"answer": "Some answer", "confidence": 0.2},
        validation={"grounded": True, "verdict": "pass"},
    )
    assert result.allowed is False
    assert "confidence" in result.reason.lower()


def test_apply_guardrails_allows_valid_answer():
    result = apply_guardrails(
        query="What is the leave policy?",
        synthesized={"answer": "Employees accrue 1.5 days per month.", "confidence": 0.95},
        validation={"grounded": True, "verdict": "pass"},
    )
    assert result.allowed is True
    assert result.final_answer == "Employees accrue 1.5 days per month."
    assert result.reason is None