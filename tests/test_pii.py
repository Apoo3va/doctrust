"""
test_pii.py
Unit tests for src/guardrails/pii.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "guardrails"))
from pii import detect_pii


def test_detect_pii_no_pii_in_plain_question():
    result = detect_pii("What is the leave policy?")
    assert result == []


def test_detect_pii_detects_email():
    result = detect_pii("My email is john.doe@example.com")
    assert "email address" in result


def test_detect_pii_detects_pan_number():
    result = detect_pii("My PAN is ABCDE1234F")
    assert "PAN number" in result


def test_detect_pii_detects_phone_number():
    result = detect_pii("Call me at 9876543210")
    assert "phone number" in result


def test_detect_pii_detects_aadhar_like_number():
    result = detect_pii("My Aadhar number is 1234 5678 9012")
    assert "Aadhar-like number" in result


def test_detect_pii_multiple_types_in_one_string():
    result = detect_pii("Email me at test@test.com or call 9876543210")
    assert "email address" in result
    assert "phone number" in result


def test_detect_pii_empty_string_returns_empty():
    assert detect_pii("") == []