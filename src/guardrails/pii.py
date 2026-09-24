"""
pii.py
Lightweight regex-based PII detection. Not exhaustive, but covers the most
common patterns worth catching before a response is returned to the user.
"""

import re

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_PATTERN = re.compile(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")            # Indian PAN card format
AADHAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")       # Indian Aadhar-like 12-digit number
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


def detect_pii(text: str) -> list[str]:
    """Returns a list of PII types found in the text, empty if none found."""
    findings = []
    if EMAIL_PATTERN.search(text):
        findings.append("email address")
    if PAN_PATTERN.search(text):
        findings.append("PAN number")
    if AADHAR_PATTERN.search(text):
        findings.append("Aadhar-like number")
    if CREDIT_CARD_PATTERN.search(text):
        findings.append("credit card-like number")
    elif PHONE_PATTERN.search(text):
        findings.append("phone number")
    return findings


if __name__ == "__main__":
    test_cases = [
        "What is the leave policy?",
        "My email is john.doe@example.com, can you help?",
        "My PAN is ABCDE1234F",
        "Call me at 9876543210",
    ]
    for t in test_cases:
        print(f"{t!r} -> {detect_pii(t)}")