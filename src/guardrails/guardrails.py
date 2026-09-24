"""
guardrails.py
Applies safety checks to a synthesized answer before it's returned to the user:
  1. PII in the user's query -- refuse to process
  2. PII in the drafted answer -- refuse to share
  3. Validator grounding check -- refuse if not grounded / verdict is "fail"
  4. Confidence threshold -- refuse if the synthesizer wasn't confident enough
"""

from dataclasses import dataclass
from typing import Optional

from pii import detect_pii

CONFIDENCE_THRESHOLD = 0.5


@dataclass
class GuardrailResult:
    allowed: bool
    reason: Optional[str]
    final_answer: str


def apply_guardrails(query: str, synthesized: dict, validation: dict) -> GuardrailResult:
    # 1. PII in the query itself
    query_pii = detect_pii(query)
    if query_pii:
        return GuardrailResult(
            allowed=False,
            reason=f"Query contains personal information ({', '.join(query_pii)}).",
            final_answer=(
                "I can't process questions that include personal information like "
                "emails, phone numbers, or ID numbers. Please rephrase without that data."
            ),
        )

    # 2. PII in the drafted answer (in case retrieved context leaked something)
    answer_text = synthesized.get("answer", "")
    answer_pii = detect_pii(answer_text)
    if answer_pii:
        return GuardrailResult(
            allowed=False,
            reason=f"Answer contains personal information ({', '.join(answer_pii)}).",
            final_answer=(
                "I found relevant information, but it appears to include personal data, "
                "so I can't share it directly. Please contact the relevant department."
            ),
        )

    # 3. Validator grounding check
    if not validation.get("grounded", False) or validation.get("verdict") != "pass":
        return GuardrailResult(
            allowed=False,
            reason=f"Failed validation: {'; '.join(validation.get('issues', [])) or 'ungrounded answer'}",
            final_answer=(
                "I don't have enough verified information in the knowledge base to answer "
                "that confidently. Please check with the relevant department directly."
            ),
        )

    # 4. Confidence threshold
    confidence = synthesized.get("confidence", 0)
    if confidence < CONFIDENCE_THRESHOLD:
        return GuardrailResult(
            allowed=False,
            reason=f"Confidence too low ({confidence}).",
            final_answer=(
                "I'm not confident enough in this answer based on the available documents. "
                "Please check with the relevant department directly."
            ),
        )

    # All checks passed
    return GuardrailResult(allowed=True, reason=None, final_answer=answer_text)