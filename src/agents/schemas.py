"""
schemas.py
Structured output schemas for the synthesizer and validator agents.
"""

from pydantic import BaseModel, Field
from typing import List


class SynthesizedAnswer(BaseModel):
    answer: str
    citations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ValidationResult(BaseModel):
    grounded: bool
    issues: List[str] = Field(default_factory=list)
    verdict: str  # "pass" or "fail"