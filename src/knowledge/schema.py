"""
schema.py
Defines the structured entity schema extracted from each chunk.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChunkEntities(BaseModel):
    policy_name: Optional[str] = Field(
        default=None,
        description="Name of the specific policy or topic this chunk is about, e.g. 'Leave Policy', 'Data Security Policy'"
    )
    department: Optional[str] = Field(
        default=None,
        description="The department most relevant to this content, e.g. 'HR', 'IT', 'Finance', 'Security'"
    )
    date: Optional[str] = Field(
        default=None,
        description="Any specific date mentioned in the chunk, in YYYY-MM-DD format if present"
    )
    category: Optional[str] = Field(
        default=None,
        description="A short category label for this content, e.g. 'leave', 'expenses', 'remote_work', 'security', 'onboarding'"
    )