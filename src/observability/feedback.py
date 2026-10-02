"""
feedback.py
Stores user feedback (helpful / not helpful) on DocTrust's answers,
tied to the original query, for later review and potential addition
to the evaluation test set.
"""

import csv
import time
import uuid
from pathlib import Path

FEEDBACK_LOG_PATH = Path(__file__).resolve().parents[2] / "feedback_log.csv"


def store_feedback(query: str, answer: str, helpful: bool, comment: str = "") -> str:
    """
    Appends a feedback row to the feedback log.
    Returns a unique feedback_id for reference.
    """
    feedback_id = str(uuid.uuid4())
    file_exists = FEEDBACK_LOG_PATH.exists()

    with open(FEEDBACK_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["feedback_id", "timestamp", "query", "answer", "helpful", "comment"])
        writer.writerow([
            feedback_id,
            time.strftime("%Y-%m-%d %H:%M:%S"),
            query,
            answer,
            helpful,
            comment,
        ])

    return feedback_id


def get_feedback_summary() -> dict:
    """Returns simple aggregate stats over all stored feedback."""
    if not FEEDBACK_LOG_PATH.exists():
        return {"total": 0, "helpful": 0, "not_helpful": 0, "helpful_rate": None}

    total = 0
    helpful_count = 0

    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row["helpful"].strip().lower() == "true":
                helpful_count += 1

    helpful_rate = round((helpful_count / total) * 100, 1) if total > 0 else None

    return {
        "total": total,
        "helpful": helpful_count,
        "not_helpful": total - helpful_count,
        "helpful_rate": helpful_rate,
    }


if __name__ == "__main__":
    fid = store_feedback(
        query="What is the HR leave policy?",
        answer="Employees accrue 1.5 days per month.",
        helpful=True,
        comment="Accurate and clear",
    )
    print(f"Stored feedback with id: {fid}")
    print(get_feedback_summary())