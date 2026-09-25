"""
run_eval.py
Runs the DocTrust pipeline against a fixed test set and scores the results
with RAGAS: faithfulness, answer relevancy, and context precision.
Run with: python src/eval/run_eval.py
"""

import os
import sys
import json
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1] / "agents"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "guardrails"))

from crew import run_query
from retriever import retrieve
from testset import TEST_SET

load_dotenv()

# RAGAS imports
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, LLMContextPrecisionWithReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# RAGAS runs metric calls concurrently by default, which blows through Groq's
# free-tier rate limit. Force it to run one call at a time with generous timeouts.
RAGAS_RUN_CONFIG = RunConfig(timeout=120, max_workers=1, max_retries=5, max_wait=60)

# Groq free tier has token limits (per-minute and per-day). The full 3-agent
# pipeline is token-heavy, so we pace requests and retry with backoff.
SECONDS_BETWEEN_QUESTIONS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 25


def run_query_with_retry(question: str) -> dict:
    for attempt in range(MAX_RETRIES + 1):
        try:
            return run_query(question)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "RateLimitError" in str(e):
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    print(f"  Rate limited. Waiting {wait}s before retry {attempt + 1}/{MAX_RETRIES}...")
                    time.sleep(wait)
                    continue
            raise
    raise RuntimeError(f"Failed to get a response for question after {MAX_RETRIES} retries: {question!r}")


def build_ragas_dataset(test_set: list[dict]) -> list[dict]:
    """
    Runs each test question through DocTrust's pipeline and collects
    everything RAGAS needs: question, answer, retrieved contexts, ground truth.
    """
    rows = []
    for i, item in enumerate(test_set):
        question = item["question"]
        print(f"\n[{i + 1}/{len(test_set)}] Running: {question!r}")

        # Get retrieved context chunks directly (faster than re-parsing crew logs)
        retrieved = retrieve(question, top_k=3)
        contexts = [r["text"] for r in retrieved]

        # Run the full pipeline (with retry/backoff) to get the actual answer
        result = run_query_with_retry(question)
        answer = result["final_answer"]

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["ground_truth"],
        })

        print(f"  Answer: {answer[:100]}...")

        # Pace requests to stay under Groq's free-tier tokens-per-minute limit
        if i < len(test_set) - 1:
            print(f"  Waiting {SECONDS_BETWEEN_QUESTIONS}s before next question...")
            time.sleep(SECONDS_BETWEEN_QUESTIONS)

    return rows


def run_evaluation():
    print("Building evaluation dataset by running the pipeline on the test set...")
    rows = build_ragas_dataset(TEST_SET)

    dataset = EvaluationDataset.from_list(rows)

    print("\nSetting up RAGAS with Groq LLM and local embeddings...")
    ragas_llm = LangchainLLMWrapper(ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        LLMContextPrecisionWithReference(llm=ragas_llm),
    ]

    print("\nRunning RAGAS evaluation (sequential, this will take a while)...\n")
    results = evaluate(dataset=dataset, metrics=metrics, run_config=RAGAS_RUN_CONFIG)

    df = results.to_pandas()
    print("\n=== RAGAS RESULTS (per question) ===")
    print(df[["user_input", "faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]])

    print("\n=== Missing/failed scores per metric ===")
    print(df[["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]].isna().sum())

    print("\n=== RAGAS RESULTS (averages, over non-missing rows only) ===")
    print(df[["faithfulness", "answer_relevancy", "llm_context_precision_with_reference"]].mean())

    # Save results to a file for the README / resume evidence
    output_path = Path(__file__).resolve().parents[2] / "eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to {output_path}")


if __name__ == "__main__":
    run_evaluation()