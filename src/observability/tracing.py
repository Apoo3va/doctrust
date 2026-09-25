"""
tracing.py
OpenTelemetry instrumentation for DocTrust. Wraps pipeline stages in spans
capturing latency, and logs per-query metrics (latency, tokens, estimated
cost, model used) to a CSV file for later inspection.
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# --- OpenTelemetry setup (console exporter -- prints spans as they complete) ---
provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("doctrust")


@contextmanager
def traced_span(name: str, **attributes):
    """Wraps a block of code in an OpenTelemetry span, recording latency automatically."""
    start = time.time()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        try:
            yield span
        finally:
            elapsed = time.time() - start
            span.set_attribute("latency_seconds", round(elapsed, 3))


# --- Rough Groq pricing for cost estimation (USD per 1M tokens, approximate) ---
MODEL_PRICING = {
    "openai/gpt-oss-20b": {"input": 0.10, "output": 0.50},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.75},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Rough cost estimate in USD based on Groq's approximate per-token pricing."""
    pricing = MODEL_PRICING.get(model, {"input": 0.10, "output": 0.50})
    cost = (prompt_tokens / 1_000_000) * pricing["input"] + \
           (completion_tokens / 1_000_000) * pricing["output"]
    return round(cost, 6)


LOG_PATH = Path(__file__).resolve().parents[2] / "observability_log.csv"


def log_query_metrics(query: str, model: str, latency: float,
                       prompt_tokens: int, completion_tokens: int, allowed: bool):
    """Appends one row of per-query metrics to a CSV log for later inspection."""
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    total_tokens = prompt_tokens + completion_tokens

    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "query", "model", "latency_seconds", "prompt_tokens",
                "completion_tokens", "total_tokens", "estimated_cost_usd", "allowed"
            ])
        writer.writerow([
            query, model, round(latency, 3), prompt_tokens,
            completion_tokens, total_tokens, cost, allowed
        ])

    print(f"[observability] model={model} latency={latency:.2f}s "
          f"tokens={total_tokens} est_cost=${cost:.6f} allowed={allowed}")