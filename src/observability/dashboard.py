"""
dashboard.py
A Streamlit dashboard visualizing DocTrust's observability log:
cost, latency, and model usage over time.
Run with: streamlit run src/observability/dashboard.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path(__file__).resolve().parents[2] / "observability_log.csv"

st.set_page_config(page_title="DocTrust Observability", layout="wide")
st.title("DocTrust Observability Dashboard")
st.caption("Live metrics from every query run through the DocTrust pipeline.")

if not LOG_PATH.exists():
    st.warning(
        "No observability log found yet. Run a few queries through `src/agents/crew.py` "
        "or the FastAPI `/query` endpoint first -- each one appends a row to "
        "`observability_log.csv`."
    )
    st.stop()

df = pd.read_csv(LOG_PATH)

if df.empty:
    st.info("The observability log exists but has no rows yet.")
    st.stop()

# --- Summary metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Queries", len(df))
col2.metric("Avg Latency (s)", f"{df['latency_seconds'].mean():.2f}")
col3.metric("Total Est. Cost (USD)", f"${df['estimated_cost_usd'].sum():.4f}")
allowed_rate = (df["allowed"].astype(str).str.lower() == "true").mean() * 100
col4.metric("Guardrail Pass Rate", f"{allowed_rate:.0f}%")

st.divider()

# --- Latency over time ---
st.subheader("Latency per Query")
st.line_chart(df["latency_seconds"])

# --- Cost over time ---
st.subheader("Estimated Cost per Query (USD)")
st.line_chart(df["estimated_cost_usd"])

# --- Model usage split ---
st.subheader("Model Usage")
model_counts = df["model"].value_counts()
st.bar_chart(model_counts)

# --- Token usage breakdown ---
st.subheader("Token Usage per Query")
st.bar_chart(df[["prompt_tokens", "completion_tokens"]])

st.divider()

# --- Raw log table ---
st.subheader("Raw Query Log")
st.dataframe(df, use_container_width=True)