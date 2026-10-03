"""
chat_app.py
A Streamlit chat interface for DocTrust, calling the FastAPI backend.
Supports multi-turn conversations by sending prior Q&A history with each request.
Run with: streamlit run src/chat_app.py
(Make sure `uvicorn src.api:app` is running separately on port 8000 first.)
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("DOCTRUST_API_KEY")

st.set_page_config(page_title="DocTrust", page_icon="📄", layout="centered")
st.title("📄 DocTrust")
st.caption("Enterprise RAG Copilot with Guardrails & Observability")

if "messages" not in st.session_state:
    st.session_state.messages = []


def build_history() -> list[dict]:
    """Builds a list of {query, answer} pairs from the session's chat history."""
    history = []
    pending_query = None
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            pending_query = msg["content"]
        elif msg["role"] == "assistant" and pending_query is not None:
            history.append({"query": pending_query, "answer": msg["content"]})
            pending_query = None
    return history


# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "feedback_given" not in msg:
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("👍 Helpful", key=f"up_{msg['id']}"):
                    try:
                        requests.post(
                            f"{API_URL}/feedback",
                            headers={"X-API-Key": API_KEY},
                            json={
                                "query": msg["query"],
                                "answer": msg["content"],
                                "helpful": True,
                            },
                            timeout=10,
                        )
                        msg["feedback_given"] = True
                        st.success("Thanks for the feedback!")
                    except Exception as e:
                        st.error(f"Could not send feedback: {e}")
            with col2:
                if st.button("👎 Not helpful", key=f"down_{msg['id']}"):
                    try:
                        requests.post(
                            f"{API_URL}/feedback",
                            headers={"X-API-Key": API_KEY},
                            json={
                                "query": msg["query"],
                                "answer": msg["content"],
                                "helpful": False,
                            },
                            timeout=10,
                        )
                        msg["feedback_given"] = True
                        st.info("Thanks, we'll use this to improve.")
                    except Exception as e:
                        st.error(f"Could not send feedback: {e}")

# --- Chat input ---
if prompt := st.chat_input("Ask a question about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt, "id": len(st.session_state.messages)})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = build_history()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    headers={"X-API-Key": API_KEY},
                    json={"query": prompt, "history": history},
                    timeout=120,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)

                    if data.get("citations"):
                        st.caption(f"Sources: {', '.join(data['citations'])}")
                    if data.get("confidence") is not None:
                        st.caption(f"Confidence: {data['confidence']:.2f}")
                    if not data.get("allowed", True):
                        st.warning(f"Guardrail note: {data.get('guardrail_reason')}")

                else:
                    answer = f"Error {response.status_code}: {response.text}"
                    st.error(answer)

            except requests.exceptions.ConnectionError:
                answer = "Could not connect to the DocTrust API. Make sure it's running on port 8000."
                st.error(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "query": prompt,
        "id": len(st.session_state.messages),
    })
    st.rerun()