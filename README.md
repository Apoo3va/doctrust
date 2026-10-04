# DocTrust

**Enterprise RAG Copilot with Guardrails & Observability**

DocTrust is a multi agent Retrieval Augmented Generation system that answers employee questions from a company's own documents, pulling from PDFs, wiki pages, and CSV records. It stays grounded in what it actually retrieves, refuses to answer when it isn't confident, and tracks its own cost, latency, and quality over time.

I built this to go past the usual "chatbot over documents" demo and actually deal with the stuff real enterprise RAG systems have to deal with. Multi source search, knowledge engineering, context engineering, multiple agents coordinating with each other, guardrails, automated evaluation, observability, and cost aware model routing. All of it running on free infrastructure.

---

## Why this project

Most RAG tutorials stop at retrieving a few chunks and stuffing them into a prompt. Real deployments need more than that.

Documents come from different sources with messy formatting, and they genuinely contradict each other sometimes. Answers need to be grounded and checkable, not just confident sounding. Systems need to be observable, so you can see what something is costing and how fast it's running, and testable, with real unit tests and CI instead of just manual spot checks. Free tier LLM APIs also come with real limits: rate limits, models getting deprecated overnight, inconsistent JSON output. You have to actually engineer around that, not just assume it won't happen.

DocTrust tries to deal with all of this using a dataset that's deliberately a bit messy and partly contradictory, so the pipeline gets tested the way a real one would be.

---

## Architecture

```mermaid
flowchart TD
    A[User Query] --> B[PII Guardrail Check]
    B -->|Clean| C[Query Contextualizer<br/>multi turn rewrite]
    B -->|Contains PII| Z[Refuse: PII in query]
    C --> D[Semantic Cache Lookup]
    D -->|Hit| Y[Return Cached Answer]
    D -->|Miss| E[Cost Aware Model Router]
    E --> F[Retriever Agent]
    F --> G[Hybrid Retrieval:<br/>Vector Search + Metadata Filter]
    G --> H[Synthesizer Agent]
    H --> I[Validator Agent]
    I --> J[Guardrails:<br/>PII + Grounding + Confidence]
    J -->|Pass| K[Final Answer + Citations]
    J -->|Fail| L[Honest Refusal]
    K --> M[Observability Log:<br/>cost, latency, tokens]
    L --> M
    K --> N[Semantic Cache Store]
```

Here's what happens on each query:

1. **PII guardrail first.** If the query contains an email, phone number, or ID like number, it gets rejected before any LLM calls happen. No point spending tokens on something we're going to refuse anyway.
2. **Query contextualization.** In a multi turn conversation, a follow up like "what about for interns?" gets rewritten into a full standalone question using the recent conversation history.
3. **Semantic cache.** If a near duplicate question has already been answered (cosine similarity of 0.92 or higher on sentence embeddings), it's served from cache instead of running the whole pipeline again.
4. **Cost aware model routing.** Short, simple questions go to a cheaper, faster Groq model. Longer, multi part questions get routed to a stronger one.
5. **Retriever agent.** Calls a hybrid retrieval tool that combines vector similarity search (ChromaDB with all MiniLM L6 v2 embeddings) with metadata filters from a keyword based department and category router.
6. **Synthesizer agent.** Answers strictly from what was retrieved, and returns a structured JSON object with the answer, citations, and a confidence score it assigns itself.
7. **Validator agent.** Independently checks whether every claim in the synthesized answer is actually backed by the retrieved context, and can fail the answer if it spots something unsupported or contradictory.
8. **Guardrails.** Block the final answer if the query or answer contains PII, if the validator failed the grounding check, or if confidence is too low. The user gets an honest "I don't know" instead of a confident wrong answer.
9. **Observability.** Every query logs latency, token usage, estimated cost, and the guardrail outcome, all visualized in a Streamlit dashboard.

---

## A real finding: catching a conflict across documents

The dataset has a contradiction built into it on purpose. A travel policy PDF says the domestic travel allowance is INR 1800 per day. A separate FAQ record says INR 1500 per day. That's exactly the kind of inconsistency that builds up in real company knowledge bases over time.

When asked what the daily travel allowance for domestic trips is, the retriever correctly pulled both conflicting chunks. On one run, the synthesizer went with one of the numbers, but the validator agent caught the contradiction on its own:

> "The claim that the daily travel allowance for domestic trips is INR 1800 contradicts the information in Chunk 1, which states the allowance is INR 1500. Since the answer presents only one of the conflicting values without addressing the discrepancy, it is not fully grounded in the provided context." Verdict: fail.

Guardrails blocked the answer after that, and the user got an honest refusal instead of a confidently wrong number. On a different run, the synthesizer surfaced both figures with their sources instead of picking one. Either way, the system never just silently presented one conflicting value as the truth, which is really the whole point of having a validator and guardrails instead of trusting one LLM call to get it right.

This also led to a real bug getting found and fixed during development. The retriever agent's iteration budget was originally set too tight, at one iteration. That meant it successfully called the retrieval tool, but then had nothing left to actually report the result with, so the model's forced "final answer" turned into a confused refusal instead of passing along what the tool had already found. Bumping that to two iterations, enough to call the tool once and then report it, without letting it call the tool again, fixed it completely. It's a good example of how tight an agent's iteration budget can fail in a way that has nothing to do with the retrieval itself actually working.

---

## Evaluation (RAGAS)

Answer quality is measured automatically with RAGAS against a 10 question hand written test set covering all the source types, run sequentially against Groq's free tier.

| Metric | Score |
|---|---|
| **Faithfulness** | **0.900** |
| **Answer Relevancy** | **0.675** |
| **Context Precision** | **0.833** |

All 30 metric computations across the 10 questions and 3 metrics completed with zero missing or failed scores. Full per question results are in [`eval_results.csv`](eval_results.csv).

---

## Features

**Multi source enterprise search** across PDF, Markdown, and CSV, with a keyword based query router filtering by department and category alongside vector similarity.

**Knowledge engineering** that tags every chunk with policy name, department, date, and category metadata at ingestion time, using an LLM for entity extraction.

**Context engineering** through hybrid retrieval that combines dense vector search with structured metadata filters, falling back to an unfiltered search if the filter comes back empty.

**Multi agent coordination** using CrewAI, with separate retriever, synthesizer, and validator agents each responsible for one narrow, checkable thing.

**Guardrails** that check for PII in both the query and the answer, check grounding and catch hallucinations, and enforce a confidence threshold, all before anything reaches the user.

**Automated evaluation** through a RAGAS harness scoring faithfulness, answer relevancy, and context precision, with retry and backoff tuned to survive a free tier rate limit.

**Observability** through OpenTelemetry tracing, per query cost estimation, and a Streamlit dashboard showing latency, cost, token usage, and guardrail pass rate over time.

**Cost optimization** through a query complexity router that picks a cheaper model for simple questions and a stronger one for complex questions, plus a semantic cache that reuses answers for near duplicate queries.

**Multi turn conversation memory**, where follow up questions get rewritten into standalone questions using recent conversation history.

**A feedback loop** with thumbs up and thumbs down feedback on answers, stored and summarized through an API endpoint.

**Structured logging** across every stage of the pipeline, written to both the console and a persistent log file under one consistent namespace.

**API authentication**, with the FastAPI backend protected by an API key header.

**Tested and containerized**, with 24 pytest unit tests covering chunking, PII detection, routing, and guardrail logic (zero API calls, runs in under a second), a Dockerfile for full containerized deployment, and a GitHub Actions workflow running the test suite on every push.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq API (free tier), openai/gpt oss 20b and openai/gpt oss 120b |
| Multi agent framework | CrewAI |
| Embeddings | sentence transformers, all MiniLM L6 v2, local and free |
| Vector store | ChromaDB, persistent, local |
| API | FastAPI and Uvicorn |
| UI | Streamlit, chat interface and observability dashboard |
| Evaluation | RAGAS |
| Observability | OpenTelemetry |
| Testing | Pytest |
| CI | GitHub Actions |
| Containerization | Docker |

Everything here runs on free tiers. No paid API keys required.

---

## Project structure

```
Doctrust/
├── data/
│   ├── pdf/              # Policy documents (generated)
│   ├── wiki/              # Markdown knowledge base pages
│   └── records/           # CSV FAQ records
├── src/
│   ├── ingestion/          # Loaders, chunker, ingestion pipeline
│   ├── knowledge/          # Entity extraction schema + extractor
│   ├── routing/            # Department/category query router
│   ├── agents/             # CrewAI agents, tools, model router, contextualizer
│   ├── guardrails/         # PII detection, grounding/confidence guardrails
│   ├── observability/      # Tracing, semantic cache, feedback, logging, dashboard
│   ├── eval/               # RAGAS test set + evaluation harness
│   ├── api.py               # FastAPI app
│   └── chat_app.py          # Streamlit chat UI
├── tests/                   # Pytest unit tests
├── .github/workflows/       # CI pipeline
├── Dockerfile
├── requirements.txt
└── eval_results.csv          # Latest RAGAS evaluation output
```

---

## Running locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Set GROQ_API_KEY and DOCTRUST_API_KEY in a .env file

python src/ingestion/run.py --path data

uvicorn src.api:app --reload           # in one terminal
streamlit run src/chat_app.py          # in another
```

Run tests:

```bash
pytest tests/ -v
```

Run evaluation:

```bash
python src/eval/run_eval.py
```

Or just run the whole thing in Docker:

```bash
docker build -t doctrust .
docker run -p 8000:8000 doctrust
```

---

## Known limitations

The router's department and category matching is keyword based rather than semantic, so some queries end up falling back to an unfiltered vector search.

The validator doesn't always catch conflicts the same way every time. Sometimes it surfaces both conflicting values, sometimes it blocks the answer outright. It has never been caught confidently presenting one conflicting value as the only truth, but the behavior isn't perfectly consistent run to run.

Confidence scores are self reported by the synthesizer LLM rather than calibrated against a held out set, so they should be read as a relative signal rather than an actual probability.

Running on the free Groq tier means occasional latency and rate limit retries under heavy load. The pipeline is built to degrade gracefully with a friendly error message instead of crashing when that happens.

---

## Author

**Apoorva Yadav**

---

## License

MIT