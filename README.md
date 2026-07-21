# Self-Healing RAG System

A hallucination-resistant document QA pipeline built with LangGraph and LangChain.
Instead of answering blindly, this system critiques its own output and retries
automatically when it detects hallucination or poor retrieval.

> **Live Demo:** *(Streamlit Cloud link — add in Phase 7)*  
> **Status:** Phase 5 — Evaluation Dashboard

---

## What Makes This Different From a Basic RAG Chatbot

| Standard RAG | Self-Healing RAG |
|---|---|
| 1 API call → 1 answer | 8 specialized agents in a loop |
| No verification | Hallucination critic checks every claim |
| No retry | 3-attempt healing loop with query reformulation |
| No grounding check | Domain-locked — only answers from verified docs |
| No metrics | RAGAs faithfulness + hallucination rate measured |
| Confident when wrong | Honest "I don't know" when knowledge base lacks info |
| Black box | Full reasoning trace visible in UI |

---

## Architecture

**Type:** Corrective RAG (CRAG) + Agentic RAG hybrid  
**Orchestration:** LangGraph StateGraph with 8 specialized nodes


User Query

↓

[Node 1] Hybrid Retriever (BM25 + Vector Search)

↓

[Node 2] Relevance Grader — are chunks relevant?

↓

├── irrelevant → [Node 6] Query Reformulator → retry (max 3)

↓ relevant

[Node 3] Generator — answer from context only

↓

[Node 4] Hallucination Critic — is answer grounded in docs?

↓

├── hallucinated → [Node 6] Query Reformulator → retry

↓ grounded

[Node 5] Answer Grader — does answer address the question?

↓

├── not useful → [Node 6] Query Reformulator → retry

↓ useful

[Node 8] Finalize → return verified answer with citations

↓

[Node 7] Fallback → honest "I don't know" if 3 retries exhausted


---

## Tech Stack

| Layer | Technology | Decision |
|---|---|---|
| Orchestration | LangGraph 1.2.5 | Stateful graph — chains cannot loop |
| LLM (dev) | Groq llama-3.1-8b-instant | Free tier — zero cost during development |
| LLM (prod) | GPT-4o-mini | Final demo and evaluation |
| Embeddings | BAAI/bge-small-en-v1.5 | Local — runs on RTX 4050, zero API cost |
| Vector Store | ChromaDB | Local — sufficient for corpus size |
| Hybrid Search | BM25 + ChromaDB EnsembleRetriever | Domain-aware weights |
| Evaluation | RAGAs | Faithfulness, answer relevancy, hallucination rate |
| Observability | LangSmith | Per-node tracing and latency |
| UI | Streamlit | Python-native, zero JS required |

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Environment setup | ✅ Complete |
| Phase 1 | Document ingestion + embeddings + ChromaDB | ✅ Complete |
| Phase 2 | Basic RAG pipeline + Streamlit UI | ✅ Complete |
| Phase 3 | Self-healing LangGraph loop — all 8 nodes | ✅ Complete |
| Phase 4 | Hybrid search + partial answer handling | ✅ Complete |
| Phase 5 | Evaluation dashboard + RAGAs metrics | 🔄 In Progress |
| Phase 6 | Multi-domain expansion | ⏳ Not Started |
| Phase 7 | MCP server + deploy + publish | ⏳ Not Started |

---

## Results

Evaluated across 28 domain-specific test cases using LLM-as-Judge
(llama-3.3-70b-versatile scoring llama-3.1-8b-instant outputs)

| Metric | Baseline RAG | Self-Healing RAG | Improvement |
|---|---|---|---|
| Faithfulness Score | 0.611 | 0.804 | +31.6% |
| OOD Fallback Accuracy | — | 100% (5/5) | — |
| Retry Rate | — | 14.3% | — |
| Avg Retries on Hard Questions | — | 1.5 | — |
| Test Questions | 28 | 28 | — |

### Evaluation Notes
- Generator model: llama-3.1-8b-instant (Groq)
- Judge model: llama-3.3-70b-versatile (separate model — unbiased scoring)
- Question types: A_easy (10), B_hard (10), C_trick (8), OOD (5)
- Self-healing retry loop fired on C_trick questions (avg 1.5 retries)
  confirming the system correctly identifies ambiguous retrieval

---

## Corpus

| Domain | Documents | Chunks | Status |
|---|---|---|---|
| Developer Docs (Python 3.14) | 555 files | 20,094 chunks | ✅ Ingested |
| Legal | - | - | ⏳ Phase 6 |
| Finance | - | - | ⏳ Phase 6 |
| Health | - | - | ⏳ Phase 6 |

---

## Key Engineering Decisions

**Why LangGraph over a simple LangChain chain?**  
A chain is linear — runs once and stops. LangGraph is a stateful graph with
conditional branching and shared state across nodes. The retry loop physically
cannot be built with a chain. It requires a graph.

**Why a separate LLM-as-Judge critic instead of Self-RAG?**  
Self-RAG requires fine-tuning a custom model with special critique tokens.
Using a separate critic agent achieves the same hallucination detection
with off-the-shelf models — practical to build and easy to explain.

**Why BGE embeddings over OpenAI embeddings during development?**  
BGE-small-en-v1.5 runs locally on RTX 4050 GPU at zero API cost.
Benchmarked against OpenAI text-embedding-3-small — quality sufficient
for development. Switching to OpenAI embeddings for final evaluation run.

**Why keep ChromaDB over Qdrant?**  
Benchmarked migration — no performance improvement justified the complexity.
ChromaDB handles 20,094 chunks efficiently for this use case.

**Why hybrid search with domain-aware weights?**  
Benchmarked BM25 + vector vs vector-only across 10 query types.
BGE embeddings handle Python API terminology well — hybrid weights
tuned to [0.7, 0.3] for developer docs. Legal/finance domains will
use [0.4, 0.6] where exact clause and regulation number matching matters.

**Why partial answer handling?**  
Initial system refused entire questions if any part lacked corpus coverage.
Updated generator prompt answers supported portions and explicitly flags
knowledge gaps — e.g. "My knowledge base does not contain information
about uvloop" — rather than blanket refusal.

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Varunbansode0209/self-healing-rag.git
cd self-healing-rag

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate     # Windows Git Bash
# venv\Scripts\activate.bat      # Windows CMD

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Fill in GROQ_API_KEY and LANGCHAIN_API_KEY

# 5. Put documents in /documents/developer_docs/
# Python 3.14 docs already crawled — see crawler.py

# 6. Ingest documents
python ingest.py --domain developer_docs

# 7. Run the app
streamlit run app.py
```

---

## Observability

LangSmith tracing enabled — every node's input, output, latency,
and token count is logged automatically.

Set in `.env`:

LANGCHAIN_TRACING_V2=true

LANGCHAIN_API_KEY=ls__your-key

LANGCHAIN_PROJECT=self-healing-rag

---

## Author

**Varun Bansode**  
Final Year BE Computer Engineering — APSIT, Thane  
Mumbai, India  
GitHub: github.com/Varunbansode0209/self-healing-rag