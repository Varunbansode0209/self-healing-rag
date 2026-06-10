# Self-Healing RAG System

A hallucination-resistant document QA pipeline built with LangGraph and LangChain.
Instead of answering blindly, this system critiques its own output and retries
automatically when it detects hallucination or poor retrieval.

## Architecture

**Type:** Corrective RAG (CRAG) + Agentic RAG hybrid

**Pipeline:**

User Query
↓
Retrieve chunks from vector store
↓
Grade relevance of retrieved chunks
↓ (if irrelevant → reformulate query → retry)
Generate answer using only retrieved context
↓
Critic checks for hallucination
↓ (if hallucinated → reformulate query → retry)
Grade answer quality
↓ (if not useful → reformulate query → retry)
Return verified answer with citations
↓ (if 3 retries exhausted → honest "I don't know")


## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Groq (dev) / GPT-4o-mini (prod) |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB → Qdrant (Phase 4) |
| Evaluation | RAGAs + DeepEval |
| Observability | LangSmith |
| UI | Streamlit |

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 0 | Environment setup | ✅ Complete |
| Phase 1 | Embeddings + Vector DB | 🔄 In Progress |
| Phase 2 | Basic RAG Pipeline | ⏳ Not Started |
| Phase 3 | Self-Healing Loop | ⏳ Not Started |
| Phase 4 | Hybrid Search + Reranker | ⏳ Not Started |
| Phase 5 | Evaluation Dashboard | ⏳ Not Started |
| Phase 6 | Polish + Publish | ⏳ Not Started |

## Results

*(To be filled in Phase 5)*

| Metric | Baseline RAG | Self-Healing RAG | Improvement |
|---|---|---|---|
| Faithfulness | - | - | - |
| Answer Relevancy | - | - | - |
| Hallucination Rate | - | - | - |

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/Varunbansode0209/self-healing-rag.git
cd self-healing-rag

# 2. Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Fill in your keys in .env

# 5. Add documents to /documents folder

# 6. Ingest documents
python ingest.py

# 7. Run the app
streamlit run app.py
```

## Domain

*Developer Documentation — LangChain, LangGraph, FastAPI*

## Key Engineering Decisions

- **Why LangGraph over a simple chain?**
  A chain is linear and runs once. LangGraph supports stateful loops with
  conditional branching — essential for the retry mechanism.

- **Why a separate critic agent?**
  Self-RAG requires fine-tuning a custom model. Using a separate LLM-as-Judge
  achieves the same result with off-the-shelf models.

- **Why hybrid search (Phase 4)?**
  Vector search misses exact keyword matches. BM25 + vector combined with
  a reranker gives significantly better retrieval precision.

## Author

Varun Bansode