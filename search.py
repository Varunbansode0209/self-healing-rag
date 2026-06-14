# search.py
# ============================================================
# Retrieval interface — query → top K chunks
# This becomes Node 1 in your LangGraph later
# Usage: python search.py --domain developer_docs
# ============================================================

import argparse
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ───────────────────────────────────────────────────
CHROMA_BASE = "./chroma_db"

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Test queries per domain
TEST_QUERIES = {
    "developer_docs": [
        "How do I use a for loop in Python?",
        "What is a Python decorator?",
        "How does exception handling work?",
        "What is the difference between list and tuple?",
        "How do I open and read a file in Python?",
        "What is a generator function?",
        "How does Python handle memory management?",
        "What is the Global Interpreter Lock?",
        "How do I use async await in Python?",
        "What are Python type hints?",
    ],
    "legal": [
        "What is the procedure for filing a PIL?",
        "What are the grounds for bail in India?",
        "What is the limitation period for contracts?",
        "What is the Indian Contract Act?",
        "What is anticipatory bail?",
    ],
    "finance": [
        "What are SEBI regulations for mutual funds?",
        "What is the repo rate set by RBI?",
        "What are insider trading rules in India?",
        "What is SEBI circular on portfolio managers?",
        "What are RBI guidelines for NBFCs?",
    ],
    "health": [
        "What is the WHO guideline for diabetes management?",
        "What are symptoms of dengue fever?",
        "What is the recommended vaccine schedule?",
        "What are WHO guidelines for hypertension?",
        "What is the treatment protocol for tuberculosis?",
    ],
}


# ── CORE SEARCH FUNCTION ─────────────────────────────────────

def search(query: str, domain: str, k: int = 4) -> list:
    """
    Main retrieval function.
    This exact function becomes your Retriever Node in LangGraph.

    Args:
        query:  The user question (or reformulated query on retry)
        domain: Which knowledge base to search
        k:      Number of chunks to return (4 is optimal for RAG)

    Returns:
        List of (Document, score) tuples
    """
    db_path = f"{CHROMA_BASE}/{domain}"

    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=EMBEDDINGS,
        collection_name=domain
    )

    results = vectorstore.similarity_search_with_score(query, k=k)
    return results


def display_results(query: str, results: list, domain: str):
    """Pretty prints search results for manual testing."""

    print(f"\n{'─'*60}")
    print(f"Domain : {domain}")
    print(f"Query  : {query}")
    print(f"{'─'*60}")

    if not results:
        print("  NO RESULTS FOUND")
        return

    for i, (doc, score) in enumerate(results):
        similarity = 1 - score   # Convert distance to similarity
        bar_length = int(similarity * 20)
        bar        = "█" * bar_length + "░" * (20 - bar_length)

        print(f"\nChunk {i+1}  [{bar}]  {similarity:.3f}")
        print(f"Source : {doc.metadata.get('filename', 'unknown')[:50]}")
        print(f"URL    : {doc.metadata.get('source', 'unknown')[:70]}")
        print(f"Content: {doc.page_content[:250].strip()}...")

    # Show best score summary
    best = 1 - results[0][1]
    if best >= 0.75:
        verdict = "✅ Strong retrieval"
    elif best >= 0.60:
        verdict = "🟡 Acceptable retrieval"
    else:
        verdict = "⚠️  Weak retrieval — self-healing will compensate"

    print(f"\n  Best score: {best:.3f} → {verdict}")


# ── BATCH TEST ───────────────────────────────────────────────

def run_batch_test(domain: str):
    """
    Runs all test queries for a domain.
    Use this to verify retrieval quality before building Phase 2.
    """
    queries = TEST_QUERIES.get(domain, [])

    print(f"\n{'='*60}")
    print(f"  BATCH TEST — {domain.upper()}")
    print(f"  {len(queries)} queries")
    print(f"{'='*60}")

    scores = []
    for q in queries:
        results = search(q, domain, k=4)
        display_results(q, results, domain)
        if results:
            scores.append(1 - results[0][1])

        input("\n  Press Enter for next query (or Ctrl+C to stop)...")

    # Summary
    if scores:
        avg   = sum(scores) / len(scores)
        above = sum(1 for s in scores if s >= 0.65)
        print(f"\n{'='*60}")
        print(f"  BATCH TEST SUMMARY")
        print(f"{'='*60}")
        print(f"  Avg similarity score : {avg:.3f}")
        print(f"  Good results (≥0.65) : {above}/{len(scores)}")
        print(f"  Weak results (<0.65) : {len(scores)-above}/{len(scores)}")
        if avg >= 0.65:
            print(f"\n  ✅ Retrieval quality good — ready for Phase 2")
        else:
            print(f"\n  ⚠️  Consider adjusting chunk size or adding more docs")


# ── SINGLE QUERY MODE ────────────────────────────────────────

def single_query(query: str, domain: str):
    """Quick single query test."""
    results = search(query, domain, k=4)
    display_results(query, results, domain)


# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search the vector store")
    parser.add_argument("--domain", required=True,
                        choices=["developer_docs","legal","finance","health"],
                        help="Domain to search")
    parser.add_argument("--query", type=str, default=None,
                        help="Single query (optional). Omit to run batch test.")
    parser.add_argument("--k", type=int, default=4,
                        help="Number of chunks to return (default: 4)")
    args = parser.parse_args()

    if args.query:
        # Single query mode
        single_query(args.query, args.domain)
    else:
        # Batch test mode — runs all 10 test queries
        run_batch_test(args.domain)