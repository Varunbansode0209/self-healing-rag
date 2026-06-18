# graph.py
# ============================================================
# Self-Healing RAG — LangGraph Pipeline
# Architecture: Corrective RAG + Agentic RAG hybrid
# Phase 3 — building node by node
# ============================================================

from typing import TypedDict, List, Literal
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os

load_dotenv()

# ============================================================
# SHARED STATE
# The single memory object passed between all nodes.
# Every node reads from it and writes back to it.
# ============================================================

class GraphState(TypedDict):
    question:        str            # original user question
    domain:          str            # which knowledge base to search
    documents:       List[Document] # retrieved chunks
    generation:      str            # LLM generated answer
    retry_count:     int            # how many retries so far
    retrieval_grade: str            # "relevant" or "irrelevant"
    hallucination:   str            # "grounded" or "hallucinated"
    answer_grade:    str            # "useful" or "not useful"
    final_answer:    str            # what user actually sees


# ============================================================
# SHARED RESOURCES
# Load once, reuse across all nodes.
# ============================================================

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)

# Groq — free, fast, used for grader + reformulator nodes
GROQ_FAST = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,              # zero temp = consistent grading
    api_key=os.getenv("GROQ_API_KEY")
)

# Groq — slightly larger model for generation
GROQ_GEN = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

MAX_RETRIES = 3


# ============================================================
# NODE 1 — RETRIEVER
# Job: search vector store, return top 4 chunks
# No LLM needed — pure vector similarity search
# ============================================================

def retrieve(state: GraphState) -> dict:
    """
    Retrieves relevant document chunks from ChromaDB.
    Uses the current question — which may be reformulated on retry.
    """
    print(f"\n[NODE 1 — RETRIEVE] attempt {state['retry_count'] + 1}")
    print(f"  Question: {state['question'][:80]}")

    vectorstore = Chroma(
        persist_directory=f"./chroma_db/{state['domain']}",
        embedding_function=EMBEDDINGS,
        collection_name=state["domain"]
    )

    documents = vectorstore.similarity_search(
        state["question"],
        k=4
    )

    print(f"  Retrieved: {len(documents)} chunks")
    for i, doc in enumerate(documents):
        print(f"  Chunk {i+1}: {doc.metadata.get('filename','?')[:50]}")

    return {"documents": documents}


# ============================================================
# NODE 2 — RELEVANCE GRADER
# Job: check if retrieved chunks are relevant to the question
# Runs BEFORE generation — catches bad retrieval early
# ============================================================

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance grader.

Given a document chunk and a user question, decide if the chunk
contains information that could help answer the question.

Be generous — if the chunk is even partially relevant, say relevant.
Only say irrelevant if the chunk has absolutely nothing to do with the question.

Return ONLY one word: relevant or irrelevant
No explanation. No punctuation. Just the single word."""),
    ("human", """Document chunk:
{document}

Question: {question}

Your grade (relevant or irrelevant):""")
])

relevance_grader = RELEVANCE_PROMPT | GROQ_FAST | StrOutputParser()


def grade_retrieval(state: GraphState) -> dict:
    """
    Grades each retrieved chunk for relevance.
    If ANY chunk is relevant → proceed to generation.
    If ALL chunks irrelevant → trigger query reformulation.
    """
    print(f"\n[NODE 2 — GRADE RETRIEVAL]")

    question  = state["question"]
    documents = state["documents"]

    relevant_docs = []

    for i, doc in enumerate(documents):
        grade = relevance_grader.invoke({
            "document": doc.page_content[:500],
            "question": question
        }).strip().lower()

        grade = grade.replace(".", "").replace(",", "").strip()

        print(f"  Chunk {i+1}: {grade}")

        # FIXED BUG
        if grade == "relevant":
            relevant_docs.append(doc)

    if relevant_docs:
        print(
            f"  Result: {len(relevant_docs)}/{len(documents)} chunks relevant → PROCEED"
        )

        return {
            "documents": relevant_docs,
            "retrieval_grade": "relevant"
        }

    else:
        print(
            f"  Result: 0/{len(documents)} chunks relevant → REFORMULATE"
        )

        return {
            "documents": [],
            "retrieval_grade": "irrelevant"
        }


# ============================================================
# NODE 3 — GENERATOR
# Job: generate answer using ONLY retrieved chunks
# ============================================================

GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant specializing in {domain}.

Answer the question using ONLY the provided context documents below.
Do not use any external knowledge or training data.

Rules:
- If context contains the answer: provide a clear, detailed answer
- Always mention which source document supports your answer
- If context does NOT contain enough information: respond with exactly:
  "I don't have enough information in my knowledge base to answer this."
- Never make up information not present in the context

Context:
{context}"""),
    ("human", "{question}")
])

generator = GENERATION_PROMPT | GROQ_GEN | StrOutputParser()


def generate(state: GraphState) -> dict:
    """
    Generates answer from relevant chunks only.
    Strictly constrained to provided context.
    """
    print(f"\n[NODE 3 — GENERATE]")

    question  = state["question"]
    documents = state["documents"]
    domain    = state["domain"]

    # Safety check
    if not documents:
        print(f"  No documents to generate from → early exit")

        return {
            "generation":
                "I don't have enough information in my knowledge base to answer this."
        }

    context = "\n\n---\n\n".join([
        f"Source {i+1} ({doc.metadata.get('filename','unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(documents)
    ])

    generation = generator.invoke({
        "domain": domain,
        "context": context,
        "question": question
    })

    print(f"  Generated: {generation[:100]}...")

    return {
        "generation": generation
    }


# ============================================================
# ROUTING FUNCTIONS
# These decide which node to go to next.
# Called by conditional edges in the graph.
# ============================================================

def route_after_grading(
    state: GraphState
) -> Literal["generate", "reformulate_query", "fallback"]:
    """
    After relevance grading:
    relevant   → generate answer
    irrelevant → reformulate query (if retries left)
    irrelevant + no retries → fallback
    """
    if state["retrieval_grade"] == "relevant":
        print(f"\n[ROUTE] relevant chunks → generate")
        return "generate"
    elif state["retry_count"] >= MAX_RETRIES:
        print(f"\n[ROUTE] irrelevant + max retries → fallback")
        return "fallback"
    else:
        print(f"\n[ROUTE] irrelevant → reformulate query")
        return "reformulate_query"


# ============================================================
# PLACEHOLDER NODES (Phase 3 — adding tomorrow)
# These exist so the graph compiles today.
# Replace one by one tomorrow.
# ============================================================

def check_hallucination(state: GraphState) -> dict:
    """
    PLACEHOLDER — Phase 3 Day 2
    For now: assume all generations are grounded.
    Tomorrow: LLM critic checks every claim.
    """
    print(f"\n[NODE 4 — HALLUCINATION CHECK] placeholder → grounded")
    return {"hallucination": "grounded"}


def grade_answer(state: GraphState) -> dict:
    """
    PLACEHOLDER — Phase 3 Day 2
    For now: assume all answers are useful.
    Tomorrow: LLM grades answer quality.
    """
    print(f"\n[NODE 5 — ANSWER GRADE] placeholder → useful")
    return {"answer_grade": "useful"}


def reformulate_query(state: GraphState) -> dict:
    """
    PLACEHOLDER — Phase 3 Day 2
    For now: add [RETRY] prefix as a simple reformulation.
    Tomorrow: LLM rewrites query intelligently.
    """
    import re
    current_retries = state.get("retry_count", 0)
    # Strip any existing [RETRY N] prefix to avoid accumulation on multiple retries
    original_question = re.sub(r'^\[RETRY \d+\]\s*', '', state["question"])
    new_question      = f"[RETRY {current_retries+1}] {original_question}"
    print(f"\n[NODE 6 — REFORMULATE] placeholder → {new_question[:60]}")
    return {
        "question":    new_question,
        "retry_count": current_retries + 1
    }


def fallback(state: GraphState) -> dict:
    """Called when max retries exhausted."""
    print(f"\n[NODE 7 — FALLBACK] max retries reached")
    return {
        "final_answer": (
            "I searched my knowledge base thoroughly across multiple "
            "attempts but couldn't find reliable information to answer "
            "your question accurately. This topic may not be covered in "
            "my current document corpus. Please try rephrasing your "
            "question or consult additional sources."
        )
    }


def finalize(state: GraphState) -> dict:
    """Packages verified answer for user."""
    print(f"\n[NODE 8 — FINALIZE] answer verified → returning to user")
    return {"final_answer": state["generation"]}


# ============================================================
# ROUTING FOR PLACEHOLDER NODES
# Simple pass-through for now — will add real logic tomorrow
# ============================================================

def route_after_hallucination(
    state: GraphState
) -> Literal["grade_answer", "reformulate_query", "fallback"]:
    if state["hallucination"] == "grounded":
        return "grade_answer"
    elif state["retry_count"] >= MAX_RETRIES:
        return "fallback"
    else:
        return "reformulate_query"


def route_after_answer_grade(
    state: GraphState
) -> Literal["finalize", "reformulate_query", "fallback"]:
    if state["answer_grade"] == "useful":
        return "finalize"
    elif state["retry_count"] >= MAX_RETRIES:
        return "fallback"
    else:
        return "reformulate_query"


# ============================================================
# BUILD THE GRAPH
# ============================================================

def build_graph():
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("retrieve",            retrieve)
    graph.add_node("grade_retrieval",     grade_retrieval)
    graph.add_node("generate",            generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("grade_answer",        grade_answer)
    graph.add_node("reformulate_query",   reformulate_query)
    graph.add_node("fallback",            fallback)
    graph.add_node("finalize",            finalize)

    # Entry point
    graph.set_entry_point("retrieve")

    # Fixed edges
    graph.add_edge("retrieve",            "grade_retrieval")
    graph.add_edge("generate",            "check_hallucination")
    graph.add_edge("reformulate_query",   "retrieve")   # THE HEALING LOOP
    graph.add_edge("fallback",            END)
    graph.add_edge("finalize",            END)

    # Conditional edges
    graph.add_conditional_edges(
        "grade_retrieval",
        route_after_grading,
        {
            "generate":          "generate",
            "reformulate_query": "reformulate_query",
            "fallback":          "fallback"
        }
    )

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "grade_answer":      "grade_answer",
            "reformulate_query": "reformulate_query",
            "fallback":          "fallback"
        }
    )

    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grade,
        {
            "finalize":          "finalize",
            "reformulate_query": "reformulate_query",
            "fallback":          "fallback"
        }
    )

    return graph.compile()


# ============================================================
# MAIN — test the graph directly
# ============================================================

def ask(question: str, domain: str = "developer_docs") -> str:
    """Run a question through the self-healing pipeline."""

    initial_state: GraphState = {
        "question":        question,
        "domain":          domain,
        "documents":       [],
        "generation":      "",
        "retry_count":     0,
        "retrieval_grade": "",
        "hallucination":   "",
        "answer_grade":    "",
        "final_answer":    ""
    }

    app = build_graph()

    print(f"\n{'='*60}")
    print(f"QUESTION: {question}")
    print(f"DOMAIN:   {domain}")
    print(f"{'='*60}")

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print(f"FINAL ANSWER:\n{final_state['final_answer']}")
    print(f"RETRIES USED: {final_state['retry_count']}")
    print(f"{'='*60}\n")

    return final_state["final_answer"]


if __name__ == "__main__":
    # Test 1 — should answer correctly
    ask("What is a Python decorator?")

    # Test 2 — should trigger fallback after retries
    ask("What is the capital of France?")