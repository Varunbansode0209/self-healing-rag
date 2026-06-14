# app.py
# ============================================================
# Basic RAG Chat UI — Phase 2
# Run with: streamlit run app.py
# ============================================================

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🔬",
    layout="wide"
)

# ── LOAD MODELS (cached so they don't reload every message) ──
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

@st.cache_resource
def load_vectorstore(domain: str):
    return Chroma(
        persist_directory=f"./chroma_db/{domain}",
        embedding_function=load_embeddings(),
        collection_name=domain
    )

# ── RAG CHAIN ─────────────────────────────────────────────────
def get_answer(question: str, domain: str) -> dict:
    """
    Basic RAG — no self-healing yet.
    Phase 3 will replace this with the LangGraph pipeline.
    """
    # Step 1 — Retrieve
    vectorstore = load_vectorstore(domain)
    docs = vectorstore.similarity_search_with_score(question, k=4)

    if not docs:
        return {
            "answer": "No relevant documents found.",
            "chunks": [],
            "scores": []
        }

    # Step 2 — Format context
    context = "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('filename','unknown')}\n{doc.page_content}"
        for doc, score in docs
    ])

    # Step 3 — Generate
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant for {domain} questions.
Answer ONLY using the provided context below.
If the context does not contain enough information, say:
"I don't have enough information in my knowledge base to answer this."

Always cite which source document your answer comes from.

Context:
{context}"""),
        ("human", "{question}")
    ])

    chain  = prompt | load_llm() | StrOutputParser()
    answer = chain.invoke({
        "domain":   domain,
        "context":  context,
        "question": question
    })

    return {
        "answer": answer,
        "chunks": [doc for doc, score in docs],
        "scores": [round(1 - score, 3) for doc, score in docs]
    }

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Self-Healing RAG System")
st.caption("Phase 2 — Basic RAG · Self-healing loop coming in Phase 3")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    domain = st.selectbox(
        "Knowledge Domain",
        options=["developer_docs", "legal", "finance", "health"],
        index=0,
        help="Select which domain to query"
    )

    st.divider()
    st.markdown("**Available domains:**")
    st.markdown("✅ developer_docs — ingested")
    st.markdown("⏳ legal — not yet")
    st.markdown("⏳ finance — not yet")
    st.markdown("⏳ health — not yet")

    st.divider()
    st.caption("Phase 3 will add: hallucination detection, auto-retry, confidence scores")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": "Hello! Ask me anything about Python documentation. I will answer only from verified sources.",
        "chunks":  [],
        "scores":  []
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"📄 Sources ({len(msg['chunks'])} chunks retrieved)"):
                for i, (chunk, score) in enumerate(
                    zip(msg["chunks"], msg["scores"])
                ):
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename','?')[:45]}`")
                    col2.metric("Score", score)
                    st.caption(chunk.page_content[:300] + "...")
                    st.divider()

# Chat input
if question := st.chat_input(f"Ask about {domain}..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user", "content": question,
        "chunks": [], "scores": []
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = get_answer(question, domain)

        st.markdown(result["answer"])

        # Show sources
        if result["chunks"]:
            with st.expander(f"📄 Sources ({len(result['chunks'])} chunks retrieved)"):
                for i, (chunk, score) in enumerate(
                    zip(result["chunks"], result["scores"])
                ):
                    col1, col2 = st.columns([3, 1])
                    col1.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename','?')[:45]}`")
                    col2.metric("Score", score)
                    st.caption(chunk.page_content[:300] + "...")
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "chunks":  result["chunks"],
        "scores":  result["scores"]
    })