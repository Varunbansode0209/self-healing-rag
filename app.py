
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph, GraphState

load_dotenv()

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Healing RAG",
    page_icon="🔬",
    layout="wide"
)

# ── LOAD GRAPH (cached — compiled once, reused every call) ────
@st.cache_resource
def get_graph():
    return build_graph()

# ── RAG via LangGraph pipeline ────────────────────────────────
def get_answer(question: str, domain: str) -> dict:
    """
    Runs the question through the self-healing LangGraph pipeline.
    Returns answer, retrieved chunks, and retry count.
    """
    app = get_graph()

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

    final_state = app.invoke(initial_state)

    return {
        "answer":       final_state["final_answer"],
        "chunks":       final_state["documents"],
        "retry_count":  final_state["retry_count"]
    }

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Self-Healing RAG System")
st.caption("Phase 3 — Self-Healing LangGraph Pipeline")

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
    st.caption("Self-healing: hallucination detection + auto-retry + query reformulation")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     "Hello! Ask me anything about Python documentation. I will answer only from verified sources.",
        "chunks":      [],
        "retry_count": 0
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show retries badge for assistant messages
        if msg["role"] == "assistant" and msg.get("retry_count", 0) > 0:
            st.caption(f"🔄 Self-healed — retries used: {msg['retry_count']}")

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("chunks"):
            with st.expander(f"📄 Sources ({len(msg['chunks'])} chunks retrieved)"):
                for i, chunk in enumerate(msg["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename', '?')[:45]}`")
                    st.caption(chunk.page_content[:300] + "...")
                    st.divider()

# Chat input
if question := st.chat_input(f"Ask about {domain}..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user", "content": question,
        "chunks": [], "retry_count": 0
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from self-healing graph
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            result = get_answer(question, domain)

        st.markdown(result["answer"])

        # Show retries badge
        if result["retry_count"] > 0:
            st.caption(f"🔄 Self-healed — retries used: {result['retry_count']}")

        # Show sources
        if result["chunks"]:
            with st.expander(f"📄 Sources ({len(result['chunks'])} chunks retrieved)"):
                for i, chunk in enumerate(result["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename', '?')[:45]}`")
                    st.caption(chunk.page_content[:300] + "...")
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role":        "assistant",
        "content":     result["answer"],
        "chunks":      result["chunks"],
        "retry_count": result["retry_count"]
    })