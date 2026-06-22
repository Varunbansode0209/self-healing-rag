
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
    Returns answer, retrieved chunks, retry count, hallucination
    verdict, and answer grade for full pipeline visibility.
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
        "answer":           final_state["final_answer"],
        "chunks":           final_state["documents"],
        "retry_count":      final_state["retry_count"],
        "retrieval_grade":  final_state.get("retrieval_grade", ""),
        "hallucination":    final_state.get("hallucination", ""),
        "answer_grade":     final_state.get("answer_grade", ""),
    }

# ── UI ────────────────────────────────────────────────────────
st.title("🔬 Self-Healing RAG System")
st.caption("Full Self-Healing Pipeline — 8-Node LangGraph")

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
    st.markdown("**🧠 Pipeline Nodes:**")
    st.markdown("""
| # | Node | Role |
|---|------|------|
| 1 | `retrieve` | Vector search |
| 2 | `grade_retrieval` | Relevance check |
| 3 | `generate` | LLM answer |
| 4 | `check_hallucination` | Grounding check |
| 5 | `grade_answer` | Usefulness check |
| 6 | `reformulate_query` | Query rewrite |
| 7 | `fallback` | Max retries exit |
| 8 | `finalize` | Package answer |
""")

    st.divider()
    st.caption("Self-healing: hallucination detection + auto-retry + query reformulation")

# ── HELPERS ──────────────────────────────────────────────────
def _grade_badge(label: str, value: str, good: str, bad_color: str = "red") -> str:
    """Returns a coloured markdown badge for a grade value."""
    if not value:
        return ""
    color = "green" if value == good else bad_color
    return f":{color}[**{label}:** `{value}`]"

# ── Chat history ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role":             "assistant",
        "content":          "Hello! Ask me anything about your ingested documents. I will answer only from verified sources.",
        "chunks":           [],
        "retry_count":      0,
        "retrieval_grade":  "",
        "hallucination":    "",
        "answer_grade":     "",
    })

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            badge_cols = st.columns(4)

            # Retry badge
            if msg.get("retry_count", 0) > 0:
                badge_cols[0].caption(f"🔄 Self-healed — retries: {msg['retry_count']}")

            # Retrieval grade badge
            rg = msg.get("retrieval_grade", "")
            if rg:
                color = "green" if rg == "relevant" else "orange"
                badge_cols[1].caption(f"📥 Retrieval: :{color}[`{rg}`]")

            # Hallucination badge
            hg = msg.get("hallucination", "")
            if hg:
                color = "green" if hg == "grounded" else "red"
                badge_cols[2].caption(f"🧪 Hallucination: :{color}[`{hg}`]")

            # Answer grade badge
            ag = msg.get("answer_grade", "")
            if ag:
                color = "green" if ag == "useful" else "orange"
                badge_cols[3].caption(f"✅ Answer: :{color}[`{ag}`]")

            # Show sources
            if msg.get("chunks"):
                with st.expander(f"📄 Sources ({len(msg['chunks'])} chunks retrieved)"):
                    for i, chunk in enumerate(msg["chunks"]):
                        st.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename', '?')[:45]}`")
                        st.caption(chunk.page_content[:300] + "...")
                        st.divider()

# ── Chat input ────────────────────────────────────────────────
if question := st.chat_input(f"Ask about {domain}..."):

    # Show user message
    st.session_state.messages.append({
        "role": "user", "content": question,
        "chunks": [], "retry_count": 0,
        "retrieval_grade": "", "hallucination": "", "answer_grade": "",
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from self-healing graph
    with st.chat_message("assistant"):
        with st.spinner("🔍 Running self-healing pipeline..."):
            result = get_answer(question, domain)

        st.markdown(result["answer"])

        # Pipeline grade badges
        badge_cols = st.columns(4)

        if result["retry_count"] > 0:
            badge_cols[0].caption(f"🔄 Self-healed — retries: {result['retry_count']}")

        rg = result.get("retrieval_grade", "")
        if rg:
            color = "green" if rg == "relevant" else "orange"
            badge_cols[1].caption(f"📥 Retrieval: :{color}[`{rg}`]")

        hg = result.get("hallucination", "")
        if hg:
            color = "green" if hg == "grounded" else "red"
            badge_cols[2].caption(f"🧪 Hallucination: :{color}[`{hg}`]")

        ag = result.get("answer_grade", "")
        if ag:
            color = "green" if ag == "useful" else "orange"
            badge_cols[3].caption(f"✅ Answer: :{color}[`{ag}`]")

        # Show sources
        if result["chunks"]:
            with st.expander(f"📄 Sources ({len(result['chunks'])} chunks retrieved)"):
                for i, chunk in enumerate(result["chunks"]):
                    st.markdown(f"**Chunk {i+1}** — `{chunk.metadata.get('filename', '?')[:45]}`")
                    st.caption(chunk.page_content[:300] + "...")
                    st.divider()

    # Save to history
    st.session_state.messages.append({
        "role":            "assistant",
        "content":         result["answer"],
        "chunks":          result["chunks"],
        "retry_count":     result["retry_count"],
        "retrieval_grade": result.get("retrieval_grade", ""),
        "hallucination":   result.get("hallucination", ""),
        "answer_grade":    result.get("answer_grade", ""),
    })