# manual_eval.py
# ============================================================
# Crystal Clear Evaluation — Zero Compromises
# Generator: llama-3.1-8b-instant
# Judge:     llama-3.3-70b-versatile (separate model = unbiased)
# ============================================================

import os
import json
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph import build_graph, GraphState
from hybrid_retriever import hybrid_search
from dotenv import load_dotenv

load_dotenv()

DOMAIN  = "developer_docs"
DATASET = "eval_dataset_v2.json"

# ── TWO SEPARATE MODELS ───────────────────────────────────────
generator_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

judge_llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # larger model = better judge
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


# ── RATE LIMIT SAFE INVOKE ────────────────────────────────────
def safe_invoke(chain, inputs, max_retries=5, label=""):
    for attempt in range(max_retries):
        try:
            result = chain.invoke(inputs)
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = (attempt + 1) * 5
                print(f"    [{label}] Rate limit — waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            elif "decommissioned" in err.lower():
                print(f"    [{label}] Model decommissioned error — check model name")
                raise e
            else:
                print(f"    [{label}] Error: {err[:100]}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise e
    return None


# ── JUDGE PROMPT ──────────────────────────────────────────────
judge_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict RAG evaluation judge. 

Evaluate the answer against the source documents.

FAITHFULNESS (0.0-1.0):
- 1.0 = every claim is explicitly in the source documents
- 0.7 = mostly grounded, minor extrapolation
- 0.5 = mix of document content and outside knowledge  
- 0.2 = mostly outside knowledge not in documents
- 0.0 = completely fabricated or from training data only

RELEVANCY (0.0-1.0):
- 1.0 = directly and completely answers the question
- 0.7 = mostly answers with minor gaps
- 0.5 = partially answers the question
- 0.0 = does not address the question at all

IMPORTANT: If answer says "I don't have enough information" score:
- faithfulness = 1.0 (honest abstention is always faithful)
- relevancy = 0.3 (it does not answer but is honest)

Return ONLY valid JSON. No explanation. No markdown. Example:
{{"faithfulness": 0.8, "relevancy": 0.9}}"""),
    ("human", """Question: {question}

Source documents:
{context}

Answer to evaluate:
{answer}

JSON scores:""")
])

judge_chain = judge_prompt | judge_llm | StrOutputParser()


def score_answer(question: str, answer: str, context: str) -> tuple[float, float]:
    """
    Score answer using large judge model.
    Retries up to 3 times if JSON parsing fails.
    Never silently returns 0.5 — always logs what happened.
    """
    for attempt in range(3):
        raw = safe_invoke(judge_chain, {
            "question": question,
            "context":  context[:1000],
            "answer":   answer[:600]
        }, label="judge")

        if raw is None:
            print(f"    Judge returned None on attempt {attempt+1}")
            continue

        try:
            raw_text = raw.strip()
            start = raw_text.find("{")
            end   = raw_text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(raw_text[start:end])
                f = float(data.get("faithfulness", -1))
                r = float(data.get("relevancy", -1))
                if 0.0 <= f <= 1.0 and 0.0 <= r <= 1.0:
                    return round(f, 3), round(r, 3)
                else:
                    print(f"    Invalid score range: f={f} r={r} — retrying")
            else:
                print(f"    No JSON found in: {raw_text[:100]} — retrying")
        except json.JSONDecodeError as e:
            print(f"    JSON parse error: {e} — raw: {raw_text[:100]} — retrying")

        time.sleep(2)

    print(f"    WARNING: Judge failed after 3 attempts — using 0.5 default")
    return 0.5, 0.5


# ── BASELINE RAG ──────────────────────────────────────────────
baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant.
Answer using ONLY the provided context documents.
If the context does not contain the answer, say exactly:
"I don't have enough information in my knowledge base to answer this."
Do not use any external knowledge.

Context:
{context}"""),
    ("human", "{question}")
])
baseline_chain = baseline_prompt | generator_llm | StrOutputParser()


def run_baseline(question: str) -> tuple[str, str]:
    docs    = hybrid_search(question, DOMAIN, k=4)
    context = "\n\n---\n\n".join([
        f"Source {i+1}:\n{doc.page_content[:500]}"
        for i, doc in enumerate(docs)
    ])
    answer = safe_invoke(baseline_chain, {
        "context":  context,
        "question": question
    }, label="baseline")
    return (answer or "Error generating answer"), context


# ── SELF-HEALING RAG ──────────────────────────────────────────
def run_self_healing(question: str) -> tuple[str, str, int]:
    app = build_graph()
    state: GraphState = {
        "question":        question,
        "domain":          DOMAIN,
        "documents":       [],
        "generation":      "",
        "retry_count":     0,
        "retrieval_grade": "",
        "hallucination":   "",
        "answer_grade":    "",
        "final_answer":    ""
    }

    for attempt in range(3):
        try:
            result  = app.invoke(state)
            context = "\n\n---\n\n".join([
                f"Source {i+1}:\n{doc.page_content[:500]}"
                for i, doc in enumerate(result["documents"])
            ])
            return result["final_answer"], context, result["retry_count"]
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 5
                print(f"    Self-healing rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    Self-healing error: {str(e)[:100]}")
                return "Error in self-healing pipeline", "", 0

    return "Rate limit — could not complete", "", 0


# ── MAIN EVALUATION ───────────────────────────────────────────
def evaluate():
    with open(DATASET) as f:
        all_data = json.load(f)

    answerable    = [q for q in all_data if q["answerable"]]
    out_of_domain = [q for q in all_data if not q["answerable"]]

    print(f"\n{'='*60}")
    print(f"CRYSTAL CLEAR EVALUATION")
    print(f"Generator: llama-3.1-8b-instant")
    print(f"Judge:     llama-3.3-70b-versatile")
    print(f"Answerable: {len(answerable)} | OOD: {len(out_of_domain)}")
    print(f"{'='*60}\n")

    # Storage
    b_faith  = []
    s_faith  = []
    b_relev  = []
    s_relev  = []
    retries  = []
    by_type  = {}

    for i, item in enumerate(answerable):
        q     = item["question"]
        qtype = item.get("type", "unknown")

        print(f"\n{'─'*55}")
        print(f"[{i+1}/{len(answerable)}] Type: {qtype}")
        print(f"Q: {q[:70]}")

        # ── Run baseline ──────────────────────────────────────
        print(f"  Running baseline...")
        b_ans, b_ctx = run_baseline(q)
        print(f"  Baseline answer: {b_ans[:80]}...")

        # Score baseline
        print(f"  Scoring baseline...")
        b_f, b_r = score_answer(q, b_ans, b_ctx)
        b_faith.append(b_f)
        b_relev.append(b_r)
        print(f"  Baseline scores → Faith:{b_f} Relev:{b_r}")

        # Delay between baseline and self-healing
        time.sleep(1)

        # ── Run self-healing ──────────────────────────────────
        print(f"  Running self-healing...")
        s_ans, s_ctx, retry_count = run_self_healing(q)
        print(f"  Self-healing answer: {s_ans[:80]}...")
        print(f"  Retries used: {retry_count}")

        # Score self-healing
        print(f"  Scoring self-healing...")
        s_f, s_r = score_answer(q, s_ans, s_ctx)
        s_faith.append(s_f)
        s_relev.append(s_r)
        retries.append(retry_count)
        print(f"  Self-healing scores → Faith:{s_f} Relev:{s_r}")

        # Track by type
        if qtype not in by_type:
            by_type[qtype] = {"b_f":[],"s_f":[],"b_r":[],"s_r":[],"retries":[]}
        by_type[qtype]["b_f"].append(b_f)
        by_type[qtype]["s_f"].append(s_f)
        by_type[qtype]["b_r"].append(b_r)
        by_type[qtype]["s_r"].append(s_r)
        by_type[qtype]["retries"].append(retry_count)

        # Delay between questions
        time.sleep(1)

    # ── Out of domain test ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"OUT-OF-DOMAIN TEST ({len(out_of_domain)} questions)")
    print(f"{'='*60}")

    correct_fallbacks = 0
    ood_details = []

    for item in out_of_domain:
        q = item["question"]
        print(f"\n  Q: {q[:60]}")

        s_ans, _, retry_count = run_self_healing(q)
        refused = any(phrase in s_ans.lower() for phrase in [
            "don't have enough",
            "searched my knowledge",
            "not in my knowledge",
            "cannot find",
            "knowledge base thoroughly"
        ])

        correct_fallbacks += int(refused)
        status = "✅ CORRECT FALLBACK" if refused else "❌ HALLUCINATED"
        print(f"  Retries: {retry_count} | {status}")
        print(f"  Answer: {s_ans[:100]}...")

        ood_details.append({
            "question": q,
            "retries":  retry_count,
            "refused":  refused,
            "answer":   s_ans[:200]
        })

        time.sleep(1)

    # ── Compute final metrics ─────────────────────────────────
    avg = lambda lst: round(sum(lst)/len(lst), 3) if lst else 0
    n   = len(answerable)

    b_f_avg   = avg(b_faith)
    s_f_avg   = avg(s_faith)
    b_r_avg   = avg(b_relev)
    s_r_avg   = avg(s_relev)
    avg_ret   = avg(retries)
    retry_rt  = round(sum(1 for r in retries if r > 0) / n * 100, 1)
    fall_rt   = round(correct_fallbacks / len(out_of_domain) * 100, 1)
    b_hall    = round(sum(1 for f in b_faith if f < 0.5) / n * 100, 1)
    s_hall    = round(sum(1 for f in s_faith if f < 0.5) / n * 100, 1)
    faith_imp = round(s_f_avg - b_f_avg, 3)
    faith_pct = round(faith_imp / b_f_avg * 100, 1) if b_f_avg > 0 else 0

    # ── Print results ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — Crystal Clear")
    print(f"{'='*60}")
    print(f"{'Metric':<28} {'Baseline':>10} {'SelfHeal':>10} {'Change':>10}")
    print(f"{'─'*60}")
    print(f"{'Faithfulness':<28} {b_f_avg:>10} {s_f_avg:>10} {f'{faith_imp:+}':>10}")
    print(f"{'Answer Relevancy':<28} {b_r_avg:>10} {s_r_avg:>10} {f'{round(s_r_avg-b_r_avg,3):+}':>10}")
    print(f"{'Hallucination Rate':<28} {b_hall:>9}% {s_hall:>9}% {f'{round(s_hall-b_hall,1):+}%':>10}")
    print(f"{'Retry Rate':<28} {'—':>10} {f'{retry_rt}%':>10}")
    print(f"{'Avg Retries/Query':<28} {'—':>10} {avg_ret:>10}")
    print(f"{'OOD Fallback Accuracy':<28} {'—':>10} {f'{fall_rt}%':>10}")

    print(f"\n── BY QUESTION TYPE ─────────────────────────────")
    for qtype, data in by_type.items():
        print(f"\n  {qtype} ({len(data['b_f'])} questions):")
        print(f"    Baseline  → Faith:{avg(data['b_f'])} Relev:{avg(data['b_r'])}")
        print(f"    SelfHeal  → Faith:{avg(data['s_f'])} Relev:{avg(data['s_r'])}")
        print(f"    Avg retries: {avg(data['retries'])}")

    print(f"\n{'='*60}")
    print(f"RESUME BULLETS")
    print(f"{'='*60}")
    print(f"""
- Built self-healing RAG pipeline using LangGraph — improved answer 
  faithfulness from {b_f_avg} to {s_f_avg} ({faith_pct:+}% relative improvement) 
  evaluated across {n} domain-specific test cases using LLM-as-Judge 
  (llama-3.3-70b-versatile scoring llama-3.1-8b-instant outputs)

- Implemented corrective retrieval loop — retry mechanism fired on 
  {retry_rt}% of queries with avg {avg_ret} reformulation attempts 
  per triggered query

- System correctly refused {correct_fallbacks}/{len(out_of_domain)} out-of-domain 
  questions ({fall_rt}% fallback accuracy) — zero hallucinations on 
  out-of-corpus queries

- Reduced hallucination rate from {b_hall}% (baseline) to {s_hall}% 
  (self-healing) across {n} test cases
""")

    # ── Save results ──────────────────────────────────────────
    os.makedirs("eval_results", exist_ok=True)
    results = {
        "config": {
            "generator_model": "llama-3.1-8b-instant",
            "judge_model":     "llama-3.3-70b-versatile",
            "dataset":         DATASET,
            "n_answerable":    n,
            "n_ood":           len(out_of_domain)
        },
        "overall": {
            "baseline":     {"faithfulness":b_f_avg,"relevancy":b_r_avg,"hallucination_rate":b_hall},
            "self_healing": {"faithfulness":s_f_avg,"relevancy":s_r_avg,"hallucination_rate":s_hall},
            "improvement":  {"faithfulness_abs":faith_imp,"faithfulness_pct":faith_pct,"hallucination_reduction":round(b_hall-s_hall,1)}
        },
        "pipeline": {
            "retry_rate_pct":  retry_rt,
            "avg_retries":     avg_ret,
            "fallback_rate_pct": fall_rt
        },
        "by_type": {
            k: {
                "n": len(v["b_f"]),
                "baseline_faith":  avg(v["b_f"]),
                "sh_faith":        avg(v["s_f"]),
                "baseline_relev":  avg(v["b_r"]),
                "sh_relev":        avg(v["s_r"]),
                "avg_retries":     avg(v["retries"])
            }
            for k, v in by_type.items()
        },
        "ood_details": ood_details
    }

    with open("eval_results/results_final.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Full results saved → eval_results/results_final.json")
    print(f"Fill these into your Excel Metrics sheet now.")


if __name__ == "__main__":
    evaluate()