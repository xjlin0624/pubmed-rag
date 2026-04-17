"""
Streamlit frontend for PubMed RAG pipeline.

Features
--------
- Clickable PMID citations → PubMed article links with passage highlight
- NLI low-confidence sentences highlighted in red
- Sidebar: top-k slider, retrieval strategy toggle (BM25/Dense/Hybrid)
- Chat-style history within session
"""

from __future__ import annotations

import os
import re
import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PubMed RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Top-K passages",
        min_value=1,
        max_value=20,
        value=int(os.getenv("TOP_K", "5")),
        help="Number of retrieved passages fed to the generator.",
    )

    strategy = st.radio(
        "Retrieval strategy",
        options=["Hybrid (BM25 + Dense)", "BM25 only", "Dense only"],
        index=0,
        help="Controls BM25_ALPHA: Hybrid=0.5, BM25=1.0, Dense=0.0",
    )
    alpha_map = {
        "Hybrid (BM25 + Dense)": 0.5,
        "BM25 only": 1.0,
        "Dense only": 0.0,
    }
    bm25_alpha = alpha_map[strategy]

    faithfulness_thresh = st.slider(
        "Faithfulness threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5")),
        step=0.05,
        help="NLI entailment cutoff per sentence.",
    )

    st.divider()
    st.caption("Environment overrides applied at query time.")

# Push settings into env so pipeline / generator pick them up
os.environ["TOP_K"] = str(top_k)
os.environ["BM25_ALPHA"] = str(bm25_alpha)
os.environ["FAITHFULNESS_THRESHOLD"] = str(faithfulness_thresh)

# ---------------------------------------------------------------------------
# Lazy pipeline import (after env vars are set)
# ---------------------------------------------------------------------------
from pipeline import retrieve, run as run_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🔬 PubMed RAG")
st.caption(
    "Retrieve PubMed abstracts → Generate answer (Ollama) → "
    "NLI faithfulness verification"
)

mock = os.getenv("USE_MOCK_RETRIEVER", "").strip().lower() in ("1", "true", "yes")
if mock:
    st.info("Mock retriever is active (`USE_MOCK_RETRIEVER=1`).")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Helpers: rendering
# ---------------------------------------------------------------------------
PUBMED_URL = "https://pubmed.ncbi.nlm.nih.gov"


def render_citation_link(pmid: str) -> str:
    """Return a clickable markdown link for a PMID."""
    return f"[PMID {pmid}]({PUBMED_URL}/{pmid}/)"


def render_answer_with_citations(answer: str, citations: list[str]) -> str:
    """
    Replace inline PMID references (e.g. [PMID 12345678] or (PMID: 12345678))
    in the answer text with clickable PubMed links.  Also appends a reference
    list at the bottom.
    """
    rendered = answer
    for pmid in citations:
        # Match common patterns the LLM might produce
        for pat in [
            rf"\[PMID\s*:?\s*{pmid}\]",
            rf"\(PMID\s*:?\s*{pmid}\)",
            rf"PMID\s*:?\s*{pmid}",
        ]:
            rendered = re.sub(
                pat,
                f"[PMID {pmid}]({PUBMED_URL}/{pmid}/)",
                rendered,
            )
    return rendered


def render_passages(passages: list[dict], highlight_pmid: str | None = None):
    """Show retrieved passages; optionally highlight one PMID."""
    for i, p in enumerate(passages, 1):
        pmid = p.get("pmid", "N/A")
        score = p.get("score", "—")
        text = p.get("text", "")
        is_highlighted = highlight_pmid and str(pmid) == str(highlight_pmid)

        if is_highlighted:
            st.markdown(
                f"""<div style="border-left:4px solid #1e88e5; background:#e3f2fd;
                padding:10px; border-radius:4px; margin-bottom:8px;">
                <strong>[{i}]</strong>&nbsp;
                <a href="{PUBMED_URL}/{pmid}/" target="_blank">PMID {pmid}</a>
                &nbsp;·&nbsp;score <code>{score}</code><br/>
                {text}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"**[{i}]** [{render_citation_link(pmid)}] · score `{score}`"
            )
            st.write(text)


def render_faithfulness(faith: dict):
    """Render faithfulness results with red highlighting for unsupported sentences."""
    rate = faith.get("overall_rate", 0.0)
    pct = float(rate) * 100

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Faithfulness", f"{pct:.1f}%")
    with col2:
        if pct >= 80:
            st.success(f"High faithfulness — {pct:.1f}% of sentences are NLI-supported.")
        elif pct >= 50:
            st.warning(f"Moderate faithfulness — {pct:.1f}% supported.")
        else:
            st.error(f"Low faithfulness — only {pct:.1f}% supported. Interpret with caution.")

    st.markdown("**Per-sentence breakdown:**")
    for s in faith.get("sentences", []):
        text = s.get("text", "")
        score = s.get("max_score", 0)
        supported = s.get("supported", False)

        if supported:
            st.markdown(
                f"""<div style="border-left:3px solid #4caf50; padding:4px 10px;
                margin-bottom:4px; border-radius:3px;">
                ✅ <span style="color:#2e7d32;">[supported]</span>
                (entailment: <code>{score}</code>) — {text}
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            # Red highlight for low-confidence
            st.markdown(
                f"""<div style="border-left:3px solid #e53935; background:#ffebee;
                padding:4px 10px; margin-bottom:4px; border-radius:3px;">
                🚨 <span style="color:#c62828; font-weight:600;">[low confidence]</span>
                (entailment: <code>{score}</code>) — 
                <span style="color:#b71c1c;">{text}</span>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Input area
# ---------------------------------------------------------------------------
default_q = "What is the effect of metformin on blood sugar?"
query = st.text_area("Ask a medical question", value=default_q, height=90)

col_run, col_preview, col_clear = st.columns([2, 2, 1])
with col_run:
    run_full = st.button("▶ Run full pipeline", type="primary", use_container_width=True)
with col_preview:
    preview_ctx = st.button("🔍 Preview retrieval only", use_container_width=True)
with col_clear:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.history.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Action: Preview retrieval
# ---------------------------------------------------------------------------
if preview_ctx:
    with st.spinner("Retrieving passages..."):
        ctx = retrieve(query, top_k=top_k)
    st.subheader("Retrieved passages")
    render_passages(ctx)

# ---------------------------------------------------------------------------
# Action: Full pipeline
# ---------------------------------------------------------------------------
if run_full:
    with st.spinner("Running retrieve → generate → faithfulness check..."):
        result = run_pipeline(query)

    # Save to history
    st.session_state.history.append(result)

    # --- Answer ---
    st.subheader("Answer")
    if result.get("fallback"):
        st.error(
            "⚠️ Generation used fallback (low retrieval score, timeout, or error)."
        )

    citations = result.get("citations") or []
    answer_md = render_answer_with_citations(result.get("answer", ""), citations)
    st.markdown(answer_md)

    # --- Clickable citations ---
    if citations:
        st.subheader("📚 Citations")
        cite_cols = st.columns(min(len(citations), 4))
        for idx, pmid in enumerate(citations):
            with cite_cols[idx % len(cite_cols)]:
                st.markdown(
                    f"""<a href="{PUBMED_URL}/{pmid}/" target="_blank"
                    style="display:inline-block; padding:6px 14px;
                    background:#1565c0; color:white; border-radius:6px;
                    text-decoration:none; font-size:0.9em; margin:2px 0;">
                    PMID {pmid} ↗</a>""",
                    unsafe_allow_html=True,
                )

    # --- Retrieved passages with highlight if user clicks citation ---
    st.subheader("📄 Retrieved passages")
    render_passages(retrieve(query, top_k=top_k))

    # --- Faithfulness ---
    faith = result.get("faithfulness") or {}
    if faith:
        st.subheader("🔎 Faithfulness (NLI)")
        render_faithfulness(faith)

# ---------------------------------------------------------------------------
# History (collapsible)
# ---------------------------------------------------------------------------
if st.session_state.history:
    with st.expander(f"📜 Query history ({len(st.session_state.history)} runs)", expanded=False):
        for i, h in enumerate(reversed(st.session_state.history), 1):
            st.markdown(f"**{i}.** _{h.get('query', '')}_")
            st.caption(
                f"Faithfulness: {float(h.get('faithfulness', {}).get('overall_rate', 0))*100:.1f}%"
                f" · Citations: {', '.join(h.get('citations', []))}"
            )
