"""Streamlit shell for the integrated RAG pipeline (works with mock or real retriever)."""

from __future__ import annotations

import os

import streamlit as st

from pipeline import retrieve, run as run_pipeline

st.set_page_config(page_title="PubMed RAG", layout="wide")
st.title("PubMed RAG")
st.caption("Retrieve → generate (Ollama) → NLI faithfulness")

mock = os.getenv("USE_MOCK_RETRIEVER", "").strip().lower() in ("1", "true", "yes")
if mock:
    st.info("Mock retriever is enabled (`USE_MOCK_RETRIEVER`).")
elif not (
    os.path.isfile(os.getenv("CHUNKS_FILE", "chunks.json"))
    and os.path.isfile(os.getenv("INDEX_FILE", "faiss.index"))
):
    st.warning(
        "Hybrid retriever indexes not found in the working directory; "
        "`pipeline.retrieve` will fall back to mock passages until you run "
        "`python -m retriever.retriever` (or mount pre-built `chunks.json` + `faiss.index`)."
    )

default_q = "What is the effect of metformin on blood sugar?"
query = st.text_area("Question", value=default_q, height=100)

col_a, col_b = st.columns(2)
with col_a:
    run_full = st.button("Run full pipeline", type="primary")
with col_b:
    preview_ctx = st.button("Preview retrieval only")

if preview_ctx:
    with st.spinner("Retrieving..."):
        ctx = retrieve(query, top_k=int(os.getenv("TOP_K", "5")))
    st.subheader("Retrieved passages")
    for i, row in enumerate(ctx, start=1):
        st.markdown(f"**[{i}]** PMID `{row.get('pmid', '')}` · score `{row.get('score', '')}`")
        st.write(row.get("text", ""))

if run_full:
    with st.spinner("Running retrieve → generate → faithfulness..."):
        result = run_pipeline(query)

    st.subheader("Answer")
    if result.get("fallback"):
        st.error("Generation used fallback (low retrieval score, timeout, or error).")
    st.markdown(result.get("answer", ""))

    cites = result.get("citations") or []
    if cites:
        st.subheader("Citations (PMID)")
        for pmid in cites:
            st.markdown(f"- [https://pubmed.ncbi.nlm.nih.gov/{pmid}/](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)")

    faith = result.get("faithfulness") or {}
    rate = faith.get("overall_rate", 0.0)
    st.subheader("Faithfulness (NLI)")
    st.metric("Supported sentence rate", f"{float(rate) * 100:.1f}%")
    for s in faith.get("sentences", []):
        label = "supported" if s.get("supported") else "low confidence"
        color = "green" if s.get("supported") else "orange"
        st.markdown(f":{color}[**{label}**] (max entailment `{s.get('max_score')}`) — {s.get('text', '')}")
