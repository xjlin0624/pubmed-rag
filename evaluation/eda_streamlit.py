"""
Streamlit EDA dashboard for PubMed RAG corpus.

Run standalone:
    streamlit run evaluation/eda_streamlit.py

Or import and call `render_eda_tab()` from the main app.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st


def _load_chunks(path: str = "chunks.json") -> list[dict]:
    """Load chunks from JSON; return empty list if not found."""
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _synthetic_chunks(n: int = 500) -> list[dict]:
    """Generate synthetic chunks for demo purposes."""
    import random
    random.seed(42)
    mesh_pool = [
        "Diabetes Mellitus, Type 2", "Metformin", "Blood Glucose",
        "Insulin Resistance", "Hemoglobin A, Glycosylated",
        "Randomized Controlled Trials", "Humans", "Adult", "Aged",
        "Female", "Male", "Hypertension", "Cardiovascular Diseases",
        "Body Mass Index", "Obesity", "Kidney Diseases", "Drug Therapy",
        "Liver", "Gluconeogenesis", "Cohort Studies",
    ]
    word_pool = (
        "metformin glucose insulin diabetes treatment clinical patients study "
        "blood sugar therapy dose efficacy safety trial placebo group outcome"
    ).split()
    chunks = []
    for _ in range(n):
        length = random.randint(20, 300)
        chunks.append({
            "pmid": str(random.randint(10000000, 99999999)),
            "text": " ".join(random.choices(word_pool, k=length)),
            "year": random.choice(range(2005, 2025)),
            "mesh": random.sample(mesh_pool, k=random.randint(2, 6)),
        })
    return chunks


def render_eda_tab(chunks: list[dict] | None = None):
    """
    Render the full EDA dashboard.

    Parameters
    ----------
    chunks : list[dict] | None
        Pre-loaded chunks. If None, loads from CHUNKS_FILE env var or falls
        back to synthetic data.
    """
    st.header("📊 Corpus EDA")

    if chunks is None:
        chunks = _load_chunks(os.getenv("CHUNKS_FILE", "chunks.json"))

    use_synthetic = False
    if not chunks:
        st.info("No `chunks.json` found — showing synthetic demo data.")
        chunks = _synthetic_chunks()
        use_synthetic = True

    # --- Summary metrics row ---
    texts = [c.get("text", "") for c in chunks]
    char_lens = [len(t) for t in texts]
    tok_lens = [len(t.split()) for t in texts]
    pmids = set(c.get("pmid", "") for c in chunks if c.get("pmid"))
    years = []
    for c in chunks:
        y = c.get("year")
        if y:
            try:
                years.append(int(y))
            except (ValueError, TypeError):
                pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Chunks", f"{len(chunks):,}")
    c2.metric("Unique PMIDs", f"{len(pmids):,}")
    c3.metric("Avg tokens/chunk", f"{sum(tok_lens)/len(tok_lens):.0f}" if tok_lens else "—")
    c4.metric("Year range", f"{min(years)}–{max(years)}" if years else "—")

    st.divider()

    # --- Length distribution ---
    st.subheader("Chunk length distribution")
    len_col1, len_col2 = st.columns(2)

    with len_col1:
        st.markdown("**Character count**")
        _render_histogram(char_lens, "Characters", "#1565c0")

    with len_col2:
        st.markdown("**Token count (whitespace)**")
        _render_histogram(tok_lens, "Tokens", "#2e7d32")

    # --- Year distribution ---
    if years:
        st.subheader("Publication year distribution")
        year_counter = Counter(years)
        sorted_years = sorted(year_counter.keys())
        year_data = [{"Year": str(y), "Count": year_counter[y]} for y in sorted_years]
        st.bar_chart(
            data={d["Year"]: d["Count"] for d in year_data},
            color="#7b1fa2",
        )

    # --- MeSH terms ---
    mesh_counter: Counter = Counter()
    for c in chunks:
        mesh = c.get("mesh") or c.get("mesh_terms") or c.get("MeSH") or []
        if isinstance(mesh, str):
            mesh = [m.strip() for m in mesh.split(";") if m.strip()]
        mesh_counter.update(mesh)

    if mesh_counter:
        st.subheader("Top MeSH terms")
        top_n = st.slider("Number of terms", 10, 50, 25, key="mesh_slider")
        top_mesh = mesh_counter.most_common(top_n)
        # Display as horizontal bar chart using st.bar_chart
        mesh_data = {term: count for term, count in reversed(top_mesh)}
        st.bar_chart(mesh_data, color="#ef6c00", horizontal=True)

    # --- Raw stats ---
    with st.expander("📋 Raw corpus statistics"):
        stats = {
            "total_chunks": len(chunks),
            "unique_pmids": len(pmids),
            "avg_char_length": round(sum(char_lens) / len(char_lens), 1) if char_lens else 0,
            "avg_token_length": round(sum(tok_lens) / len(tok_lens), 1) if tok_lens else 0,
            "year_range": f"{min(years)}–{max(years)}" if years else "N/A",
            "unique_mesh_terms": len(mesh_counter),
            "top_10_mesh": mesh_counter.most_common(10),
            "synthetic_data": use_synthetic,
        }
        st.json(stats)


def _render_histogram(values: list[int], xlabel: str, color: str):
    """Simple histogram using Streamlit's native bar_chart."""
    if not values:
        st.write("No data.")
        return

    # Bin the values
    import math
    n_bins = 30
    mn, mx = min(values), max(values)
    if mn == mx:
        st.metric(xlabel, f"{mn}")
        return
    bin_width = math.ceil((mx - mn) / n_bins) or 1
    bins: Counter = Counter()
    for v in values:
        b = ((v - mn) // bin_width) * bin_width + mn
        bins[b] += 1

    sorted_bins = sorted(bins.keys())
    chart_data = {f"{b}": bins[b] for b in sorted_bins}
    st.bar_chart(chart_data, color=color)

    # Quick stats
    median = sorted(values)[len(values) // 2]
    st.caption(
        f"min={min(values)}, max={max(values)}, "
        f"mean={sum(values)/len(values):.0f}, median={median}"
    )


# ---------------------------------------------------------------------------
# Standalone mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    st.set_page_config(page_title="PubMed RAG — EDA", layout="wide", page_icon="📊")
    render_eda_tab()
