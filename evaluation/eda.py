"""
Exploratory Data Analysis (EDA) for PubMed RAG corpus.

Generates publication-ready charts:
1. Abstract / chunk length distribution (token & character counts)
2. Publication year distribution
3. Top MeSH term frequency
4. Corpus summary statistics

Usage
-----
    # Generate charts from the real corpus
    python evaluation/eda.py --chunks chunks.json --out evaluation/figures/

    # Smoke test with synthetic data
    python evaluation/eda.py --smoke --out evaluation/figures/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Chart generation (matplotlib)
# ---------------------------------------------------------------------------

def _ensure_matplotlib():
    """Import matplotlib with Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_length_distribution(chunks: list[dict], out_dir: Path):
    """Histogram of chunk lengths in characters and whitespace tokens."""
    plt = _ensure_matplotlib()

    texts = [c.get("text", "") for c in chunks]
    char_lens = [len(t) for t in texts]
    tok_lens = [len(t.split()) for t in texts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(char_lens, bins=40, color="#1565c0", edgecolor="white", alpha=0.85)
    axes[0].set_title("Chunk length (characters)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Characters")
    axes[0].set_ylabel("Count")
    axes[0].axvline(sum(char_lens) / len(char_lens), color="#e53935", ls="--", label=f"mean={sum(char_lens)//len(char_lens)}")
    axes[0].legend()

    axes[1].hist(tok_lens, bins=40, color="#2e7d32", edgecolor="white", alpha=0.85)
    axes[1].set_title("Chunk length (tokens)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Tokens (whitespace)")
    axes[1].set_ylabel("Count")
    axes[1].axvline(sum(tok_lens) / len(tok_lens), color="#e53935", ls="--", label=f"mean={sum(tok_lens)//len(tok_lens)}")
    axes[1].legend()

    fig.tight_layout()
    path = out_dir / "length_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_year_distribution(chunks: list[dict], out_dir: Path):
    """Bar chart of publication years."""
    plt = _ensure_matplotlib()

    years = [c.get("year") for c in chunks if c.get("year")]
    if not years:
        logger.warning("No year data found in chunks; skipping year chart.")
        return None

    # Convert to int, handle strings
    int_years = []
    for y in years:
        try:
            int_years.append(int(y))
        except (ValueError, TypeError):
            pass

    if not int_years:
        return None

    counter = Counter(int_years)
    sorted_years = sorted(counter.keys())
    counts = [counter[y] for y in sorted_years]

    fig, ax = plt.subplots(figsize=(max(8, len(sorted_years) * 0.4), 4.5))
    bars = ax.bar(
        [str(y) for y in sorted_years],
        counts,
        color="#7b1fa2",
        edgecolor="white",
        alpha=0.85,
    )
    ax.set_title("Publication year distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of chunks")

    # Rotate labels if many years
    if len(sorted_years) > 15:
        ax.tick_params(axis="x", rotation=45)

    # Annotate top bar
    max_count = max(counts)
    for bar, cnt in zip(bars, counts):
        if cnt == max_count:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_count * 0.02,
                str(cnt),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    path = out_dir / "year_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_mesh_distribution(chunks: list[dict], out_dir: Path, top_n: int = 25):
    """Horizontal bar chart of the top-N MeSH terms."""
    plt = _ensure_matplotlib()

    mesh_counter: Counter = Counter()
    for c in chunks:
        mesh = c.get("mesh") or c.get("mesh_terms") or c.get("MeSH") or []
        if isinstance(mesh, str):
            mesh = [m.strip() for m in mesh.split(";") if m.strip()]
        for term in mesh:
            mesh_counter[term] += 1

    if not mesh_counter:
        logger.warning("No MeSH terms found in chunks; skipping MeSH chart.")
        return None

    top = mesh_counter.most_common(top_n)
    terms = [t for t, _ in reversed(top)]
    counts = [c for _, c in reversed(top)]

    fig, ax = plt.subplots(figsize=(8, max(4, len(terms) * 0.35)))
    ax.barh(terms, counts, color="#ef6c00", edgecolor="white", alpha=0.85)
    ax.set_title(f"Top {top_n} MeSH terms", fontsize=12, fontweight="bold")
    ax.set_xlabel("Frequency")

    for i, (cnt, term) in enumerate(zip(counts, terms)):
        ax.text(cnt + max(counts) * 0.01, i, str(cnt), va="center", fontsize=8)

    fig.tight_layout()
    path = out_dir / "mesh_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def compute_summary_stats(chunks: list[dict]) -> dict[str, Any]:
    """Corpus-level statistics."""
    texts = [c.get("text", "") for c in chunks]
    char_lens = [len(t) for t in texts]
    tok_lens = [len(t.split()) for t in texts]
    pmids = set(c.get("pmid", "") for c in chunks if c.get("pmid"))
    years = [int(c["year"]) for c in chunks if c.get("year") and str(c["year"]).isdigit()]

    mesh_counter: Counter = Counter()
    for c in chunks:
        mesh = c.get("mesh") or c.get("mesh_terms") or c.get("MeSH") or []
        if isinstance(mesh, str):
            mesh = [m.strip() for m in mesh.split(";") if m.strip()]
        mesh_counter.update(mesh)

    stats = {
        "total_chunks": len(chunks),
        "unique_pmids": len(pmids),
        "char_length": {
            "mean": sum(char_lens) / len(char_lens) if char_lens else 0,
            "min": min(char_lens) if char_lens else 0,
            "max": max(char_lens) if char_lens else 0,
            "median": sorted(char_lens)[len(char_lens) // 2] if char_lens else 0,
        },
        "token_length": {
            "mean": sum(tok_lens) / len(tok_lens) if tok_lens else 0,
            "min": min(tok_lens) if tok_lens else 0,
            "max": max(tok_lens) if tok_lens else 0,
            "median": sorted(tok_lens)[len(tok_lens) // 2] if tok_lens else 0,
        },
        "year_range": f"{min(years)}–{max(years)}" if years else "N/A",
        "unique_mesh_terms": len(mesh_counter),
        "top_5_mesh": mesh_counter.most_common(5),
    }
    return stats


# ---------------------------------------------------------------------------
# Smoke test with synthetic data
# ---------------------------------------------------------------------------

_SAMPLE_MESH = [
    "Diabetes Mellitus, Type 2", "Metformin", "Blood Glucose", "Insulin Resistance",
    "Hemoglobin A, Glycosylated", "Randomized Controlled Trials", "Double-Blind Method",
    "Humans", "Adult", "Aged", "Female", "Male", "Hypertension", "Cardiovascular Diseases",
    "Body Mass Index", "Obesity", "Kidney Diseases", "Drug Therapy", "Liver",
    "Gluconeogenesis", "Cohort Studies", "Retrospective Studies",
]


def _generate_synthetic_chunks(n: int = 500) -> list[dict]:
    """Create synthetic chunks for testing when chunks.json is unavailable."""
    random.seed(42)
    chunks = []
    for i in range(n):
        pmid = str(random.randint(10000000, 99999999))
        year = random.choice(range(2005, 2025))
        length = random.randint(20, 300)
        text = " ".join(random.choices(
            "metformin glucose insulin diabetes treatment clinical patients study blood sugar".split(),
            k=length,
        ))
        mesh = random.sample(_SAMPLE_MESH, k=random.randint(2, 6))
        chunks.append({
            "pmid": pmid,
            "text": text,
            "year": year,
            "mesh": mesh,
            "score": round(random.uniform(0.3, 0.99), 3),
        })
    return chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def plot_question_type_distribution(gold_path: str, out_dir: Path):
    """Bar chart of PubMedQA question types (yes / no / maybe)."""
    plt = _ensure_matplotlib()

    gold_file = Path(gold_path)
    if not gold_file.exists():
        logger.warning("Gold file not found at %s; skipping question type chart.", gold_path)
        return None

    data = json.loads(gold_file.read_text())
    questions = data.get("questions", data) if isinstance(data, dict) else data

    counter: Counter = Counter()
    for q in questions:
        decision = q.get("final_decision", "").strip().lower()
        if decision:
            counter[decision] += 1

    if not counter:
        logger.warning("No final_decision field found in gold data; skipping question type chart.")
        return None

    labels = sorted(counter.keys())
    counts = [counter[l] for l in labels]
    colors = {"yes": "#2e7d32", "no": "#c62828", "maybe": "#f57f17"}
    bar_colors = [colors.get(l, "#1565c0") for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, counts, color=bar_colors, edgecolor="white", alpha=0.85)
    ax.set_title("PubMedQA question type distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Final decision")
    ax.set_ylabel("Count")

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.02,
                str(cnt), ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "question_type_distribution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def main():
    parser = argparse.ArgumentParser(description="EDA charts for PubMed RAG corpus")
    parser.add_argument("--chunks", type=str, default="chunks.json", help="Path to chunks.json")
    parser.add_argument("--gold", type=str, default="data/pubmedqa_gold.json", help="Path to PubMedQA gold JSON")
    parser.add_argument("--out", type=str, default="evaluation/figures", help="Output directory for charts")
    parser.add_argument("--top-mesh", type=int, default=25, help="Number of top MeSH terms to show")
    parser.add_argument("--smoke", action="store_true", help="Use synthetic data for testing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        logger.info("Using synthetic data (--smoke).")
        chunks = _generate_synthetic_chunks(500)
    else:
        chunks_path = Path(args.chunks)
        if not chunks_path.exists():
            logger.error("chunks.json not found at %s. Use --smoke for synthetic data.", chunks_path)
            return
        chunks = json.loads(chunks_path.read_text())

    logger.info("Loaded %d chunks.", len(chunks))

    # Summary
    stats = compute_summary_stats(chunks)
    print("\n" + "=" * 50)
    print("Corpus Summary")
    print("=" * 50)
    print(f"  Total chunks:      {stats['total_chunks']}")
    print(f"  Unique PMIDs:      {stats['unique_pmids']}")
    print(f"  Char length:       mean={stats['char_length']['mean']:.0f}, "
          f"median={stats['char_length']['median']}, "
          f"range=[{stats['char_length']['min']}, {stats['char_length']['max']}]")
    print(f"  Token length:      mean={stats['token_length']['mean']:.1f}, "
          f"median={stats['token_length']['median']}, "
          f"range=[{stats['token_length']['min']}, {stats['token_length']['max']}]")
    print(f"  Year range:        {stats['year_range']}")
    print(f"  Unique MeSH terms: {stats['unique_mesh_terms']}")
    print(f"  Top 5 MeSH:        {stats['top_5_mesh']}")
    print()

    # Save stats JSON
    stats_path = out_dir / "corpus_stats.json"
    # Convert top_5_mesh tuples to serializable format
    stats["top_5_mesh"] = [[t, c] for t, c in stats["top_5_mesh"]]
    stats_path.write_text(json.dumps(stats, indent=2))
    logger.info("Saved %s", stats_path)

    # Charts
    plot_length_distribution(chunks, out_dir)
    plot_year_distribution(chunks, out_dir)
    plot_mesh_distribution(chunks, out_dir, top_n=args.top_mesh)
    plot_question_type_distribution(args.gold, out_dir)

    print(f"All charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
