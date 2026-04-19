"""
End-to-end RAG pipeline (Part A — integration layer).

retrieve(query, top_k)  →  generator.run (generate_answer + check_faithfulness)
"""
from __future__ import annotations

import logging
import os

from generator import run as generate

logger = logging.getLogger(__name__)

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_ITER = int(os.getenv("MAX_ITER", "2"))
MAX_CONTEXT = int(os.getenv("MAX_CONTEXT", "15"))
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))

# ---------------------------------------------------------------------------
# Retriever: Part B `retrieve()` when indexes exist; lazy-import to avoid
# loading FAISS / sentence-transformers when using mock-only mode.
# Signature: retrieve(query: str, top_k: int) -> list[dict]
# ---------------------------------------------------------------------------

_NOT_LOADED = object()
_real_retrieve: object | None = _NOT_LOADED
_real_retrieve_error: str | None = None


def _try_load_retriever():
    global _real_retrieve, _real_retrieve_error
    if _real_retrieve is not _NOT_LOADED:
        return
    try:
        from retriever.retriever import retrieve as fn

        _real_retrieve = fn
        _real_retrieve_error = None
    except Exception as exc:  # pragma: no cover - optional heavy deps
        _real_retrieve = None
        _real_retrieve_error = str(exc)


def _mock_retrieve(query: str, top_k: int) -> list[dict]:
    """Fallback retriever for demos before indexes are built."""
    _ = query
    return [
        {
            "pmid": "12345678",
            "text": (
                "Metformin is a first-line medication for type 2 diabetes. "
                "It works by decreasing hepatic glucose production and improving insulin sensitivity."
            ),
            "score": 0.85,
        },
        {
            "pmid": "87654321",
            "text": (
                "Clinical trials show metformin reduces HbA1c levels by 1-2% on average. "
                "It is generally well tolerated with gastrointestinal side effects being most common."
            ),
            "score": 0.76,
        },
    ][:top_k]


def _index_files_present() -> bool:
    chunks = os.getenv("CHUNKS_FILE", "chunks.json")
    index_path = os.getenv("INDEX_FILE", "faiss.index")
    return os.path.isfile(chunks) and os.path.isfile(index_path)


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if os.getenv("USE_MOCK_RETRIEVER", "").strip().lower() in ("1", "true", "yes"):
        logger.info("Using mock retriever (USE_MOCK_RETRIEVER).")
        return _mock_retrieve(query, top_k)

    if not _index_files_present():
        logger.warning(
            "Retriever indexes not found (need %s + %s); using mock retriever.",
            os.getenv("CHUNKS_FILE", "chunks.json"),
            os.getenv("INDEX_FILE", "faiss.index"),
        )
        return _mock_retrieve(query, top_k)

    _try_load_retriever()
    if _real_retrieve is None:
        logger.warning(
            "Retriever import failed (%s); using mock retriever.",
            _real_retrieve_error or "unknown",
        )
        return _mock_retrieve(query, top_k)

    try:
        return _real_retrieve(query, top_k)  # type: ignore[misc]
    except Exception:
        logger.exception("retrieve() failed; falling back to mock retriever.")
        return _mock_retrieve(query, top_k)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run(query: str) -> dict:
    """
    End-to-end pipeline: retrieve → generate → faithfulness check,
    with iterative re-retrieval on unsupported sentences (up to MAX_ITER rounds).

    Args:
        query: User's natural language question.

    Returns:
        {
            query:        str,
            answer:       str,
            citations:    list[str],   # PMIDs
            fallback:     bool,
            faithfulness: {sentences, overall_rate}
        }
    """
    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", str(FAITHFULNESS_THRESHOLD)))
    max_iter = int(os.getenv("MAX_ITER", str(MAX_ITER)))

    context = retrieve(query, top_k=TOP_K)
    result = generate(query, context)

    for _ in range(max_iter):
        if result.get("fallback"):
            break
        faith = result.get("faithfulness", {})
        if faith.get("overall_rate", 1.0) >= threshold:
            break

        unsupported = [
            s["text"] for s in faith.get("sentences", [])
            if not s["supported"]
        ]
        if not unsupported:
            break

        max_context = int(os.getenv("MAX_CONTEXT", str(MAX_CONTEXT)))
        seen_pmids = {c["pmid"] for c in context}
        for sub_query in unsupported:
            if len(context) >= max_context:
                break
            for chunk in retrieve(sub_query, top_k=3):
                if chunk["pmid"] not in seen_pmids and len(context) < max_context:
                    context.append(chunk)
                    seen_pmids.add(chunk["pmid"])

        result = generate(query, context)

    return {"query": query, **result}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    q = "What is the effect of metformin on blood sugar?"
    out = run(q)

    print(f"Query: {out['query']}\n")
    print(f"Answer: {out['answer']}\n")
    print(f"Citations: {out['citations']}")
    print(f"Fallback: {out['fallback']}")
    print(f"Faithfulness rate: {out['faithfulness']['overall_rate']}\n")
    print("Per-sentence breakdown:")
    for s in out["faithfulness"]["sentences"]:
        status = "supported" if s["supported"] else "UNSUPPORTED"
        print(f"  [{status}] ({s['max_score']}) {s['text']}")
