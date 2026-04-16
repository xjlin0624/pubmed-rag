import os
from generator import run as generate

TOP_K = int(os.getenv("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Retriever interface (stub — replace with B's implementation)
# Expected signature: retrieve(query: str, top_k: int) -> list[dict]
# Each result: {"pmid": str, "passage": str, "score": float}
# ---------------------------------------------------------------------------

def _mock_retrieve(query: str, top_k: int) -> list[dict]:
    """Stub retriever for local testing. Replace with retriever.retrieve()."""
    return [
        {
            "pmid": "12345678",
            "passage": "Metformin is a first-line medication for type 2 diabetes. "
                       "It works by decreasing hepatic glucose production and improving insulin sensitivity.",
            "score": 0.85
        },
        {
            "pmid": "87654321",
            "passage": "Clinical trials show metformin reduces HbA1c levels by 1-2% on average. "
                       "It is generally well tolerated with gastrointestinal side effects being most common.",
            "score": 0.76
        }
    ][:top_k]


try:
    from retriever.retriever import retrieve as _retrieve
except ImportError:
    _retrieve = None


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if _retrieve is not None:
        return _retrieve(query, top_k)
    return _mock_retrieve(query, top_k)


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(query: str) -> dict:
    """
    End-to-end pipeline: retrieve → generate → faithfulness check.

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
    context = retrieve(query, top_k=TOP_K)
    result = generate(query, context)
    return {"query": query, **result}


if __name__ == "__main__":
    import json
    query = "What is the effect of metformin on blood sugar?"
    result = run(query)

    print(f"Query: {result['query']}\n")
    print(f"Answer: {result['answer']}\n")
    print(f"Citations: {result['citations']}")
    print(f"Fallback: {result['fallback']}")
    print(f"Faithfulness rate: {result['faithfulness']['overall_rate']}\n")
    print("Per-sentence breakdown:")
    for s in result["faithfulness"]["sentences"]:
        status = "supported" if s["supported"] else "UNSUPPORTED"
        print(f"  [{status}] ({s['max_score']}) {s['text']}")
