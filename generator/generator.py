import os
import re
import requests
from transformers import pipeline

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.5"))
MIN_RETRIEVAL_SCORE = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.3"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))

nli = pipeline(
    "text-classification",
    model="cross-encoder/nli-MiniLM2-L6-H768",
    device=-1
)


def generate_answer(query: str, context: list[dict]) -> dict:
    """
    Generate a grounded answer from retrieved PubMed abstracts.

    Args:
        query:   The user's question.
        context: List of retrieval results from B's retrieve().
                 Each dict: {pmid, text, score, ...}

    Returns:
        {
            answer:    str,
            citations: list[str],   # PMIDs referenced
            fallback:  bool         # True if retrieval quality was too low
        }
    """
    if not context or context[0]["score"] < MIN_RETRIEVAL_SCORE:
        return {
            "answer": "Insufficient information found in retrieved sources to answer this question.",
            "citations": [],
            "fallback": True
        }

    context_text = "\n\n".join(
        f"[{i+1}] PMID {r['pmid']}: {r['text']}"
        for i, r in enumerate(context)
    )

    user_message = f"""Passages:
{context_text}

Question: {query}"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical literature assistant. "
                            "Write a concise 2-3 sentence answer using ONLY the passages given. "
                            "Cite with [1], [2], etc. Output the answer only — no reasoning, no analysis, no preamble."
                        ),
                    },
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": {
                    "num_predict": MAX_TOKENS,
                    "temperature": 0.1,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        answer = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    except requests.exceptions.Timeout:
        return {
            "answer": "Request timed out. The model took too long to respond.",
            "citations": [],
            "fallback": True
        }
    except Exception as e:
        return {
            "answer": f"Generation error: {str(e)}",
            "citations": [],
            "fallback": True
        }

    citations = [r["pmid"] for r in context]

    return {
        "answer": answer,
        "citations": citations,
        "fallback": False
    }


def split_sentences(text: str) -> list[str]:
    """Split answer text into individual sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def check_faithfulness(answer: str, context: list[dict]) -> dict:
    """
    Check whether each sentence in the answer is supported
    by at least one retrieved passage using NLI entailment.

    Args:
        answer:  The generated answer string.
        context: Same context list passed to generate_answer().

    Returns:
        {
            sentences:    list of {text, supported, max_score},
            overall_rate: float  # fraction of sentences that are supported
        }
    """
    sentences = split_sentences(answer)
    passages = [r["text"] for r in context]

    if not sentences:
        return {"sentences": [], "overall_rate": 0.0}

    threshold = float(os.getenv("FAITHFULNESS_THRESHOLD", str(FAITHFULNESS_THRESHOLD)))
    results = []
    for sent in sentences:
        scores = []
        for passage in passages:
            try:
                output = nli(
                    f"{passage} [SEP] {sent}",
                    truncation=True,
                    max_length=512,
                    top_k=None
                )
                entail_score = next(
                    (r["score"] for r in output if r["label"].upper() == "ENTAILMENT"),
                    0.0
                )
                scores.append(entail_score)
            except Exception:
                scores.append(0.0)

        max_score = max(scores) if scores else 0.0
        results.append({
            "text": sent,
            "supported": max_score >= threshold,
            "max_score": round(max_score, 3)
        })

    supported = sum(1 for r in results if r["supported"])
    overall_rate = round(supported / len(results), 3)

    return {
        "sentences": results,
        "overall_rate": overall_rate
    }


def run(query: str, context: list[dict]) -> dict:
    """
    Full pipeline: generate answer then check faithfulness.

    Returns:
        {
            answer:           str,
            citations:        list[str],
            fallback:         bool,
            faithfulness:     {sentences, overall_rate}
        }
    """
    gen = generate_answer(query, context)

    if gen["fallback"]:
        return {**gen, "faithfulness": {"sentences": [], "overall_rate": 0.0}}

    faith = check_faithfulness(gen["answer"], context)

    return {**gen, "faithfulness": faith}


if __name__ == "__main__":
    mock_context = [
        {
            "pmid": "12345678",
            "text": "Metformin is a first-line medication for type 2 diabetes. It works by decreasing hepatic glucose production and improving insulin sensitivity.",
            "score": 0.85
        },
        {
            "pmid": "87654321",
            "text": "Clinical trials show metformin reduces HbA1c levels by 1-2% on average. It is generally well tolerated with gastrointestinal side effects being most common.",
            "score": 0.76
        }
    ]

    result = run("What is the effect of metformin on blood sugar?", mock_context)

    print("Answer:", result["answer"])
    print("Citations:", result["citations"])
    print("Fallback:", result["fallback"])
    print("Faithfulness rate:", result["faithfulness"]["overall_rate"])
    print("\nPer-sentence breakdown:")
    for s in result["faithfulness"]["sentences"]:
        status = "supported" if s["supported"] else "UNSUPPORTED"
        print(f"  [{status}] ({s['max_score']}) {s['text']}")