"""
BioASQ-style evaluation metrics for PubMed RAG.

Metrics
-------
- **Exact Match (EM)**: Strict string match after normalization.
- **Token-level F1**: Precision/recall over whitespace tokens.
- **Recall@k**: Fraction of gold PMIDs found in top-k retrieved results.
- **Faithfulness rate**: Average NLI-supported sentence ratio across queries.

Usage
-----
    # Against a BioASQ-format JSON file
    python evaluation/bioasq_eval.py --gold data/bioasq_gold.json --pred evaluation/predictions.json

    # Quick smoke test with built-in examples
    python evaluation/bioasq_eval.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import string
import sys
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ───────────────────────────── text normalisation ─────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation & articles, collapse whitespace."""
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ──────────────────────────── answer-level metrics ────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    """1.0 if normalised strings are identical, else 0.0."""
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> dict[str, float]:
    """Token-level precision, recall, F1."""
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)

    if not gold_toks and not pred_toks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_toks or not pred_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_toks) & Counter(gold_toks)
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_toks)
    recall = num_common / len(gold_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


# ─────────────────────────── retrieval-level metrics ──────────────────────────

def recall_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int | None = None) -> float:
    """
    Fraction of gold PMIDs present in the (optionally truncated) retrieved list.

    Parameters
    ----------
    retrieved_pmids : list[str]
        PMIDs returned by the retriever, in ranked order.
    gold_pmids : list[str]
        Ground-truth PMIDs for this query.
    k : int | None
        If given, only the top-k retrieved PMIDs are considered.
    """
    if not gold_pmids:
        return 1.0  # vacuously true
    top = set(retrieved_pmids[:k] if k else retrieved_pmids)
    gold_set = set(gold_pmids)
    return len(top & gold_set) / len(gold_set)


def precision_at_k(retrieved_pmids: list[str], gold_pmids: list[str], k: int | None = None) -> float:
    """Fraction of top-k retrieved PMIDs that are in the gold set."""
    if not retrieved_pmids:
        return 0.0
    top = retrieved_pmids[:k] if k else retrieved_pmids
    if not top:
        return 0.0
    gold_set = set(gold_pmids)
    return sum(1 for p in top if p in gold_set) / len(top)


def mean_reciprocal_rank(retrieved_pmids: list[str], gold_pmids: list[str]) -> float:
    """MRR: reciprocal rank of the first relevant result."""
    gold_set = set(gold_pmids)
    for i, pmid in enumerate(retrieved_pmids, 1):
        if pmid in gold_set:
            return 1.0 / i
    return 0.0


# ─────────────────────────── faithfulness metric ──────────────────────────────

def faithfulness_rate(pipeline_result: dict) -> float:
    """
    Extract the NLI-based faithfulness rate from a single pipeline result.

    Expects: pipeline_result["faithfulness"]["overall_rate"] ∈ [0, 1].
    """
    faith = pipeline_result.get("faithfulness") or {}
    return float(faith.get("overall_rate", 0.0))


# ─────────────────────────── batch evaluation ─────────────────────────────────

def evaluate_dataset(
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate a list of predictions against gold annotations.

    Expected schemas
    ~~~~~~~~~~~~~~~~
    gold item:
        {
            "id": "...",
            "ideal_answer": "...",
            "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/12345678", ...]
        }
    prediction item:
        {
            "id": "...",
            "answer": "...",
            "citations": ["12345678", ...],
            "retrieved_pmids": ["12345678", ...],   # full ranked list
            "faithfulness": {"overall_rate": 0.85, ...}
        }

    Returns
    -------
    dict with averaged metrics.
    """
    if ks is None:
        ks = [1, 3, 5, 10]

    # Build gold lookup
    gold_map: dict[str, dict] = {}
    for g in gold:
        qid = g.get("id", g.get("body", ""))
        gold_map[qid] = g

    metrics: dict[str, list[float]] = {
        "exact_match": [],
        "f1": [],
        "precision": [],
        "recall": [],
        "faithfulness": [],
        "mrr": [],
    }
    for k in ks:
        metrics[f"recall@{k}"] = []
        metrics[f"precision@{k}"] = []

    matched = 0
    for pred in predictions:
        qid = pred.get("id", "")
        if qid not in gold_map:
            logger.warning("Prediction id=%s has no gold entry; skipping.", qid)
            continue
        matched += 1
        g = gold_map[qid]

        # --- Answer metrics ---
        gold_answer = g.get("ideal_answer", "")
        if isinstance(gold_answer, list):
            gold_answer = gold_answer[0] if gold_answer else ""
        pred_answer = pred.get("answer", "")

        metrics["exact_match"].append(exact_match(pred_answer, gold_answer))
        tf1 = token_f1(pred_answer, gold_answer)
        metrics["f1"].append(tf1["f1"])
        metrics["precision"].append(tf1["precision"])
        metrics["recall"].append(tf1["recall"])

        # --- Retrieval metrics ---
        gold_docs = g.get("documents", [])
        # Extract PMID from full PubMed URLs
        gold_pmids = []
        for d in gold_docs:
            m = re.search(r"(\d{5,})", str(d))
            if m:
                gold_pmids.append(m.group(1))

        retrieved = pred.get("retrieved_pmids", pred.get("citations", []))
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(retrieved, gold_pmids, k))
            metrics[f"precision@{k}"].append(precision_at_k(retrieved, gold_pmids, k))
        metrics["mrr"].append(mean_reciprocal_rank(retrieved, gold_pmids))

        # --- Faithfulness ---
        metrics["faithfulness"].append(faithfulness_rate(pred))

    if matched == 0:
        logger.error("No predictions matched gold IDs.")
        return {}

    # Average
    result = {}
    for name, vals in metrics.items():
        result[name] = sum(vals) / len(vals) if vals else 0.0

    result["num_evaluated"] = matched
    return result


# ─────────────────────────── smoke test ───────────────────────────────────────

def _smoke_test():
    """Self-contained example to verify the script works."""
    print("=" * 60)
    print("BioASQ Evaluation — Smoke Test")
    print("=" * 60)

    gold = [
        {
            "id": "q1",
            "ideal_answer": "Metformin reduces blood sugar by decreasing hepatic glucose production.",
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
                "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
            ],
        },
        {
            "id": "q2",
            "ideal_answer": "Aspirin inhibits platelet aggregation through COX-1 inhibition.",
            "documents": [
                "http://www.ncbi.nlm.nih.gov/pubmed/11111111",
            ],
        },
    ]

    preds = [
        {
            "id": "q1",
            "answer": "Metformin lowers blood sugar by reducing hepatic glucose production and improving insulin sensitivity.",
            "retrieved_pmids": ["12345678", "99999999", "87654321", "00000001", "00000002"],
            "citations": ["12345678", "87654321"],
            "faithfulness": {"overall_rate": 0.85, "sentences": []},
        },
        {
            "id": "q2",
            "answer": "Aspirin prevents blood clots by inhibiting COX-1 enzyme in platelets.",
            "retrieved_pmids": ["22222222", "11111111", "33333333"],
            "citations": ["11111111"],
            "faithfulness": {"overall_rate": 1.0, "sentences": []},
        },
    ]

    results = evaluate_dataset(preds, gold, ks=[1, 3, 5])

    print(f"\nEvaluated {results.get('num_evaluated', 0)} questions\n")
    print(f"  Exact Match:     {results['exact_match']:.4f}")
    print(f"  Token F1:        {results['f1']:.4f}")
    print(f"  Token Precision: {results['precision']:.4f}")
    print(f"  Token Recall:    {results['recall']:.4f}")
    print(f"  MRR:             {results['mrr']:.4f}")
    print()
    for k in [1, 3, 5]:
        print(f"  Recall@{k}:       {results[f'recall@{k}']:.4f}")
        print(f"  Precision@{k}:    {results[f'precision@{k}']:.4f}")
    print()
    print(f"  Faithfulness:    {results['faithfulness']:.4f}")
    print()
    print("Smoke test passed ✓")
    return results


# ─────────────────────────── CLI ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BioASQ-style evaluation for PubMed RAG")
    parser.add_argument("--gold", type=str, help="Path to gold annotations JSON (BioASQ format)")
    parser.add_argument("--pred", type=str, help="Path to predictions JSON")
    parser.add_argument("--ks", type=str, default="1,3,5,10", help="Comma-separated k values for Recall@k")
    parser.add_argument("--output", type=str, default=None, help="Write results JSON to this path")
    parser.add_argument("--smoke", action="store_true", help="Run built-in smoke test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.smoke:
        _smoke_test()
        return

    if not args.gold or not args.pred:
        parser.error("--gold and --pred are required (or use --smoke)")

    gold = json.loads(Path(args.gold).read_text())
    preds = json.loads(Path(args.pred).read_text())

    # BioASQ nests questions under a "questions" key
    if isinstance(gold, dict) and "questions" in gold:
        gold = gold["questions"]

    ks = [int(k.strip()) for k in args.ks.split(",")]
    results = evaluate_dataset(preds, gold, ks=ks)

    if not results:
        print("No matched predictions. Check that 'id' fields align between gold and pred.")
        sys.exit(1)

    print(json.dumps(results, indent=2))

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
