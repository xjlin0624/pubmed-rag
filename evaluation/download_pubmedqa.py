"""
Download PubMedQA and convert to BioASQ-compatible format for bioasq_eval.py.

PubMedQA (qiaojin/PubMedQA) questions are derived from PubMed abstracts,
making it a better fit than BioASQ for a PubMed-grounded RAG system.

Output schema matches bioasq_eval.py expectations:
    {
        "id": "<pubmed_id>",
        "body": "<question>",
        "ideal_answer": "<long_answer>",
        "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/<pubmed_id>"]
    }

Usage
-----
    # Full labeled set (1k expert-annotated)
    python evaluation/download_pubmedqa.py --out data/pubmedqa_gold.json

    # Diabetes-only subset
    python evaluation/download_pubmedqa.py --out data/pubmedqa_gold.json --diabetes-only

    # Also include unlabeled set (61k auto-generated, larger but noisier)
    python evaluation/download_pubmedqa.py --out data/pubmedqa_gold.json --include-unlabeled
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DIABETES_KEYWORDS = [
    "diabetes", "diabetic", "insulin", "metformin", "hba1c", "hemoglobin a1c",
    "hyperglycemi", "hypoglycemi", "glycemic", "glycaemic", "blood glucose",
    "blood sugar", "glucagon", "sglt2", "glp-1", "dpp-4", "insulin resistance",
    "beta cell", "islet", "type 2 diabetes", "type 1 diabetes",
]


def is_diabetes_related(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DIABETES_KEYWORDS)


def convert_record(record: dict) -> dict | None:
    """Convert a PubMedQA record to BioASQ-compatible format."""
    pmid = str(record.get("pubid", record.get("pubmed_id", ""))).strip()
    question = record.get("question", "").strip()
    long_answer = record.get("long_answer", "").strip()

    if not pmid or not question or not long_answer:
        return None

    return {
        "id": pmid,
        "body": question,
        "ideal_answer": long_answer,
        "final_decision": record.get("final_decision", ""),
        "documents": [f"http://www.ncbi.nlm.nih.gov/pubmed/{pmid}"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/pubmedqa_gold.json")
    parser.add_argument("--diabetes-only", action="store_true",
                        help="Only keep diabetes-related questions")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Also include pqa_unlabeled split (61k, noisier)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Install the datasets library first: pip install datasets")
        return

    splits = ["pqa_labeled"]
    if args.include_unlabeled:
        splits.append("pqa_unlabeled")

    questions = []
    for split in splits:
        print(f"Downloading PubMedQA split: {split}...")
        ds = load_dataset("qiaojin/PubMedQA", split, split="train")
        print(f"  {len(ds)} records")
        for record in ds:
            converted = convert_record(record)
            if converted is None:
                continue
            if args.diabetes_only and not is_diabetes_related(converted["body"]):
                continue
            questions.append(converted)

    print(f"\nTotal kept: {len(questions)} questions")
    if args.diabetes_only:
        print("(filtered to diabetes-related only)")

    Path(args.out).write_text(json.dumps({"questions": questions}, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
