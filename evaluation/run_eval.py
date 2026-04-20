"""
Run the RAG pipeline over BioASQ gold questions and save predictions.

Usage
-----
    # Full run (all questions)
    python evaluation/run_eval.py --gold data/bioasq_gold.json --out evaluation/predictions.json

    # Quick subset (first N questions)
    python evaluation/run_eval.py --gold data/bioasq_gold.json --out evaluation/predictions.json --limit 50

    # Then score:
    python evaluation/bioasq_eval.py --gold data/bioasq_gold.json --pred evaluation/predictions.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def run_pipeline_on_dataset(
    questions: list[dict],
    limit: int | None = None,
) -> list[dict]:
    import pipeline  # noqa: PLC0415 — lazy import so env is set before loading

    if limit:
        questions = questions[:limit]

    predictions = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid = q.get("id", "")
        body = q.get("body", "")
        logger.info("[%d/%d] %s", i, total, body[:80])

        t0 = time.time()
        try:
            result = pipeline.run(body)
        except Exception:
            logger.exception("Pipeline failed for id=%s; skipping.", qid)
            continue
        elapsed = time.time() - t0

        predictions.append(
            {
                "id": qid,
                "answer": result.get("answer", ""),
                "citations": result.get("citations", []),
                "retrieved_pmids": result.get("retrieved_pmids", result.get("citations", [])),
                "faithfulness": result.get("faithfulness", {}),
                "elapsed_s": round(elapsed, 2),
            }
        )

        if i % 10 == 0:
            logger.info("Progress: %d/%d done", i, total)

    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG predictions for BioASQ evaluation")
    parser.add_argument("--gold", required=True, help="Path to BioASQ gold JSON")
    parser.add_argument("--out", required=True, help="Where to write predictions JSON")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N questions")
    parser.add_argument("--resume", action="store_true", help="Skip questions already in --out file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    gold_data = json.loads(Path(args.gold).read_text())
    if isinstance(gold_data, dict) and "questions" in gold_data:
        questions = gold_data["questions"]
    else:
        questions = gold_data

    # Resume: skip already-predicted IDs
    existing: list[dict] = []
    if args.resume and Path(args.out).exists():
        existing = json.loads(Path(args.out).read_text())
        done_ids = {p["id"] for p in existing}
        questions = [q for q in questions if q.get("id") not in done_ids]
        logger.info("Resuming: %d already done, %d remaining", len(existing), len(questions))

    new_preds = run_pipeline_on_dataset(questions, limit=args.limit)
    all_preds = existing + new_preds

    Path(args.out).write_text(json.dumps(all_preds, indent=2))
    logger.info("Saved %d predictions to %s", len(all_preds), args.out)


if __name__ == "__main__":
    sys.exit(main())
