"""
Lightweight ablation runner for integration (Part A).

Runs the same smoke question under four configs by varying `BM25_ALPHA`
(BM25-only, dense-only, hybrid) plus a stricter `FAITHFULNESS_THRESHOLD`.

If retrieval indexes are missing, `pipeline` uses the mock retriever; BM25_ALPHA
ablations still run for wiring validation, though scores will not differ.

Usage (from repository root):

    python evaluation/ablation_smoke.py
    python evaluation/ablation_smoke.py --query "insulin resistance mechanisms"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _run_once(name: str, env_updates: dict, query: str) -> dict:
    prev: dict[str, str | None] = {}
    for k, v in env_updates.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)

    import pipeline as pl

    try:
        out = pl.run(query)
    finally:
        for k, old in prev.items():
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old

    faith = out.get("faithfulness") or {}
    return {
        "branch": name,
        "fallback": out.get("fallback"),
        "faithfulness_rate": faith.get("overall_rate"),
        "answer_preview": (out.get("answer") or "")[:160].replace("\n", " "),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke ablation matrix (fusion + NLI strictness).")
    parser.add_argument(
        "--query",
        default=os.getenv("ABLATION_QUERY", "What is the effect of metformin on blood sugar?"),
        help="Single QA query to repeat under each branch.",
    )
    parser.add_argument(
        "--out",
        default="evaluation/ablation_smoke_results.json",
        help="Where to write JSON results (relative to current working directory).",
    )
    args = parser.parse_args()

    base_nli = os.getenv("FAITHFULNESS_THRESHOLD", "0.5")
    branches = [
        ("bm25_only", {"BM25_ALPHA": "1.0", "FAITHFULNESS_THRESHOLD": base_nli}),
        ("dense_only", {"BM25_ALPHA": "0.0", "FAITHFULNESS_THRESHOLD": base_nli}),
        ("hybrid_default", {"BM25_ALPHA": "0.5", "FAITHFULNESS_THRESHOLD": base_nli}),
        ("hybrid_strict_nli", {"BM25_ALPHA": "0.5", "FAITHFULNESS_THRESHOLD": os.getenv("ABLATION_STRICT_NLI", "0.75")}),
    ]

    rows = [_run_once(name, env, args.query) for name, env in branches]

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "runs": rows}, f, indent=2)

    print(f"Wrote {args.out}\n")
    print("| Branch | Fallback | Faithfulness rate | Answer preview |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        prev = r["answer_preview"].replace("|", "\\|")
        print(f"| {r['branch']} | {r['fallback']} | {r['faithfulness_rate']} | {prev} |")


if __name__ == "__main__":
    main()
