"""
Retrieval quality evaluation: Recall@K and MRR against a labeled golden set.
==============================================================================

Runs the actual hybrid retriever (BM25 + BGE-M3 + RRF, no reranker/generation
involved) against each (query -> expected_chunk_id) pair in the golden set and
reports whether/where the correct chunk showed up.

Usage:
  python -m tests.eval.evaluate_retrieval
  python -m tests.eval.evaluate_retrieval --golden tests/eval/data/retrieval_golden.jsonl --top-k 10
"""

import sys
import json
import argparse
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "data" / "retrieval_golden.jsonl"


def load_golden(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--report", type=Path, default=None,
                         help="Optional path to write full per-query results as JSON.")
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}\nGenerate one first: python -m tests.eval.generate_synthetic_golden")
        return

    golden_rows = load_golden(args.golden)
    if not golden_rows:
        print(f"Golden file {args.golden} is empty.")
        return

    print(f"Loaded {len(golden_rows)} golden queries. Loading retriever (embedder + BM25 index)...")
    retriever = RetrieverAgent()

    results = []
    for row in golden_rows:
        query = row["query"]
        expected = row["expected_chunk_id"]
        candidates = retriever.retrieve_hybrid(query, top_k=args.top_k)
        chunk_ids = [c.get("chunk_id") for c in candidates]

        rank = None
        if expected in chunk_ids:
            rank = chunk_ids.index(expected) + 1

        results.append({
            "query": query,
            "expected_chunk_id": expected,
            "origin": row.get("origin", "unknown"),
            "found": rank is not None,
            "rank": rank
        })

    total = len(results)
    found_count = sum(1 for r in results if r["found"])
    recall_at_k = found_count / total if total else 0.0
    mrr = sum((1.0 / r["rank"]) if r["found"] else 0.0 for r in results) / total if total else 0.0

    print(f"\n{'Query':<60} {'Found':<7} {'Rank':<5} {'Origin'}")
    print("-" * 90)
    for r in results:
        q_display = (r["query"][:57] + "...") if len(r["query"]) > 60 else r["query"]
        print(f"{q_display:<60} {'yes' if r['found'] else 'NO':<7} {r['rank'] or '-':<5} {r['origin']}")

    print("-" * 90)
    print(f"Recall@{args.top_k}: {recall_at_k:.2%}  ({found_count}/{total})")
    print(f"MRR:        {mrr:.3f}")

    misses = [r for r in results if not r["found"]]
    if misses:
        print(f"\n{len(misses)} query(ies) failed to retrieve the expected chunk within top-{args.top_k}:")
        for r in misses:
            print(f"  - {r['query']}  (expected {r['expected_chunk_id']})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({
                "top_k": args.top_k,
                "recall_at_k": recall_at_k,
                "mrr": mrr,
                "total": total,
                "found": found_count,
                "results": results
            }, f, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    main()
