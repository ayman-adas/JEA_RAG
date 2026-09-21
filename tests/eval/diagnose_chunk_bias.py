"""
Diagnostic: why does retrieval favor a document's first chunk (_c1)?
=======================================================================

Read-only investigation script. Does NOT modify the retrieval pipeline --
it only calls existing components (BM25Indexer, BGEM3Embedder, RetrieverAgent)
to break the hybrid retrieval score apart into its two channels for inspection.

For each golden query that the real hybrid retriever misses, this:
  1. Re-runs BM25 alone and dense-embedding search alone (bypassing RRF fusion
     and the priority reranker) to see which channel(s) actually placed the
     document's own first chunk (_c1) above the expected chunk.
  2. Prints both chunks' raw text so lexical/semantic overlap with the query
     can be judged by eye.

Usage:
  python -m tests.eval.diagnose_chunk_bias
  python -m tests.eval.diagnose_chunk_bias --golden tests/eval/data/retrieval_golden.jsonl --limit 15
"""

import sys
import re
import json
import argparse
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.pipeline.stage_02_retrieve.bm25_search import BM25Indexer
from src.kb_ingestor.ingestion import BGEM3Embedder

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "data" / "retrieval_golden.jsonl"
TOP_K = 20
CHUNK_SUFFIX_RE = re.compile(r'^(.*)_c(\d+)$')


def load_golden(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rank_of(target_id: str, ranked_ids: list) -> "int | None":
    return (ranked_ids.index(target_id) + 1) if target_id in ranked_ids else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=15, help="Max number of misses to diagnose.")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--show-text", type=int, default=5, help="How many misses to print full chunk text for.")
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}")
        return

    golden_rows = load_golden(args.golden)
    print(f"Loaded {len(golden_rows)} golden queries. Loading retriever (embedder + BM25 + Chroma)...")

    retriever = RetrieverAgent()
    bm25 = BM25Indexer()
    embedder = BGEM3Embedder()
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR), settings=ChromaSettings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    misses = []
    for row in golden_rows:
        query = row["query"]
        expected = row["expected_chunk_id"]
        hybrid_candidates = retriever.retrieve_hybrid(query, top_k=args.top_k)
        hybrid_ids = [c.get("chunk_id") for c in hybrid_candidates]
        if expected not in hybrid_ids:
            misses.append(row)
        if len(misses) >= args.limit:
            break

    print(f"Diagnosing {len(misses)} misses (capped at --limit {args.limit})...\n")

    header = f"{'Query':<45} {'BM25:exp':<9} {'BM25:c1':<8} {'Dense:exp':<10} {'Dense:c1':<9} {'sibling_c1_id'}"
    print(header)
    print("-" * len(header))

    detailed = []
    for row in misses:
        query = row["query"]
        expected = row["expected_chunk_id"]

        m = CHUNK_SUFFIX_RE.match(expected)
        source_id = m.group(1) if m else None
        sibling_c1 = f"{source_id}_c1" if source_id else None

        # BM25 alone
        bm25_results = bm25.search(query, top_k=args.top_k)
        bm25_ids = [chunk.get("chunk_id") for chunk, _score in bm25_results]
        bm25_rank_expected = rank_of(expected, bm25_ids)
        bm25_rank_c1 = rank_of(sibling_c1, bm25_ids) if sibling_c1 else None

        # Dense alone
        dense_vecs, _ = embedder.embed_texts([query])
        qv = dense_vecs[0] if dense_vecs else None
        dense_ids = []
        if qv is not None:
            dense_results = collection.query(query_embeddings=[qv], n_results=args.top_k, include=[])
            dense_ids = dense_results.get("ids", [[]])[0]
        dense_rank_expected = rank_of(expected, dense_ids)
        dense_rank_c1 = rank_of(sibling_c1, dense_ids) if sibling_c1 else None

        q_display = (query[:42] + "...") if len(query) > 45 else query
        print(f"{q_display:<45} {str(bm25_rank_expected or '-'):<9} {str(bm25_rank_c1 or '-'):<8} "
              f"{str(dense_rank_expected or '-'):<10} {str(dense_rank_c1 or '-'):<9} {sibling_c1 or '-'}")

        detailed.append({
            "query": query,
            "expected_chunk_id": expected,
            "sibling_c1_id": sibling_c1,
            "bm25_rank_expected": bm25_rank_expected,
            "bm25_rank_c1": bm25_rank_c1,
            "dense_rank_expected": dense_rank_expected,
            "dense_rank_c1": dense_rank_c1,
        })

    # Aggregate: how often does the c1 sibling out-rank the expected chunk in each channel?
    def c1_wins(d, channel):
        exp_r = d[f"{channel}_rank_expected"]
        c1_r = d[f"{channel}_rank_c1"]
        if c1_r is None:
            return False
        if exp_r is None:
            return True
        return c1_r < exp_r

    bm25_c1_wins = sum(1 for d in detailed if c1_wins(d, "bm25"))
    dense_c1_wins = sum(1 for d in detailed if c1_wins(d, "dense"))
    total = len(detailed)

    print("-" * len(header))
    print(f"c1 sibling out-ranks expected chunk in BM25 alone:  {bm25_c1_wins}/{total}")
    print(f"c1 sibling out-ranks expected chunk in Dense alone: {dense_c1_wins}/{total}")

    # Print full text for a handful so the overlap can be judged by eye
    print(f"\n{'=' * 80}\nFull text comparison for first {min(args.show_text, len(detailed))} misses:\n{'=' * 80}")
    for d in detailed[:args.show_text]:
        got = collection.get(ids=[cid for cid in [d["expected_chunk_id"], d["sibling_c1_id"]] if cid],
                              include=["documents"])
        text_by_id = dict(zip(got.get("ids", []), got.get("documents", [])))

        print(f"\nQUERY: {d['query']}")
        print(f"  [expected: {d['expected_chunk_id']}] (BM25 rank={d['bm25_rank_expected']}, Dense rank={d['dense_rank_expected']})")
        print(f"  {text_by_id.get(d['expected_chunk_id'], '<not found in collection>')[:400]}")
        print(f"  [sibling  : {d['sibling_c1_id']}] (BM25 rank={d['bm25_rank_c1']}, Dense rank={d['dense_rank_c1']})")
        print(f"  {text_by_id.get(d['sibling_c1_id'], '<not found in collection>')[:400]}")


if __name__ == "__main__":
    main()
