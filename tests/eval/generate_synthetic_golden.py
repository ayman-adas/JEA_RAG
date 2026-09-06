"""
Synthetic retrieval golden-set generator.
==========================================

Pulls chunks already indexed in ChromaDB, samples a handful per source
document, and asks the LLM to write one question each chunk directly
answers. The chunk itself is the known-correct retrieval target, so this
produces (query -> expected_chunk_id) pairs with zero manual labeling.

Usage:
  python -m tests.eval.generate_synthetic_golden
  python -m tests.eval.generate_synthetic_golden --samples-per-source 3 --max-total 200
"""

import sys
import json
import asyncio
import random
import argparse
import re
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "retrieval_golden.jsonl"
MIN_CHUNK_CHARS = 60
MAX_CONCURRENCY = 4

QUESTION_PROMPT = (
    "فيما يلي مقطع نصي من وثيقة رسمية لنقابة المهندسين الأردنيين.\n"
    "اكتب سؤالاً واحداً محدداً باللغة العربية تجيب عليه هذه الفقرة إجابة كاملة ومباشرة، "
    "بحيث يعتمد السؤال فقط على معلومات موجودة فعلاً في النص.\n"
    "أعد السؤال فقط بدون أي شرح أو مقدمة.\n\n"
    f"{'=' * 40}\n"
    "النص:\n{chunk_text}\n"
    f"{'=' * 40}\n"
    "السؤال:"
)


def clean_question(raw: str) -> str:
    q = raw.strip().strip('"').strip("'").strip()
    q = q.split("\n")[0].strip()
    q = re.sub(r'^(السؤال|Question)\s*:\s*', '', q, flags=re.IGNORECASE)
    return q.strip()


async def generate_question(generator: ResponseGeneratorAgent, chunk_text: str, sem: asyncio.Semaphore) -> str:
    prompt = QUESTION_PROMPT.format(chunk_text=chunk_text[:2000])
    async with sem:
        answer, ok = await generator.generate(prompt)
    return clean_question(answer) if ok and answer else ""


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-source", type=int, default=2,
                         help="How many chunks to sample per source document.")
    parser.add_argument("--max-total", type=int, default=150,
                         help="Cap on total questions generated (LLM call budget).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PERSIST_DIR),
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data.get("ids", [])
    docs = all_data.get("documents", [])
    metas = all_data.get("metadatas", [])

    if not ids:
        print(f"No chunks found in Chroma collection '{CHROMA_COLLECTION_NAME}'. Ingest documents first.")
        return

    by_source = {}
    for cid, text, meta in zip(ids, docs, metas):
        if not text or len(text.strip()) < MIN_CHUNK_CHARS:
            continue
        source_id = (meta or {}).get("source_id", "unknown")
        by_source.setdefault(source_id, []).append((cid, text))

    selected = []
    for source_id, chunks in by_source.items():
        random.shuffle(chunks)
        for cid, text in chunks[:args.samples_per_source]:
            selected.append((cid, text, source_id))

    random.shuffle(selected)
    selected = selected[:args.max_total]

    print(f"Sampled {len(selected)} chunks from {len(by_source)} source documents. Generating questions...")

    generator = ResponseGeneratorAgent()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def worker(item):
        cid, text, source_id = item
        question = await generate_question(generator, text, sem)
        return cid, source_id, question

    results = await asyncio.gather(*(worker(item) for item in selected))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for cid, source_id, question in results:
            if not question:
                continue
            row = {
                "query": question,
                "expected_chunk_id": cid,
                "expected_source_id": source_id,
                "origin": "synthetic"
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    skipped = len(results) - written
    print(f"Wrote {written} golden rows to {args.output} ({skipped} chunks skipped: empty/failed generation).")


if __name__ == "__main__":
    asyncio.run(main())
