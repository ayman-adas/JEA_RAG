"""
Answer quality evaluation: faithfulness & relevancy via LLM-as-judge.
=======================================================================

Runs each golden query through the real pipeline (retrieve -> rerank ->
build_prompt -> generate) and then asks the LLM to judge the resulting
answer against the sources actually used:
  - faithfulness: is every claim in the answer supported by the sources?
  - relevancy:    does the answer actually address the question?

No hand-written reference answers are required -- the judge only needs the
(query, sources, answer) triple, which every pipeline run already produces.

Usage:
  python -m tests.eval.evaluate_answers
  python -m tests.eval.evaluate_answers --golden tests/eval/data/retrieval_golden.jsonl --top-k 15
"""

import sys
import json
import re
import argparse
import asyncio
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent

DEFAULT_GOLDEN = Path(__file__).resolve().parent / "data" / "retrieval_golden.jsonl"
MAX_CONCURRENCY = 4

JUDGE_PROMPT = """أنت مقيّم صارم لجودة إجابات نظام استرجاع معلومات (RAG). لديك سؤال، والمصادر التي استخدمها النظام، والإجابة الناتجة.

قيّم الإجابة وفق معيارين، كل منهما درجة من 0.0 إلى 1.0:
1. faithfulness: هل كل ادعاء في الإجابة مدعوم فعلاً بنص المصادر؟ (1.0 = كل شيء مدعوم، 0.0 = اختلاق كامل لمعلومات غير موجودة في المصادر)
2. relevancy: هل تجيب الإجابة فعلاً على سؤال المستخدم؟ (1.0 = إجابة مباشرة وكاملة، 0.0 = غير ذات صلة)

أعد النتيجة بصيغة JSON فقط بدون أي شرح إضافي، بهذا الشكل بالضبط:
{{"faithfulness": 0.0, "relevancy": 0.0, "reasoning": "سبب موجز"}}

{sep}
المصادر:
{sources_text}
{sep}
السؤال: {query}
{sep}
الإجابة: {answer}
{sep}
JSON:"""


def extract_json(raw: str) -> dict:
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


async def judge_answer(generator: ResponseGeneratorAgent, query: str, sources: list, answer: str, sem: asyncio.Semaphore) -> dict:
    sources_text = "\n\n".join(f"[مصدر {i+1}] {s.get('text', '')[:1500]}" for i, s in enumerate(sources))
    prompt = JUDGE_PROMPT.format(sep="=" * 40, sources_text=sources_text, query=query, answer=answer)
    async with sem:
        raw, ok = await generator.generate(prompt)
    if not ok or not raw:
        return {"faithfulness": None, "relevancy": None, "reasoning": "judge_call_failed"}
    parsed = extract_json(raw)
    return {
        "faithfulness": parsed.get("faithfulness"),
        "relevancy": parsed.get("relevancy"),
        "reasoning": parsed.get("reasoning", "")
    }


async def evaluate_one(retriever, reranker, generator, sem, row, top_k):
    query = row["query"]
    fused = retriever.retrieve_hybrid(query, top_k=top_k)
    reranked = reranker.apply_priority_boost(fused)
    prompt, sources_used = generator.build_prompt(query=query, standalone_query=query, sources=reranked, detected_accent="msa")

    if not prompt:
        return {"query": query, "answer": "", "no_answer": True, "faithfulness": None, "relevancy": None, "reasoning": "no_sources_above_threshold"}

    async with sem:
        answer, ok = await generator.generate(prompt)

    if not ok or not answer:
        return {"query": query, "answer": "", "no_answer": True, "faithfulness": None, "relevancy": None, "reasoning": "generation_failed"}

    judged = await judge_answer(generator, query, sources_used, answer, sem)
    return {"query": query, "answer": answer, "no_answer": False, **judged}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"Golden file not found: {args.golden}\nGenerate one first: python -m tests.eval.generate_synthetic_golden")
        return

    golden_rows = []
    with open(args.golden, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                golden_rows.append(json.loads(line))

    if not golden_rows:
        print(f"Golden file {args.golden} is empty.")
        return

    print(f"Loaded {len(golden_rows)} golden queries. Loading retriever + generator...")
    retriever = RetrieverAgent()
    reranker = PriorityReranker()
    generator = ResponseGeneratorAgent()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = await asyncio.gather(*(
        evaluate_one(retriever, reranker, generator, sem, row, args.top_k) for row in golden_rows
    ))

    print(f"\n{'Query':<55} {'Faith':<7} {'Relev':<7} {'Notes'}")
    print("-" * 100)
    for r in results:
        q_display = (r["query"][:52] + "...") if len(r["query"]) > 55 else r["query"]
        if r.get("no_answer"):
            print(f"{q_display:<55} {'-':<7} {'-':<7} NO ANSWER ({r['reasoning']})")
        else:
            f_score = r["faithfulness"]
            r_score = r["relevancy"]
            f_str = f"{f_score:.2f}" if isinstance(f_score, (int, float)) else "?"
            r_str = f"{r_score:.2f}" if isinstance(r_score, (int, float)) else "?"
            print(f"{q_display:<55} {f_str:<7} {r_str:<7} {r.get('reasoning', '')[:30]}")

    scored = [r for r in results if isinstance(r.get("faithfulness"), (int, float)) and isinstance(r.get("relevancy"), (int, float))]
    no_answer_count = sum(1 for r in results if r.get("no_answer"))

    print("-" * 100)
    if scored:
        avg_faith = sum(r["faithfulness"] for r in scored) / len(scored)
        avg_relev = sum(r["relevancy"] for r in scored) / len(scored)
        print(f"Mean faithfulness: {avg_faith:.3f}  (over {len(scored)} scored answers)")
        print(f"Mean relevancy:    {avg_relev:.3f}")
    print(f"No-answer / escalation rate: {no_answer_count}/{len(results)} ({no_answer_count/len(results):.1%})")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"results": results}, f, ensure_ascii=False, indent=2)
        print(f"\nFull report written to {args.report}")


if __name__ == "__main__":
    asyncio.run(main())
