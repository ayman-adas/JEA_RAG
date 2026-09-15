import time
import asyncio
from typing import Dict, Any, List, Optional

from src.monitoring.tracing import RequestTracer
from src.monitoring.profiler import PipelineProfiler
from src.monitoring.diagnostic_report import DiagnosticAnalyzer

from src.pipeline.stage_01_understand.query_preprocessor import QueryPreprocessor
from src.cache_db.semantic_cache import SemanticCache
from src.pipeline.stage_02_retrieve.bm25_retriever import BM25Retriever
from src.pipeline.stage_02_retrieve.rrf_fusion import RRFFusion
from src.pipeline.stage_03_verify_rerank.reranker import PriorityReranker
from src.pipeline.stage_03_verify_rerank.confidence import DeterministicConfidenceEvaluator
from src.pipeline.stage_04_answer.answer_router import AnswerRouter
from src.pipeline.stage_04_answer.generator_agent import ResponseGeneratorAgent
from src.pipeline.stage_02_retrieve.retriever_agent import RetrieverAgent
from src.pipeline.stage_05_ticket.ticket_agent import TicketIntakeAgent


class RAGChatbotEngine:
    """
    Central Orchestrator Engine for Eng-Guild Chatbot.
    Implements the complete Normalize → Understand → Retrieve → Verify → Answer architecture
    with first-class 13-stage diagnostic profiling.
    """

    def __init__(self):
        self.nlp_pipeline = QueryPreprocessor()
        self.cache = SemanticCache()
        self.bm25 = BM25Retriever()
        self.rrf = RRFFusion()
        self.reranker = PriorityReranker()
        self.retriever = RetrieverAgent()
        self.confidence_eval = DeterministicConfidenceEvaluator()
        self.router = AnswerRouter()
        self.generator = ResponseGeneratorAgent()
        self.ticket_agent = TicketIntakeAgent()

    async def process_query(self, user_query: str,
                            history: Optional[List[Dict[str, str]]] = None,
                            user: Optional[Dict[str, str]] = None,
                            session_id: Optional[str] = None) -> Dict[str, Any]:
        request_id = RequestTracer.generate_request_id()
        profiler = PipelineProfiler(request_id)
        hist = history or []

        # 0. Ticket Intake Stage — runs before cache/retrieval since this
        # trigger needs neither. Identity comes from the caller (the backend
        # already knows the user), so the ticket files in this same turn --
        # no contact-collection round trip. Resolution after that point is a
        # purely human process.
        with profiler.time_stage("ticket_intake"):
            if await self.ticket_agent.detect_intent(user_query, hist):
                ticket_id, answer_text = await asyncio.to_thread(
                    self.ticket_agent.escalate, "user_intent", user_query, hist, user, session_id
                )
                report = profiler.generate_report()
                profiler.write_request_log(user_query, answer_text, cache_hit=False, route="TICKET_FILED")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": answer_text,
                    "sources": [],
                    "cache_hit": False,
                    "route": {"route": "TICKET_FILED", "ticket_id": ticket_id, "escalation_reason": "user_intent"},
                    "escalated": True,
                    "is_escalated": True,
                    "ticket_id": ticket_id,
                    "escalation_reason": "user_intent",
                    "session_id": session_id,
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

        # 1. Cache Stage
        with profiler.time_stage("cache"):
            query_vec = self.retriever.embedder.embed_single_text(user_query) if hasattr(self.retriever, "embedder") else None
            cached_res = self.cache.get(user_query, query_vec=query_vec, session_id=session_id) if not hist else None
            if cached_res:
                report = profiler.generate_report()
                profiler.write_request_log(user_query, cached_res["answer"], cache_hit=True, route="CACHE")
                return {
                    "request_id": request_id,
                    "query": {"original": user_query},
                    "answer": cached_res["answer"],
                    "sources": cached_res.get("sources", []),
                    "cache_hit": True,
                    "escalated": False,
                    "is_escalated": False,
                    "ticket_id": None,
                    "escalation_reason": None,
                    "session_id": session_id,
                    "profiling": report,
                    "text_breakdown": profiler.format_text_breakdown()
                }

        # 2. Language Detection Stage
        with profiler.time_stage("language_detection"):
            lang_res = self.nlp_pipeline.detect_language(user_query)

        # 3. Normalization Stage
        with profiler.time_stage("normalization"):
            norm_text = self.nlp_pipeline.normalize_arabic(user_query)

        # 4. MSA Conversion & Multi-Representation Stage
        with profiler.time_stage("msa_conversion"):
            query_obj = self.nlp_pipeline.process(user_query)

        # 5. Intent Stage
        with profiler.time_stage("intent"):
            intent_res = {"intents": query_obj["intent"], "confidence": query_obj["intent_confidence"]}

        # 6-9. Dense Vector & Sparse Hybrid Retrieval Stage
        with profiler.time_stage("bm25"):
            fused_cands = self.retriever.retrieve_hybrid(query_obj["normalized_text"], top_k=15, query_vec=query_vec)

        with profiler.time_stage("bge_m3"):
            pass

        with profiler.time_stage("rapidfuzz"):
            pass

        with profiler.time_stage("rrf"):
            pass

        # 10. Reranker & Priority Boost Stage
        with profiler.time_stage("priority"):
            reranked_cands = self.reranker.rerank(query_obj["normalized_text"], fused_cands)

        # 11. CRAG & Confidence Stage
        with profiler.time_stage("crag"):
            conf_res = self.confidence_eval.evaluate(reranked_cands, query_obj)

        # 12. Answer Routing & Synthesis Stage
        escalated = False
        with profiler.time_stage("answer"):
            route_res = self.router.route(query_obj, conf_res)
            route = route_res["route"]

            if route == "DETERMINISTIC":
                primary_intent = query_obj["intent"][0] if query_obj["intent"] else "MEMBERSHIP"
                tmpl_ans = StructuredAnswerTemplates.get_template_answer(primary_intent)
                answer_text = tmpl_ans or "يرجى زيارة بوابة نقابة المهندسين للحصول على التفاصيل."
            else:
                prompt_str, sources_used = self.generator.build_prompt(
                    query=user_query,
                    standalone_query=query_obj["canonical_msa"],
                    sources=reranked_cands,
                    detected_accent=query_obj["language"]
                )
                answer_text = None
                if prompt_str:
                    gen_out = await self.generator.generate(prompt_str)
                    answer_text = gen_out[0] if isinstance(gen_out, tuple) else gen_out

                # No grounded answer -> escalate to a human ticket, in this
                # same turn, rather than a dead end.
                if not answer_text:
                    ticket_id, answer_text = await asyncio.to_thread(
                        self.ticket_agent.escalate, "low_confidence", user_query, hist, user, session_id
                    )
                    escalated = True
                    route_res = {**route_res, "ticket_id": ticket_id, "escalation_reason": "low_confidence"}

        # Cache final answer if valid (single-turn only; never cache an
        # escalation reply -- it embeds a one-off ticket number)
        if answer_text and not hist and not escalated:
            self.cache.put(user_query, answer_text, reranked_cands[:3], query_vec=query_vec, session_id=session_id)

        profiler.record_stage("answer", profiler.stage_latencies["answer"], {"route": route})
        report = profiler.generate_report()
        diagnostics = DiagnosticAnalyzer.analyze_trace(report)
        profiler.write_request_log(user_query, answer_text, cache_hit=False, route=route)

        return {
            "request_id": request_id,
            "query": query_obj,
            "answer": answer_text,
            "sources": [] if escalated else reranked_cands[:5],
            "confidence": conf_res,
            "route": route_res,
            "cache_hit": False,
            "escalated": escalated,
            "is_escalated": escalated,
            "ticket_id": route_res.get("ticket_id") if escalated else None,
            "escalation_reason": route_res.get("escalation_reason") if escalated else None,
            "session_id": session_id,
            "profiling": report,
            "diagnostics": diagnostics,
            "text_breakdown": profiler.format_text_breakdown()
        }
