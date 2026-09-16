import os
import re
import json
from typing import List, Dict, Any, Tuple, AsyncGenerator, Optional
from pathlib import Path

import asyncio
from src.pipeline.stage_04_answer.answer_router import StructuredAnswerTemplates

def clean_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r'\*+', '', text)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class RAGChatbot:
    """Wrapper around RAGChatbotEngine for API & CLI backward compatibility."""

    def __init__(self, provider: str = "ollama"):
        from src.rag_chatbot_engine import RAGChatbotEngine
        self.engine = RAGChatbotEngine()

    @property
    def cache_agent(self):
        return self.engine.cache

    async def answer_question(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15,
                               user: Optional[Dict[str, str]] = None, session_id: Optional[str] = None,
                               message_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        res = await self.engine.process_query(query, history=history, user=user, session_id=session_id, message_id=message_id)
        ans = res.get("answer", "")
        if isinstance(ans, (tuple, list)):
            ans = ans[0] if ans else ""
        
        is_escalated = res.get("is_escalated", False) or res.get("escalated", False)
        ticket_id = res.get("ticket_id")
        escalation_reason = res.get("escalation_reason")

        return {
            "answer": ans,
            "sources": res.get("sources", [])[:top_k],
            "cache_hit": res.get("cache_hit", False),
            "is_black_message": res.get("is_black_message", False),
            "is_black": res.get("is_black", False),
            "black_reason": res.get("black_reason"),
            "escalated": is_escalated,
            "is_escalated": is_escalated,
            "ticket_id": ticket_id,
            "escalation_reason": escalation_reason,
            "session_id": session_id,
            "message_id": message_id,
            "metadata": res.get("profiling", {})
        }

    async def answer_question_stream(self, query: str, history: List[Dict[str, str]] = None, top_k: int = 15,
                                      user: Optional[Dict[str, str]] = None, session_id: Optional[str] = None,
                                      message_id: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        hist = history or []

        # NODE 00: Moderation check — flags black message and short-circuits
        is_black, black_reason, black_cat = await self.engine.black_message_agent.is_black_message_async(query, hist)
        if is_black:
            black_res = await self.engine.black_message_agent.handle_black_message_async(
                query=query,
                message_id=message_id,
                session_id=session_id,
                reason=black_reason,
                category=black_cat,
            )
            ans_text = black_res["answer"]
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'cache_hit': False, 'is_black_message': True, 'is_black': True, 'black_reason': black_reason, 'escalated': False, 'is_escalated': False, 'ticket_id': None, 'session_id': session_id, 'message_id': message_id, 'profiling': {}}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': ans_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': ans_text, 'is_black_message': True, 'is_black': True, 'black_reason': black_reason, 'escalated': False, 'is_escalated': False, 'ticket_id': None, 'session_id': session_id, 'message_id': message_id, 'text_breakdown': 'Black Message'}, ensure_ascii=False)}\n\n"
            return

        # NODE 0: Ticket intake intercept (mirrors process_query) -- neither
        # trigger touches the knowledge base. Identity comes from the caller,
        # so the ticket files in this same turn -- no contact-collection
        # round trip.
        if await self.engine.ticket_agent.detect_intent(query, hist):
            ticket_id, answer_text = await asyncio.to_thread(
                self.engine.ticket_agent.escalate, "user_intent", query, hist, user, session_id
            )
            yield f"data: {json.dumps({'type': 'meta', 'sources': [], 'cache_hit': False, 'escalated': True, 'is_escalated': True, 'ticket_id': ticket_id, 'escalation_reason': 'user_intent', 'session_id': session_id, 'profiling': {}}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': answer_text}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'answer': answer_text, 'escalated': True, 'is_escalated': True, 'ticket_id': ticket_id, 'escalation_reason': 'user_intent', 'session_id': session_id, 'text_breakdown': 'Ticket Filed'}, ensure_ascii=False)}\n\n"
            return

        cached_res = self.engine.cache.get(query) if not hist else None
        if cached_res:
            ans = cached_res.get("answer", "")
            if isinstance(ans, (tuple, list)):
                ans = ans[0] if ans else ""
            meta_payload = json.dumps({
                "type": "meta",
                "sources": cached_res.get("sources", [])[:top_k],
                "cache_hit": True,
                "escalated": False,
                "is_escalated": False,
                "ticket_id": None,
                "escalation_reason": None,
                "session_id": session_id,
                "profiling": {}
            }, ensure_ascii=False)
            yield f"data: {meta_payload}\n\n"

            if ans:
                for token in ans:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            done_payload = json.dumps({
                "type": "done",
                "answer": ans,
                "escalated": False,
                "is_escalated": False,
                "ticket_id": None,
                "escalation_reason": None,
                "session_id": session_id,
                "text_breakdown": "Cache Hit"
            }, ensure_ascii=False)
            yield f"data: {done_payload}\n\n"
            return

        query_obj = self.engine.nlp_pipeline.process(query)
        fused_cands = self.engine.retriever.retrieve_hybrid(query_obj["normalized_text"], top_k=15)
        reranked_cands = self.engine.reranker.apply_priority_boost(fused_cands)
        conf_res = self.engine.confidence_eval.evaluate(reranked_cands, query_obj)
        route_res = self.engine.router.route(query_obj, conf_res)
        route = route_res["route"]

        meta_payload = json.dumps({
            "type": "meta",
            "sources": reranked_cands[:top_k],
            "cache_hit": False,
            "escalated": False,
            "is_escalated": False,
            "ticket_id": None,
            "escalation_reason": None,
            "session_id": session_id,
            "profiling": {}
        }, ensure_ascii=False)
        yield f"data: {meta_payload}\n\n"

        full_answer = ""
        escalated = False
        escalated_ticket_id = None
        escalation_reason = None
        if route == "DETERMINISTIC":
            primary_intent = query_obj["intent"][0] if query_obj["intent"] else "MEMBERSHIP"
            tmpl_ans = StructuredAnswerTemplates.get_template_answer(primary_intent)
            full_answer = tmpl_ans or "يرجى زيارة بوابة نقابة المهندسين للحصول على التفاصيل."
            for char in full_answer:
                yield f"data: {json.dumps({'type': 'token', 'token': char}, ensure_ascii=False)}\n\n"
        else:
            prompt_str, sources_used = self.engine.generator.build_prompt(
                query=query,
                standalone_query=query_obj["canonical_msa"],
                sources=reranked_cands,
                detected_accent=query_obj["language"]
            )
            if prompt_str:
                async for token in self.engine.generator.generate_stream(prompt_str):
                    full_answer += token
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

            if not full_answer:
                gen_out = await self.engine.generator.generate(prompt_str) if prompt_str else ""
                full_answer = gen_out[0] if isinstance(gen_out, tuple) else gen_out
                if not full_answer:
                    escalated_ticket_id, full_answer = await asyncio.to_thread(
                        self.engine.ticket_agent.escalate, "low_confidence", query, hist, user, session_id
                    )
                    escalated = True
                    escalation_reason = "low_confidence"
                for token in full_answer:
                    yield f"data: {json.dumps({'type': 'token', 'token': token}, ensure_ascii=False)}\n\n"

        if full_answer and not hist and not escalated:
            self.engine.cache.put(query, full_answer, reranked_cands[:3])

        done_payload = json.dumps({
            "type": "done",
            "answer": full_answer,
            "escalated": escalated,
            "is_escalated": escalated,
            "ticket_id": escalated_ticket_id if escalated else None,
            "escalation_reason": escalation_reason if escalated else None,
            "session_id": session_id,
            "text_breakdown": ""
        }, ensure_ascii=False)
        yield f"data: {done_payload}\n\n"


class KnowledgeRetriever:
    """Knowledge retriever alias for API dependencies."""
    def __init__(self):
        from src.rag_chatbot_engine import RAGChatbotEngine
        self.engine = RAGChatbotEngine()

    @property
    def collection(self):
        return self.engine.retriever.collection

    async def search(self, query: str, top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        res = await self.engine.process_query(query)
        return res.get("sources", [])[:top_k]

    async def retrieve(self, query_text: str = "", query: str = "", top_k: int = 5, **kwargs) -> List[Dict[str, Any]]:
        q = query_text or query
        res = await self.engine.process_query(q)
        return res.get("sources", [])[:top_k]
