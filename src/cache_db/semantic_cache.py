import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

from src.config import CACHE_SIMILARITY_THRESHOLD, CACHE_MAX_ENTRIES, CACHE_PERSIST_PATH


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


class SemanticCache:
    """Fast semantic cache intercepting queries using exact match and strict vector cosine similarity (< 10ms responses)."""

    def __init__(self, persist_path: Path = CACHE_PERSIST_PATH):
        self.persist_path = Path(persist_path)
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception:
                self.entries = []

    def get(self, query: str, query_vec: Optional[List[float]] = None,
            threshold: float = CACHE_SIMILARITY_THRESHOLD, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        q_norm = query.strip().lower()
        words = q_norm.split()

        # Restrict matching to entries belonging to the same session_id (or,
        # if no session_id given, to entries recorded without one) so a
        # cache hit never leaks an answer from an unrelated conversation.
        if not session_id:
            candidate_entries = [e for e in self.entries if not e.get("session_id")]
        else:
            candidate_entries = [e for e in self.entries if e.get("session_id") == session_id]

        if not candidate_entries:
            return None

        # 1. Exact string match
        for entry in candidate_entries:
            if entry.get("query_normalized") == q_norm:
                return entry

        # Vector similarity requires at least 2 words to avoid single-token false matches
        if len(words) < 2:
            return None

        # 2. Vector cosine similarity match
        if query_vec:
            best_entry = None
            best_sim = 0.0
            for entry in candidate_entries:
                cached_vec = entry.get("query_vec")
                if cached_vec:
                    sim = cosine_similarity(query_vec, cached_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_entry and best_sim >= threshold:
                return best_entry

        return None

    def put(self, query: str, answer: str, sources: List[Dict[str, Any]],
            query_vec: Optional[List[float]] = None, session_id: Optional[str] = None):
        if not answer or not answer.strip():
            return  # Do not cache empty answers

        fallback_phrases = [
            "لم أتمكن من العثور على معلومات دقيقة",
            "لم أتمكن من العثور على هذه المعلومة",
            "I couldn't find that information",
            "لا تتوفر تفاصيل دقيقة",
            "غير مذكور في المصادر",
            "عذراً، لا يمكنني إجابة",
            "no relevant information found"
        ]
        ans_lower = answer.lower()
        if any(phrase.lower() in ans_lower for phrase in fallback_phrases):
            return  # Do not cache failure/uncertain answers

        q_norm = query.strip().lower()
        entry = {
            "query_normalized": q_norm,
            "answer": answer,
            "sources": sources
        }
        if session_id:
            entry["session_id"] = session_id
        if query_vec:
            entry["query_vec"] = query_vec

        self.entries.insert(0, entry)
        self.entries = self.entries[:CACHE_MAX_ENTRIES]
        self._save()

    def clear(self):
        self.entries = []
        self._save()

    def _save(self):
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
