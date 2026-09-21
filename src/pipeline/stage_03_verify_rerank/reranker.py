from typing import Dict, Any, List

from src.config import DOCUMENT_PRIORITY, BGE_RERANKER_MODEL_NAME, USE_FP16, MIN_VRAM_GB_RERANKER
from src.core.gpu_utils import pick_device


class PriorityReranker:
    """Reranks candidates using a BGE cross-encoder (query-document joint scoring)
    followed by document authority priority boosts."""

    def __init__(self, model_name: str = BGE_RERANKER_MODEL_NAME, priority_map: Dict[str, float] = DOCUMENT_PRIORITY):
        self.model_name = model_name
        self.priority_map = priority_map
        self.reranker_model = None

    def _lazy_load(self):
        if self.reranker_model is None:
            try:
                from FlagEmbedding import FlagReranker
                device = pick_device(MIN_VRAM_GB_RERANKER)
                print(f"[Reranker] Loading Cross-Encoder model '{self.model_name}' on {device}...")
                use_fp16 = USE_FP16 and device != "cpu"
                self.reranker_model = FlagReranker(self.model_name, use_fp16=use_fp16, devices=device)
                print(f"[Reranker] Successfully loaded Cross-Encoder '{self.model_name}' on {device}.")
            except Exception as e:
                print(f"[Reranker Warning] Could not load FlagReranker '{self.model_name}': {e}. Using RRF rank fallback.")
                self.reranker_model = False

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 15) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        self._lazy_load()
        boosted = []

        if self.reranker_model:
            try:
                pairs = []
                for cand in candidates:
                    text = cand.get("text", "")
                    parent = cand.get("parent_text", "")
                    doc_str = parent if parent else text
                    pairs.append([query, doc_str])

                raw_scores = self.reranker_model.compute_score(pairs, normalize=True)
                if isinstance(raw_scores, (float, int)):
                    raw_scores = [float(raw_scores)]

                for cand, score in zip(candidates, raw_scores):
                    c = dict(cand)
                    c["cross_encoder_score"] = round(float(score), 4)
                    boosted.append(c)
            except Exception as e:
                print(f"[Reranker Warning] Scoring failed: {e}")
                boosted = [dict(c) for c in candidates]
        else:
            boosted = [dict(c) for c in candidates]

        return self.apply_priority_boost(boosted)[:top_k]

    def apply_priority_boost(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        boosted = []
        for cand in candidates:
            c = dict(cand)
            title = c.get("title", "") + " " + c.get("file_name", "")
            boost = 0.0
            for key, val in self.priority_map.items():
                if key in title:
                    boost = max(boost, val)

            if "cross_encoder_score" in c:
                base_sim = round(float(c["cross_encoder_score"]) * 100.0, 1)
            elif "distance" in c and c["distance"] is not None:
                base_sim = round(max(0.0, min(1.0, 1.0 - float(c["distance"]))) * 100.0, 1)
            elif "similarity_score" in c and float(c.get("similarity_score", 0)) > 1.0:
                base_sim = float(c["similarity_score"])
            elif "rrf_score" in c and c["rrf_score"] is not None:
                rrf_val = float(c["rrf_score"])
                base_sim = round(min(95.0, max(40.0, rrf_val * 2400.0)), 1)
            else:
                base_sim = 50.0

            final_score = round(min(98.0, max(50.0, base_sim + (boost * 20.0))), 1)
            c["priority_boost"] = boost
            c["final_score"] = final_score
            c["similarity_score"] = final_score
            boosted.append(c)

        boosted.sort(key=lambda x: x["similarity_score"], reverse=True)
        return boosted
