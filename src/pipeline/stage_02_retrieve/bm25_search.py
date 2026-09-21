import os
import re
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from src.config import BM25_INDEX_PATH

def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[ؗ-ًؚ-ْ]', '', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[إأآ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = text.lower()
    return text

def tokenize_arabic(text: str) -> List[str]:
    norm = normalize_arabic_text(text)
    tokens = re.findall(r'[\w\d]+', norm)
    return tokens

class BM25Indexer:
    def __init__(self, index_path: Path = BM25_INDEX_PATH, k1: float = 1.5, b: float = 0.75):
        self.index_path = Path(index_path)
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.inverted_index: Dict[str, List[int]] = {}
        self.load()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        for chunk in chunks:
            text = chunk.get("text", "")
            tokens = tokenize_arabic(text)
            self.corpus.append(chunk)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            freqs = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
        self._recalc_index()

    def _recalc_index(self):
        N = len(self.corpus)
        if N == 0:
            self.avgdl = 0.0
            self.idf = {}
            self.inverted_index = {}
            return
        self.avgdl = sum(self.doc_len) / float(N)
        df: Dict[str, int] = {}
        self.inverted_index = {}
        for idx, freqs in enumerate(self.doc_freqs):
            for term in freqs.keys():
                df[term] = df.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append(idx)

        self.idf = {}
        for term, freq in df.items():
            idf_val = math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            self.idf[term] = max(0.1, idf_val)

    def search(self, query: str, top_k: int = 15) -> List[Tuple[Dict[str, Any], float]]:
        q_tokens = tokenize_arabic(query)
        if not q_tokens or not self.corpus:
            return []

        # Find matching document indices via inverted index instead of
        # scanning every doc_freqs entry per query term.
        matching_indices = set()
        for t in q_tokens:
            if t in self.inverted_index:
                matching_indices.update(self.inverted_index[t])

        if not matching_indices:
            return []

        scores: Dict[int, float] = {idx: 0.0 for idx in matching_indices}
        for t in q_tokens:
            if t not in self.idf or t not in self.inverted_index:
                continue
            idf_val = self.idf[t]
            for idx in self.inverted_index[t]:
                freqs = self.doc_freqs[idx]
                tf = freqs[t]
                dl = self.doc_len[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl or 1.0)))
                score = idf_val * (tf * (self.k1 + 1.0)) / denom
                scores[idx] += score

        results = [(self.corpus[i], score) for i, score in scores.items() if score > 0]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self.corpus,
            "doc_tokens": self.doc_tokens,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "doc_freqs": self.doc_freqs,
            "idf": self.idf,
            "inverted_index": self.inverted_index
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        if self.index_path.exists():
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.corpus = data.get("corpus", [])
                    self.doc_tokens = data.get("doc_tokens", [])
                    self.doc_len = data.get("doc_len", [])
                    self.avgdl = data.get("avgdl", 0.0)
                    self.doc_freqs = data.get("doc_freqs", [])
                    self.idf = data.get("idf", {})
                    self.inverted_index = data.get("inverted_index", {})
                    if not self.inverted_index and self.doc_freqs:
                        self._recalc_index()
            except Exception:
                pass
