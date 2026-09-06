import os
import re
import json
import time
import logging
import httpx
from typing import Dict, Any, List, Tuple, Optional, AsyncGenerator

from src.config import (
    VLLM_CHAT_URL, VLLM_CHAT_MODEL, NUM_PREDICT, RELEVANCE_THRESHOLD
)

logger = logging.getLogger(__name__)

# Repetition/frequency penalty guards against degenerate decoding loops
# (verbatim phrase repetition, or drift into unrelated-language tokens) that
# low-temperature sampling with no penalty is prone to getting stuck in.
REPETITION_PENALTY = float(os.getenv("GENERATION_REPETITION_PENALTY", "1.2"))
FREQUENCY_PENALTY = float(os.getenv("GENERATION_FREQUENCY_PENALTY", "0.4"))

DEGENERATE_NGRAM_SIZE = 8
DEGENERATE_NGRAM_MAX_REPEATS = 3

# Scripts that are never legitimate in this system's output (Arabic or
# English only, per build_prompt's two response templates). Their presence
# -- even as a handful of stray characters inside one otherwise-normal
# Arabic word -- is a reliable sign of decoding drift, unlike a low
# Arabic-character *ratio*, which would also misfire on legitimate
# English-language answers.
_DISALLOWED_SCRIPT_RE = re.compile(
    '['
    '฀-๿'   # Thai
    '一-鿿'   # CJK Unified Ideographs
    '぀-ヿ'   # Hiragana + Katakana
    '가-힣'   # Hangul syllables
    'ऀ-ॿ'   # Devanagari
    'Ѐ-ӿ'   # Cyrillic
    ']'
)


def is_degenerate_output(text: str) -> bool:
    """Flags garbled model output before it reaches a user: script drift
    (e.g. Thai/CJK tokens leaking into an Arabic answer) or a repetition
    loop (the same phrase repeated well beyond normal emphasis)."""
    if not text or not text.strip():
        return True

    if _DISALLOWED_SCRIPT_RE.search(text):
        return True

    words = text.split()
    if len(words) >= DEGENERATE_NGRAM_SIZE * DEGENERATE_NGRAM_MAX_REPEATS:
        seen_counts: Dict[str, int] = {}
        for i in range(len(words) - DEGENERATE_NGRAM_SIZE + 1):
            gram = " ".join(words[i:i + DEGENERATE_NGRAM_SIZE])
            count = seen_counts.get(gram, 0) + 1
            seen_counts[gram] = count
            if count >= DEGENERATE_NGRAM_MAX_REPEATS:
                return True

    return False


_THINK_OPEN_RE = re.compile(r'<\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r'</\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)
_THINK_PAIR_RE = re.compile(
    r'<\s*(think|thinking|reasoning)\s*>.*?</\s*\1\s*>',
    re.IGNORECASE | re.DOTALL
)


def strip_reasoning(text: str) -> str:
    if not text:
        return ""

    text = _THINK_PAIR_RE.sub('', text)

    m_close = _THINK_CLOSE_RE.search(text)
    if m_close and not _THINK_OPEN_RE.search(text[:m_close.start()]):
        text = text[m_close.end():]

    m_open = _THINK_OPEN_RE.search(text)
    if m_open and not _THINK_CLOSE_RE.search(text[m_open.end():]):
        tail = text[m_open.end():]
        parts = re.split(r'\n\s*\n', tail.strip())
        text = text[:m_open.start()] + (parts[-1] if parts else "")

    return text.strip()


class ReasoningStreamFilter:
    _MAX_TAG_LEN = len('</reasoning>') + 4

    def __init__(self):
        self._buf = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        self._buf += token
        out = []
        while True:
            if self._in_think:
                m = _THINK_CLOSE_RE.search(self._buf)
                if m:
                    self._buf = self._buf[m.end():]
                    self._in_think = False
                    continue
                self._buf = self._buf[-self._MAX_TAG_LEN:]
                break
            else:
                m = _THINK_OPEN_RE.search(self._buf)
                if m:
                    out.append(self._buf[:m.start()])
                    self._buf = self._buf[m.end():]
                    self._in_think = True
                    continue
                safe_len = len(self._buf) - self._MAX_TAG_LEN
                if safe_len > 0:
                    out.append(self._buf[:safe_len])
                    self._buf = self._buf[safe_len:]
                break
        return "".join(out)

    def flush(self) -> str:
        remainder = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        if remainder and _THINK_OPEN_RE.match(remainder.strip()):
            return ""
        return remainder


def clean_formatting(text: str) -> str:
    if not text:
        return ""
    cleaned = strip_reasoning(text)
    cleaned = re.sub(r'\*+', '', cleaned)
    cleaned = re.sub(r'^#+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned.strip()


class ResponseGeneratorAgent:
    def __init__(self, vllm_url: str = VLLM_CHAT_URL, model: str = VLLM_CHAT_MODEL):
        self.vllm_url = vllm_url
        self.model = model

    def build_prompt(self,
                     query: str,
                     standalone_query: str,
                     sources: List[Dict[str, Any]],
                     detected_accent: str = "msa") -> Tuple[str, List[Dict[str, Any]]]:
        MAX_CONTEXT_CHARS = 9000
        context_blocks = []
        accumulated_chars = 0
        included_sources = []

        for src in sources:
            score = src.get("similarity_score", 0)
            rank = len(included_sources) + 1
            title = src.get("title", "وثيقة")
            section = src.get("section_title", "")
            raw_text = src.get("text", "").strip()
            if not raw_text:
                continue

            header = f"[المصدر {rank}] {title}"
            if section:
                header += f" — {section}"
            header += f" (نسبة التطابق: {score}%)"

            block_str = f"{header}\n{raw_text}\n"
            if accumulated_chars + len(block_str) > MAX_CONTEXT_CHARS and context_blocks:
                break

            context_blocks.append(block_str)
            accumulated_chars += len(block_str)
            included_sources.append(src)

        if not included_sources:
            return "", []

        context_str = "\n" + ("=" * 50) + "\n" + "\n".join(context_blocks) + ("=" * 50)

        acc_lower = str(detected_accent).lower().strip()
        is_english = acc_lower in ("english", "en", "mixed") or acc_lower.startswith("en")
        if is_english:
            system_prompt = (
                f"You are an Eng-Guild information assistant for the Jordan Engineers Association (JEA).\n"
                f"Strict Response Guidelines (AGENTS.md):\n"
                f"1. GIVE THE DIRECT ANSWER FIRST in sentence 1 at the very top. ZERO preamble, ZERO introductory filler, ZERO greeting.\n"
                f"2. Mandatory Structure: Direct Answer -> Essential Details -> Actionable Next Steps.\n"
                f"3. Short Sentences & Maximum Density: Write using short, concise sentences. Retain maximum factual detail while eliminating fluff.\n"
                f"4. Language Matching: Respond strictly in professional English.\n"
                f"5. Source Citation: Cite source numbers like [Source 1], [Source 2] next to facts.\n"
                f"6. NO Chatbot Artifacts: NO generic sign-offs ('Let me know if you need anything else'), NO conversational filler.\n"
                f"7. Accuracy: Do NOT invent facts. If missing, state: 'I couldn't find that information in the guild resources.'\n"
            )
            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"Knowledge Base Sources:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"User Question: {query}\n"
                f"{'=' * 60}\n"
                f"Direct Answer:"
            )
        else:
            if detected_accent in ("jordanian", "ar-JO"):
                lang_instruction = "مطابقة اللغة والهجة: أجب باللغة العربية بأسلوب نقابي أردني مباشر ومحترف يطابق سؤال المستخدم."
            else:
                lang_instruction = "مطابقة اللغة والهجة: أجب باللغة العربية الفصحى المباشرة."

            system_prompt = (
                f"أنت مساعد معلومات نقابة المهندسين الأردنيين (JEA).\n"
                f"التزم بقواعد الإجابة الصارمة وفق دليل (AGENTS.md):\n\n"
                f"1. قدم الإجابة المباشرة فوراً في السطر الأول. يمنع منعاً باتاً كتابة مقدمات تمهيدية أو جمل ترحيبية أو تمهيد.\n"
                f"2. الهيكل الإجباري للإجابة: [الإجابة المباشرة] ← [التفاصيل الجوهرية والنقاط الأساسية] ← [الخطوات أو الإجراءات العملية التالية].\n"
                f"3. الجمل القصيرة والكثافة العالية: اكتب في جمل قصيرة ومباشرة، وحافظ على جميع التفاصيل والمعلومات الدقيقة مع إلغاء الكلمات الزائدة الحشو.\n"
                f"4. {lang_instruction}\n"
                f"5. التوثيق: اذكر رقم المصدر [المصدر N] بجانب كل معلومة واستشهاد.\n"
                f"6. خلو من المظاهر الزائفة: يُحظر إضافة أي خاتمة روتينية أو جمل ترحيبية أو عرض مساعدة إضافية في النهاية (مثل 'تختص النقابة...' أو 'في حال وجود استفسارات').\n"
                f"7. الدقة والاعتماد على المصادر: اجب اعتماداً على المعلومات الواردة في المصادر المرفقة بأعلاه، والخص كافة التفاصيل والخطوات والشروط المذكورة بدقة.\n"
            )

            full_prompt = (
                f"{system_prompt}\n"
                f"{'=' * 60}\n"
                f"المصادر من قاعدة المعرفة:\n{context_str}\n"
                f"{'=' * 60}\n"
                f"سؤال المستخدم: {query}\n"
                f"{'=' * 60}\n"
                f"الإجابة المباشرة:"
            )

        return full_prompt, included_sources

    async def generate(self, prompt: str) -> Tuple[str, bool]:
        """Batch generation against the vLLM OpenAI-compatible chat endpoint."""
        if not prompt:
            return "", False

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.1,
                "max_tokens": NUM_PREDICT,
                "repetition_penalty": REPETITION_PENALTY,
                "frequency_penalty": FREQUENCY_PENALTY,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=3.0)) as client:
                res = await client.post(self.vllm_url, json=payload)
            if res.status_code == 200:
                content = res.json()["choices"][0]["message"]["content"].strip()
                if content:
                    cleaned = clean_formatting(content)
                    if is_degenerate_output(cleaned):
                        logger.warning("[GeneratorAgent] Discarding degenerate output (repetition loop or script drift): %r", cleaned[:200])
                        return "", False
                    return cleaned, True
            else:
                logger.warning("[GeneratorAgent] vLLM returned %s: %s", res.status_code, res.text[:300])
        except Exception:
            logger.exception("[GeneratorAgent] generate() failed calling vLLM at %s", self.vllm_url)

        return "", False

    async def generate_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """Yields real-time token chunks from the vLLM chat endpoint with stream=True."""
        if not prompt:
            return

        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
                "temperature": 0.1,
                "max_tokens": NUM_PREDICT,
                "repetition_penalty": REPETITION_PENALTY,
                "frequency_penalty": FREQUENCY_PENALTY,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=3.0)) as client:
                async with client.stream("POST", self.vllm_url, json=payload) as res:
                    if res.status_code != 200:
                        body = await res.aread()
                        logger.warning("[GeneratorAgent] vLLM stream returned %s: %s", res.status_code, body[:300])
                        return

                    reasoning_filter = ReasoningStreamFilter()
                    async for line in res.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data)
                            delta = chunk_json["choices"][0].get("delta", {})
                            token = delta.get("content") or ""
                            safe = reasoning_filter.feed(token)
                            safe = safe.replace("*", "").replace("#", "")
                            if safe:
                                yield safe
                        except Exception:
                            logger.exception("[GeneratorAgent] Failed to parse vLLM stream chunk: %r", data[:300])

                    tail = reasoning_filter.flush()
                    tail = tail.replace("*", "").replace("#", "")
                    if tail:
                        yield tail
        except Exception:
            logger.exception("[GeneratorAgent] generate_stream() failed calling vLLM at %s", self.vllm_url)
