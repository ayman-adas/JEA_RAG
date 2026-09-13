import re
import json
import logging
import httpx
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    ACTIVE_CHAT_URL,
    ACTIVE_CHAT_MODEL,
    ACTIVE_CHAT_HEADERS,
    LLM_PROVIDER,
    VLLM_CHAT_URL,
    VLLM_CHAT_MODEL,
    INTERNAL_TICKETS_API_URL,
    INTERNAL_TICKETS_API_KEY,
)
from src.cache_db.document_registry import TicketRegistry
from src.pipeline.stage_05_ticket.ticketing_client import TicketingClient

logger = logging.getLogger(__name__)


# Emergency fallback ONLY -- used when the LLM intent classifier is unreachable
# (vLLM down/timeout). Deliberately a blunt keyword net: it will misfire on
# purely informational questions ("what's the complaint procedure?"), which is
# an acceptable trade-off only because it's the degraded path, not the primary one.
_INTENT_FALLBACK_PATTERNS = [
    re.compile(r'شكو[ىي]', re.IGNORECASE),
    re.compile(r'اشتك', re.IGNORECASE),
    re.compile(r'تذكرة', re.IGNORECASE),
    re.compile(r'موظف\s*(بشري|حقيقي|مختص)?', re.IGNORECASE),
    re.compile(r'شخص\s*(حقيقي|فعلي)', re.IGNORECASE),
    re.compile(r'(كلم|حول|حولني|حولوني|وصلني|وصلوني)\w*\s*(موظف|احد|حدا|شخص)', re.IGNORECASE),
    re.compile(r'\bcomplaint\b', re.IGNORECASE),
    re.compile(r'\bticket\b', re.IGNORECASE),
    re.compile(r'\bhuman\s*(support|agent|being)?\b', re.IGNORECASE),
    re.compile(r'\bagent\b', re.IGNORECASE),
    re.compile(r'\brepresentative\b', re.IGNORECASE),
    re.compile(r'talk to (a |an )?(human|agent|person|representative)', re.IGNORECASE),
    re.compile(r'customer service', re.IGNORECASE),
]

_INTENT_CLASSIFIER_PROMPT_TEMPLATE = (
    "أنت مصنّف نوايا لمساعد نقابة المهندسين الأردنيين.\n"
    "مهمتك: حدد فقط إذا كانت رسالة المستخدم الحالية تعبّر عن رغبة فعلية بالتحدث مع موظف بشري، "
    "تقديم شكوى حقيقية عن مشكلة يعيشها، أو طلب متابعة إنسانية مباشرة لقضيته.\n\n"
    "لا تعتبرها نية تصعيد لمجرد ورود كلمة مثل 'شكوى' أو 'موظف' أو 'تذكرة' ضمن سؤال معلوماتي عادي.\n"
    "مثال على سؤال عادي (الجواب: لا): \"ما هي خطوات تقديم شكوى ضد مكتب هندسي؟\" -- هذا استفسار عن "
    "إجراء، وليس شكوى فعلية.\n"
    "مثال على نية تصعيد حقيقية (الجواب: نعم): \"قدمت طلب تسجيل من شهر ولا حد رد علي، بدي احكي مع حدا\" "
    "أو \"هذا رد غير مقبول، بدي اشتكي رسمياً\" أو \"وصلني رد بالخطأ وبدي اتواصل مع موظف مباشرة\".\n\n"
    "{context}"
    "رسالة المستخدم الحالية: \"{query}\"\n\n"
    "أجب بكلمة واحدة فقط بدون أي شرح أو علامات ترقيم: نعم أو لا."
)

# Data-correction language bumps an explicit escalation to high priority --
# same signal the ported-from reference project uses.
_PRIORITY_BUMP_RE = re.compile(r'خطأ|خطا|تعديل|تحديث', re.IGNORECASE)
_UUID_RE = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')

_MSG_LOW_CONFIDENCE_APOLOGY = (
    "للأسف ما لقيت إجابة دقيقة كافية بخصوص سؤالك ضمن قاعدة المعرفة الحالية.\n"
    "بقدر أحوّل استفسارك لأحد موظفي النقابة ليتابعه معك مباشرة."
)


class TicketIntakeAgent:
    """Detects the two ticket-filing triggers (explicit escalation request /
    low RAG confidence) and files a ticket immediately, using the identity
    the caller already supplies -- no follow-up turn is needed to collect
    contact info. Resolution after that point is a purely human process;
    this agent never generates suggestions or follow-up answers for an
    existing ticket."""

    def __init__(
        self,
        chat_url: Optional[str] = None,
        model: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        provider: Optional[str] = None,
        internal_tickets_url: str = INTERNAL_TICKETS_API_URL,
        internal_tickets_key: str = INTERNAL_TICKETS_API_KEY,
        vllm_url: Optional[str] = None,  # backward compatibility alias
    ):
        self.registry = TicketRegistry()
        self.chat_url = chat_url or vllm_url or ACTIVE_CHAT_URL
        self.vllm_url = self.chat_url  # alias for backward compatibility
        self.model = model or ACTIVE_CHAT_MODEL
        self.headers = headers if headers is not None else ACTIVE_CHAT_HEADERS
        self.provider = (provider or LLM_PROVIDER).lower()
        self.internal_tickets_url = internal_tickets_url
        self.internal_tickets_key = internal_tickets_key
        self.ticketing_client = TicketingClient(
            base_url=internal_tickets_url,
            token=internal_tickets_key,
        )
        self._client: Optional[httpx.Client] = None

    def _get_http_client(self) -> httpx.Client:
        if self._client is None or getattr(self._client, "is_closed", None) is True:
            self._client = httpx.Client(
                timeout=httpx.Timeout(4.0, connect=1.5),
                limits=httpx.Limits(max_keepalive_connections=15, max_connections=50),
            )
        return self._client


    async def detect_intent(self, query: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
        """Semantic escalation-intent classifier -- judges what the user MEANS
        (wants a human / is filing a real complaint / wants to escalate an
        actual problem) rather than matching fixed trigger phrases, so an
        informational question that merely mentions 'complaint' or 'ticket'
        is not misfired on. Falls back to a keyword heuristic only if the
        classifier LLM itself is unreachable."""
        if not query or not query.strip():
            return False

        context = ""
        if history:
            formatted = []
            for h in history[-2:]:
                role = "المستخدم" if h.get("role") == "user" else "المساعد"
                content = h.get("content", "").strip()
                if content:
                    formatted.append(f"{role}: {content}")
            if formatted:
                context = "سياق آخر رسالتين من المحادثة:\n" + "\n".join(formatted) + "\n\n"

        prompt = _INTENT_CLASSIFIER_PROMPT_TEMPLATE.format(context=context, query=query.strip())

        try:
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 5,
            }
            if self.provider == "vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
                res = await client.post(self.chat_url, json=payload, headers=self.headers)
            if res.status_code == 200:
                verdict = res.json()["choices"][0]["message"]["content"].strip()
                return verdict.startswith("نعم") or verdict.lower().startswith("yes")
            logger.warning("[TicketAgent] Intent classifier returned %s: %s", res.status_code, res.text[:200])
        except Exception:
            logger.exception("[TicketAgent] Intent classifier call failed; falling back to keyword heuristic")

        return self._detect_intent_keywords(query)

    def _detect_intent_keywords(self, query: str) -> bool:
        return any(p.search(query) for p in _INTENT_FALLBACK_PATTERNS)

    def classify_department(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Classifies user query and conversation context into the most relevant Syndicate department (sector)
        and optional section using the sectors structure from jea_backend.
        """
        try:
            structure = self.ticketing_client.fetch_sectors_structure()
            departments = structure.get("departments") or structure.get("sectors") or []
            department_sections = structure.get("departmentSections") or {}

            if not departments:
                return None, None

            # Build compact prompt listing the available departments
            dept_lines = []
            for d in departments:
                secs = department_sections.get(d, [])
                if secs:
                    dept_lines.append(f"- {d} (أقسام: {', '.join(secs[:3])})")
                else:
                    dept_lines.append(f"- {d}")
            depts_text = "\n".join(dept_lines)

            context = ""
            if history:
                formatted = []
                for h in history[-2:]:
                    role = "المستخدم" if h.get("role") == "user" else "المساعد"
                    content = h.get("content", "").strip()
                    if content:
                        formatted.append(f"{role}: {content}")
                if formatted:
                    context = "سياق المحادثة:\n" + "\n".join(formatted) + "\n\n"

            prompt = (
                "أنت مصنّف لاختصاصات ودوائر نقابة المهندسين الأردنيين.\n"
                "مهمتك: حدد الدائرة والقسم الأنسب لقضية أو شكوى المستخدم من بين القائمة التالية حصراً:\n\n"
                f"{depts_text}\n\n"
                f"{context}"
                f"رسالة المستخدم: \"{query.strip()}\"\n\n"
                "أجب بصيغة JSON فقط كالتالي بدون أي نص إضافي:\n"
                "{\"department\": \"اسم الدائرة من القائمة\", \"section\": \"اسم القسم إن وُجد وإلا فارغ\"}"
            )

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 80,
            }
            if self.provider == "vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": False}

            client = self._get_http_client()
            res = client.post(self.chat_url, json=payload, headers=self.headers)
            if res.status_code == 200:
                raw_reply = res.json()["choices"][0]["message"]["content"].strip()
                cleaned_json = re.sub(r'```(?:json)?\s*|\s*```', '', raw_reply).strip()
                data = json.loads(cleaned_json)
                chosen_dept = data.get("department", "").strip() or None
                chosen_sec = data.get("section", "").strip() or None

                if chosen_dept:
                    for d in departments:
                        if d == chosen_dept or d in chosen_dept or chosen_dept in d:
                            logger.info("[TicketAgent] LLM classified ticket to department='%s', section='%s'", d, chosen_sec)
                            return d, chosen_sec
                    return chosen_dept, chosen_sec
        except Exception as e:
            logger.warning("[TicketAgent] LLM department classification failed: %s. Using keyword heuristic fallback.", e)

        # Keyword heuristic fallback across known departments
        q_norm = query.lower()
        for d in departments:
            d_norm = d.lower()
            words = [w for w in re.split(r'\s+', d_norm) if len(w) > 3 and w not in ["دائرة", "الدائرة", "قسم"]]
            if any(w in q_norm for w in words):
                logger.info("[TicketAgent] Keyword heuristic matched department='%s'", d)
                return d, None

        return None, None

    def _post_to_internal_ticket_api(
        self,
        reason: str,
        query: str,
        history: Optional[List[Dict[str, str]]],
        user: Optional[Dict[str, str]],
        priority: str,
        session_id: Optional[str] = None,
        department: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Optional[Any]:
        """Optionally post ticket directly to jea_backend POST /api/v1/tickets/internal."""
        if not self.internal_tickets_url:
            return None

        # Headers matching jea_backend tickets.controller.ts:130 (x-internal-token)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-internal-token": self.internal_tickets_key or "jea_rag_token",
        }
        if self.internal_tickets_key:
            headers["X-API-Key"] = self.internal_tickets_key

        # Payload matching jea_backend CreateTicketDto
        ticket_priority = "HIGH" if priority.lower() == "high" else "MEDIUM"
        title = (query[:50] + "...") if len(query) > 50 else query

        history_summary = ""
        if history:
            history_summary = "\n\nسياق المحادثة:\n" + "\n".join(
                f"{h.get('role', 'msg')}: {h.get('content', '')}" for h in history[-3:]
            )

        content = f"استفسار/شكوى المستخدم عبر الشات بوت:\n{query}\n\nسبب التصعيد: {reason}{history_summary}"

        payload: Dict[str, Any] = {
            "title": title,
            "content": content,
            "ticketPriority": ticket_priority,
            "skipWorkingHoursCheck": True,
        }

        if session_id and _UUID_RE.match(session_id.strip()):
            payload["sessionId"] = session_id.strip()

        if department:
            payload["department"] = department.strip()
            payload["sector"] = department.strip()

        if section:
            payload["section"] = section.strip()

        phone = (
            (user or {}).get("phone")
            or (user or {}).get("userPhoneNumber")
            or (user or {}).get("user_phone_number")
            or (user or {}).get("phoneNumber")
        )
        if phone:
            payload["userPhoneNumber"] = str(phone).strip()

        try:
            client = self._get_http_client()
            res = client.post(self.internal_tickets_url, json=payload, headers=headers)
            if res.status_code in (200, 201):
                data = res.json()
                ticket_id = data.get("id") or data.get("ticket_id")
                if isinstance(data.get("data"), dict):
                    ticket_id = ticket_id or data["data"].get("id") or data["data"].get("ticket_id")
                if ticket_id is not None:
                    logger.info("[TicketAgent] Successfully filed ticket via jea_backend internal API: #%s", ticket_id)
                    return ticket_id
            logger.warning("[TicketAgent] Internal ticket API returned %s: %s", res.status_code, res.text[:200])
        except Exception as e:
            logger.warning("[TicketAgent] Internal ticket API call failed: %s. Falling back to local storage.", e)
        return None

    def escalate(
        self,
        reason: str,
        query: str,
        history: Optional[List[Dict[str, str]]],
        user: Optional[Dict[str, str]],
        session_id: Optional[str] = None,
    ) -> Tuple[Any, str]:
        """Files the ticket and returns (ticket_id, user_facing_message).
        Synchronous (SQLite + optional jea_backend internal API) -- call via asyncio.to_thread from async code."""
        priority = "high" if reason == "user_intent" and _PRIORITY_BUMP_RE.search(query) else "medium"

        # Classify department & section from Syndicate sectors structure
        department, section = self.classify_department(query, history)

        # 1. Try jea_backend internal API if configured
        internal_ticket_id = self._post_to_internal_ticket_api(
            reason=reason,
            query=query,
            history=history,
            user=user,
            priority=priority,
            session_id=session_id,
            department=department,
            section=section,
        )

        # 2. Always persist locally in SQLite registry for audit & offline fallback
        local_ticket_id = self.registry.create_ticket(
            reason=reason,
            query=query,
            history=history or [],
            user=user or {},
            priority=priority,
            session_id=session_id,
            external_ticket_id=str(internal_ticket_id) if internal_ticket_id else "",
            department=department or "",
            section=section or "",
        )

        # 3. Use internal ticket ID if available, otherwise local SQLite ID
        ticket_id = internal_ticket_id if internal_ticket_id is not None else local_ticket_id
        ticket_display = str(ticket_id)[:8] if len(str(ticket_id)) > 8 else str(ticket_id)

        if reason == "low_confidence":
            message = f"{_MSG_LOW_CONFIDENCE_APOLOGY}\n\nتم فتح تذكرة متابعة رقم #{ticket_display}."
        else:
            message = self._build_confirmation(ticket_display)
        return ticket_id, message

    def _build_confirmation(self, ticket_display: Any) -> str:
        return (
            f"تم فتح تذكرة رقم #{ticket_display} بنجاح، وسيتواصل معك أحد موظفي النقابة قريباً لمتابعة طلبك.\n"
            "شكراً لتواصلك مع النقابة."
        )
