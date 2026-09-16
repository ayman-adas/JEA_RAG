import time
import logging
import httpx
from typing import Dict, Any, Optional

from src.config import (
    JEA_BACKEND_BASE_URL,
    INTERNAL_BYPASS_TOKEN,
)
from src.services.http_client_pool import get_sync_http_client, get_async_http_client

logger = logging.getLogger(__name__)


DEFAULT_FALLBACK_SECTORS_STRUCTURE: Dict[str, Any] = {
    "sectors": [
        "دائرة التأمين الصحي",
        "دائرة التقاعد والضمان الاجتماعي",
        "دائرة الشؤون الهندسية والمكاتب",
        "دائرة التدريب والتأهيل والمؤتمرات",
        "الدائرة المالية",
        "الدائرة الإدارية والموارد البشرية",
        "دائرة تكنولوجيا المعلومات والتحول الرقمي",
        "الدائرة القانونية",
    ],
    "departments": [
        "دائرة التأمين الصحي",
        "دائرة التقاعد والضمان الاجتماعي",
        "دائرة الشؤون الهندسية والمكاتب",
        "دائرة التدريب والتأهيل والمؤتمرات",
        "الدائرة المالية",
        "الدائرة الإدارية والموارد البشرية",
        "دائرة تكنولوجيا المعلومات والتحول الرقمي",
        "الدائرة القانونية",
    ],
    "sections": [
        "قسم التأمين الصحي والمطالبات",
        "قسم الرواتب التقاعدية",
        "قسم تسجيل المكاتب والشركات الهندسية",
        "قسم التصديق والشهادات الهندسية",
        "قسم الدورات والتدريب المهني",
        "قسم التحصيل والاشتراكات",
        "قسم الشؤون القانونية والشكاوى",
        "قسم الدعم الفني وتكنولوجيا المعلومات",
    ],
    "departmentSections": {
        "دائرة التأمين الصحي": ["قسم التأمين الصحي والمطالبات", "قسم الموافقات الطبية"],
        "دائرة التقاعد والضمان الاجتماعي": ["قسم الرواتب التقاعدية", "قسم القروض والتسهيلات"],
        "دائرة الشؤون الهندسية والمكاتب": ["قسم تسجيل المكاتب والشركات الهندسية", "قسم التصديق والشهادات الهندسية"],
        "دائرة التدريب والتأهيل والمؤتمرات": ["قسم الدورات والتدريب المهني", "قسم المؤتمرات والندوات"],
        "الدائرة المالية": ["قسم التحصيل والاشتراكات", "قسم المحاسبة العامة"],
        "الدائرة الإدارية والموارد البشرية": ["قسم شؤون الموظفين", "قسم المراسلات والديوان"],
        "دائرة تكنولوجيا المعلومات والتحول الرقمي": ["قسم الدعم الفني وتكنولوجيا المعلومات", "قسم التطوير والنظم"],
        "الدائرة القانونية": ["قسم الشؤون القانونية والشكاوى", "قسم التحكيم والنزاعات"],
    },
}


class TicketingClient:
    """
    HTTP client for JEA Backend (NestJS) Internal Ticketing APIs.
    Fetches the syndicate sectors/departments structure used to route
    tickets, and creates escalation tickets.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        raw_url = base_url or JEA_BACKEND_BASE_URL or "http://localhost:3000"
        # If full /api/v1/tickets/internal was passed as base, strip down to domain root
        if "/api/" in raw_url:
            raw_url = raw_url.split("/api/")[0]
        self.base_url = raw_url.rstrip("/")

        self.token = token or INTERNAL_BYPASS_TOKEN or "jea_rag_token"
        self.headers = {
            "x-internal-token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.sectors_ttl_seconds = 86400  # 24 hours
        self._sectors_cache: Dict[str, Any] = {}
        self._last_sectors_fetch_time: float = 0.0

    def fetch_sectors_structure(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Fetch departments and sections hierarchy from jea_backend internal endpoint:
        GET /api/v1/employees/internal/sectors
        Caches result in-memory for TTL duration (default 24h).
        Falls back to DEFAULT_FALLBACK_SECTORS_STRUCTURE if backend is unavailable.
        """
        now = time.time()
        if (
            not force_refresh
            and self._sectors_cache
            and (now - self._last_sectors_fetch_time < self.sectors_ttl_seconds)
        ):
            return self._sectors_cache

        endpoints = [
            f"{self.base_url}/api/v1/employees/internal/sectors",
            f"{self.base_url}/employees/internal/sectors",
        ]

        for url in endpoints:
            try:
                client = get_sync_http_client(timeout=4.0)
                response = client.get(url, headers=self.headers)
                if response.status_code == 200:
                    envelope = response.json()
                    data = (
                        envelope.get("data", envelope)
                        if isinstance(envelope, dict) and "data" in envelope
                        else envelope
                    )
                    if isinstance(data, dict) and ("departments" in data or "sectors" in data):
                        self._sectors_cache = {
                            "sectors": data.get("sectors") or data.get("departments", []),
                            "departments": data.get("departments") or data.get("sectors", []),
                            "sections": data.get("sections", []),
                            "departmentSections": data.get("departmentSections", {}),
                        }
                        self._last_sectors_fetch_time = now
                        logger.info(
                            "[TicketingClient] Successfully fetched sectors structure: %d departments, %d sections",
                            len(self._sectors_cache["departments"]),
                            len(self._sectors_cache["sections"]),
                        )
                        return self._sectors_cache
            except Exception as e:
                logger.debug("[TicketingClient] Failed querying %s: %s", url, e)

        logger.warning(
            "[TicketingClient] Backend sectors endpoint unreachable; using built-in syndicate sectors fallback."
        )
        return self._sectors_cache or DEFAULT_FALLBACK_SECTORS_STRUCTURE

    def create_ticket(
        self,
        title: str,
        content: str,
        priority: str = "MEDIUM",
        session_id: Optional[str] = None,
        phone: Optional[str] = None,
        reason: Optional[str] = None,
        department: Optional[str] = None,
        section: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Submits an escalation ticket to POST /api/v1/tickets/internal on jea_backend.
        """
        url = f"{self.base_url}/api/v1/tickets/internal"

        prio_upper = priority.upper()
        if prio_upper not in ["HIGH", "MEDIUM", "LOW"]:
            prio_upper = "MEDIUM"

        final_content = content
        if reason and "سبب التصعيد" not in final_content:
            final_content = f"{final_content}\n\nسبب التصعيد: {reason}"

        payload: Dict[str, Any] = {
            "title": title[:100],
            "content": final_content,
            "ticketPriority": prio_upper,
            "skipWorkingHoursCheck": True,
        }
        import re
        uuid_re = re.compile(r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$')
        if session_id and uuid_re.match(session_id.strip()):
            payload["sessionId"] = session_id.strip()

        if phone:
            payload["userPhoneNumber"] = str(phone).strip()

        if department:
            payload["department"] = department.strip()
            payload["sector"] = department.strip()

        if section:
            payload["section"] = section.strip()


        try:
            client = get_sync_http_client(timeout=5.0)
            response = client.post(url, json=payload, headers=self.headers)
            if response.status_code in [200, 201]:
                envelope = response.json()
                ticket_data = (
                    envelope.get("data")
                    if isinstance(envelope, dict) and "data" in envelope
                    else envelope
                )
                if ticket_data:
                    ticket_id = ticket_data.get("id") or ticket_data.get("ticket_id")
                    logger.info(
                        "[TicketingClient] Successfully created ticket in backend: ID=%s",
                        ticket_id,
                    )
                return ticket_data
            else:
                logger.warning(
                    "[TicketingClient] Failed to create ticket: status=%s, body=%s",
                    response.status_code,
                    response.text[:250],
                )
        except Exception as e:
            logger.warning("[TicketingClient] Error calling backend internal ticket creation endpoint: %s", e)

        return None
