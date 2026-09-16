import logging
import httpx
from typing import Dict, Any, Optional

from src.config import (
    JEA_BACKEND_BASE_URL,
    INTERNAL_BYPASS_TOKEN,
)
from src.services.http_client_pool import get_sync_http_client, get_async_http_client

logger = logging.getLogger(__name__)


class BlackMessageClient:
    """
    Isolated HTTP client for the JEA Backend Live Chat Internal Black Message Moderation APIs.

    Connects to:
      - PATCH /api/v1/live-chat/messages/:messageId/black/internal
      - PATCH /api/v1/live-chat/sessions/:sessionId/messages/:messageId/black/internal
      - POST  /api/v1/live-chat/messages/black/internal

    Used by AI / RAG moderation pipelines or quality guardrails to flag or unflag
    a message as a black message (isBlackMessage: boolean) without touching or opening tickets.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        internal_token: Optional[str] = None,
        timeout: float = 5.0,
    ):
        raw_url = base_url or JEA_BACKEND_BASE_URL or "http://localhost:3000"
        # If full /api/... path was supplied, strip down to base domain URL
        if "/api/" in raw_url:
            raw_url = raw_url.split("/api/")[0]
        self.base_url = raw_url.rstrip("/")

        self.token = internal_token or INTERNAL_BYPASS_TOKEN or "jea_rag_token"
        self.timeout = timeout
        self.headers = {
            "x-internal-token": self.token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _resolve_url(
        self,
        message_id: str,
        session_id: Optional[str] = None,
    ) -> str:
        """Constructs the standard or session-scoped internal PATCH URL."""
        if session_id and session_id.strip():
            return f"{self.base_url}/api/v1/live-chat/sessions/{session_id.strip()}/messages/{message_id.strip()}/black/internal"
        return f"{self.base_url}/api/v1/live-chat/messages/{message_id.strip()}/black/internal"

    def mark_black_message(
        self,
        message_id: str,
        is_black: bool = True,
        reason: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Synchronously flags or unflags a message as a black message using connection pool.
        """
        url = self._resolve_url(message_id, session_id)
        payload: Dict[str, Any] = {
            "isBlackMessage": is_black,
        }
        if reason:
            payload["reason"] = str(reason)[:500]

        try:
            client = get_sync_http_client(timeout=self.timeout)
            response = client.patch(url, json=payload, headers=self.headers)
            if response.status_code == 200:
                envelope = response.json()
                data = envelope.get("data") if isinstance(envelope, dict) and "data" in envelope else envelope
                logger.info(
                    "[BlackMessageClient] Successfully set isBlackMessage=%s on message %s",
                    is_black,
                    message_id,
                )
                return data
            elif response.status_code == 401:
                logger.error(
                    "[BlackMessageClient] Unauthorized (401) calling %s - check INTERNAL_BYPASS_TOKEN",
                    url,
                )
            elif response.status_code == 404:
                logger.warning(
                    "[BlackMessageClient] Message %s not found (404) in live chat database",
                    message_id,
                )
            else:
                logger.warning(
                    "[BlackMessageClient] Error status %s from %s: %s",
                    response.status_code,
                    url,
                    response.text[:200],
                )
        except Exception as e:
            logger.warning("[BlackMessageClient] Connection error calling %s: %s", url, e)

        return None

    async def mark_black_message_async(
        self,
        message_id: str,
        is_black: bool = True,
        reason: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Asynchronously flags or unflags a message as a black message using connection pool.
        """
        url = self._resolve_url(message_id, session_id)
        payload: Dict[str, Any] = {
            "isBlackMessage": is_black,
        }
        if reason:
            payload["reason"] = str(reason)[:500]

        try:
            client = get_async_http_client(timeout=self.timeout)
            response = await client.patch(url, json=payload, headers=self.headers)
            if response.status_code == 200:
                envelope = response.json()
                data = envelope.get("data") if isinstance(envelope, dict) and "data" in envelope else envelope
                logger.info(
                    "[BlackMessageClient] Successfully (async) set isBlackMessage=%s on message %s",
                    is_black,
                    message_id,
                )
                return data
            elif response.status_code == 401:
                logger.error(
                    "[BlackMessageClient] Unauthorized (401) calling %s - check INTERNAL_BYPASS_TOKEN",
                    url,
                )
            elif response.status_code == 404:
                logger.warning(
                    "[BlackMessageClient] Message %s not found (404) in live chat database",
                    message_id,
                )
            else:
                logger.warning(
                    "[BlackMessageClient] Error status %s from %s: %s",
                    response.status_code,
                    url,
                    response.text[:200],
                )
        except Exception as e:
            logger.warning("[BlackMessageClient] Async connection error calling %s: %s", url, e)

        return None

    def unflag_message(
        self,
        message_id: str,
        reason: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Convenience method to unflag / mark isBlackMessage=False."""
        return self.mark_black_message(
            message_id=message_id,
            is_black=False,
            reason=reason,
            session_id=session_id,
        )

    async def unflag_message_async(
        self,
        message_id: str,
        reason: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Async convenience method to unflag / mark isBlackMessage=False."""
        return await self.mark_black_message_async(
            message_id=message_id,
            is_black=False,
            reason=reason,
            session_id=session_id,
        )

    def mark_black_message_by_body(
        self,
        message_id: str,
        is_black: bool = True,
        reason: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Flags or unflags via POST /api/v1/live-chat/messages/black/internal
        where messageId is passed in the JSON payload body.
        """
        url = f"{self.base_url}/api/v1/live-chat/messages/black/internal"
        payload: Dict[str, Any] = {
            "messageId": message_id.strip(),
            "isBlackMessage": is_black,
        }
        if reason:
            payload["reason"] = str(reason)[:500]

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(url, json=payload, headers=self.headers)
                if response.status_code == 200:
                    envelope = response.json()
                    return envelope.get("data") if isinstance(envelope, dict) and "data" in envelope else envelope
                logger.warning(
                    "[BlackMessageClient] Error status %s from POST %s: %s",
                    response.status_code,
                    url,
                    response.text[:200],
                )
        except Exception as e:
            logger.warning("[BlackMessageClient] Connection error calling POST %s: %s", url, e)

        return None
