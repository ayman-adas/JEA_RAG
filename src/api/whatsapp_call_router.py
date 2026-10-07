"""FastAPI router for Meta WhatsApp Calling Webhooks (Option 3 - Native WebRTC).

Handles Webhook verification (GET), call signaling offers (POST), and call state transitions.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse

from src.config import (
    WHATSAPP_APP_SECRET,
    WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_WEBHOOK_VERIFY_TOKEN,
)
from src.voice.whatsapp_webrtc import get_whatsapp_webrtc_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/whatsapp", tags=["WhatsApp Calling"])


def verify_meta_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """Verifies HMAC-SHA256 signature from Meta if WHATSAPP_APP_SECRET is set."""
    if not WHATSAPP_APP_SECRET:
        return True
    if not signature_header:
        logger.warning("Missing X-Hub-Signature-256 header")
        return False

    prefix = "sha256="
    if not signature_header.startswith(prefix):
        return False

    received_sig = signature_header[len(prefix):]
    expected_sig = hmac.new(
        WHATSAPP_APP_SECRET.encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(received_sig, expected_sig)


@router.get("/calls", response_class=PlainTextResponse)
@router.get("/webhook", response_class=PlainTextResponse)
async def verify_whatsapp_webhook(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
):
    """Meta webhook verification endpoint.
    
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge
    when configuring the webhook in the Meta Developer Portal.
    """
    logger.info("WhatsApp webhook verification received: mode=%s", hub_mode)

    if hub_mode == "subscribe":
        if not WHATSAPP_WEBHOOK_VERIFY_TOKEN or hub_verify_token == WHATSAPP_WEBHOOK_VERIFY_TOKEN:
            logger.info("WhatsApp webhook verified successfully")
            return PlainTextResponse(content=hub_challenge or "", status_code=status.HTTP_200_OK)
        else:
            logger.warning("WhatsApp webhook token mismatch: %s != %s", hub_verify_token, WHATSAPP_WEBHOOK_VERIFY_TOKEN)
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Verification token mismatch")

    return PlainTextResponse(content="WhatsApp VoIP calling endpoint active", status_code=status.HTTP_200_OK)


@router.post("/calls")
@router.post("/webhook")
async def handle_whatsapp_webhook(
    request: Request,
    x_hub_signature_256: Optional[str] = Header(None, alias="x-hub-signature-256"),
):
    """Processes inbound Meta WhatsApp webhook events.
    
    Extracts call events (connect, offer, terminate) and delegates
    to the WhatsAppWebRTCManager.
    """
    raw_body = await request.body()

    # Validate HMAC signature if app secret is provided
    if not verify_meta_signature(raw_body, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    try:
        payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except Exception as exc:
        logger.error("Failed to parse WhatsApp webhook JSON: %s", exc)
        return {"status": "error", "message": "Invalid JSON"}

    manager = get_whatsapp_webrtc_manager()

    # Direct signaling from jea_backend or test runner
    if "call_id" in payload and "sdp" in payload:
        call_id = payload["call_id"]
        caller_number = payload.get("from", "unknown")
        offer_sdp = payload["sdp"]
        try:
            sdp_answer = await manager.start_session(call_id, caller_number, offer_sdp)
            return {"status": "ok", "call_id": call_id, "sdp": sdp_answer}
        except Exception as exc:
            logger.exception("Failed to negotiate WebRTC SDP answer for call %s: %s", call_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

    if "call_id" in payload and payload.get("event") in ["terminate", "hungup", "missed", "rejected"]:
        call_id = payload["call_id"]
        await manager.end_call(call_id)
        return {"status": "ok", "call_id": call_id, "message": "Call session ended"}

    # Standard Meta Webhook payload parsing
    entry_list = payload.get("entry", [])
    call_events_found = 0

    for entry in entry_list:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            calls = value.get("calls", [])
            for call_data in calls:
                call_events_found += 1
                logger.info(
                    "Extracted WhatsApp call event: id=%s, event=%s, from=%s",
                    call_data.get("id"),
                    call_data.get("event"),
                    call_data.get("from"),
                )
                asyncio.create_task(manager.handle_inbound_call(call_data))

    logger.info("Processed WhatsApp webhook with %d call events", call_events_found)
    return {"status": "ok", "calls_processed": call_events_found}


@router.get("/calls/health")
async def whatsapp_calling_health():
    """Health check for WhatsApp Calling VoIP subsystem."""
    manager = get_whatsapp_webrtc_manager()
    return {
        "status": "ok",
        "configured": bool(WHATSAPP_PHONE_NUMBER_ID),
        "active_calls_count": len(manager.active_sessions),
        "active_call_ids": list(manager.active_sessions.keys()),
    }


@router.post("/calls/end")
async def end_whatsapp_call(call_id: str = Query(..., description="ID of the call to terminate")):
    """Manually ends a WhatsApp VoIP call."""
    manager = get_whatsapp_webrtc_manager()
    await manager.end_call(call_id)
    return {"status": "ok", "call_id": call_id, "message": "Call terminated"}
