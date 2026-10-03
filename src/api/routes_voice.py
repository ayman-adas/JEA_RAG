"""WebRTC voice session routes for the JEA chatbot."""

import logging

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect

from src.api.dependencies import ADMIN_API_KEY, USER_API_KEY
from src.api.security import safe_compare
from src.config import effective_delegation_model, effective_live_model, effective_live_voice
from src.voice.gpt_live import handle_webrtc_session

logger = logging.getLogger(__name__)

router = APIRouter()


def _voice_key_ok(key: str) -> bool:
    if not USER_API_KEY and not ADMIN_API_KEY:
        return True
    if key and USER_API_KEY and safe_compare(key, USER_API_KEY):
        return True
    if key and ADMIN_API_KEY and safe_compare(key, ADMIN_API_KEY):
        return True
    return False


@router.get("/api/voice/health")
async def voice_health():
    return {
        "status": "ok",
        "model": effective_live_model(),
        "voice": effective_live_voice(),
        "delegation_model": effective_delegation_model(),
    }


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Browser microphone PCM ↔ GPT-Live-1, with RAG answers via gpt-6-luna."""
    key = websocket.query_params.get("key", "")
    if not _voice_key_ok(key):
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Voice client connected")
    try:
        await handle_webrtc_session(client_ws=websocket)
    except WebSocketDisconnect:
        logger.info("Voice client disconnected")
    except Exception as exc:
        logger.exception("Voice session error: %s", exc)
