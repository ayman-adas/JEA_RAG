"""WhatsApp VoIP Calling WebRTC Manager for JEA Chatbot (Option 3 - Native aiortc).

Directly connects Meta WhatsApp Cloud Calling API with GPT-Live-1 voice agent
and JEA RAG knowledge base via native WebRTC peer connections.
"""

import asyncio
import base64
import json
import logging
import time
from fractions import Fraction
from typing import Any, Dict, Optional

import av
import httpx
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)

from src.config import (
    OPENAI_API_KEY,
    WHATSAPP_APP_SECRET,
    WHATSAPP_GRAPH_API_BASE_URL,
    WHATSAPP_GRAPH_API_VERSION,
    WHATSAPP_PHONE_NUMBER_ID,
    WHATSAPP_STUN_SERVER,
    WHATSAPP_TOKEN,
    WHATSAPP_TURN_CREDENTIAL,
    WHATSAPP_TURN_SERVER,
    WHATSAPP_TURN_USERNAME,
    effective_delegation_model,
    effective_live_model,
    effective_live_voice,
)
from src.voice.gpt_live import (
    GPT_LIVE_WS_URL,
    USER_AGENT,
    _send_json,
    build_session_config,
    handle_tool_call,
)
from src.voice.prompts import GREETING, build_backend_prompt, build_system_prompt

logger = logging.getLogger(__name__)

AUDIO_SAMPLE_RATE = 24000  # 24 kHz mono PCM16
AUDIO_CHANNELS = 1
SAMPLES_PER_FRAME = 480  # 20ms of audio at 24000 Hz (480 samples * 2 bytes = 960 bytes)
BYTES_PER_SAMPLE = 2
FRAME_BYTES = SAMPLES_PER_FRAME * BYTES_PER_SAMPLE * AUDIO_CHANNELS  # 960 bytes


class WhatsAppAudioOutputTrack(MediaStreamTrack):
    """WebRTC audio track that streams assistant audio to WhatsApp caller.

    Reads 24 kHz PCM16 mono chunks received from OpenAI Realtime (or TTS),
    buffers them into 20ms frames, and emits av.AudioFrame instances.
    When idle, it emits silence frames to maintain RTP pacing.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = AUDIO_SAMPLE_RATE):
        super().__init__()
        self.sample_rate = sample_rate
        self.samples_per_frame = SAMPLES_PER_FRAME
        self._timestamp = 0
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._buffer = bytearray()
        self._is_active = True

    def put_pcm_chunk(self, chunk: bytes) -> None:
        """Feed raw 24kHz PCM16 mono bytes into the playback queue."""
        if self._is_active and chunk:
            self._audio_queue.put_nowait(chunk)

    def stop(self) -> None:
        self._is_active = False
        super().stop()

    async def recv(self) -> av.AudioFrame:
        """Called by aiortc RTP sender for each audio packet (every 20ms)."""
        await asyncio.sleep(0.020)

        while not self._audio_queue.empty():
            try:
                chunk = self._audio_queue.get_nowait()
                self._buffer.extend(chunk)
            except asyncio.QueueEmpty:
                break

        if len(self._buffer) >= FRAME_BYTES:
            frame_data = bytes(self._buffer[:FRAME_BYTES])
            del self._buffer[:FRAME_BYTES]
        else:
            frame_data = b'\x00' * FRAME_BYTES

        frame = av.AudioFrame(format="s16", layout="mono", samples=self.samples_per_frame)
        frame.planes[0].update(frame_data)
        frame.sample_rate = self.sample_rate
        frame.pts = self._timestamp
        frame.time_base = Fraction(1, self.sample_rate)
        self._timestamp += self.samples_per_frame

        return frame


class WhatsAppCallingClient:
    """HTTP client for Meta Graph API WhatsApp Calling endpoints."""

    def __init__(
        self,
        token: str = WHATSAPP_TOKEN,
        phone_number_id: str = WHATSAPP_PHONE_NUMBER_ID,
        api_version: str = WHATSAPP_GRAPH_API_VERSION,
        base_url: str = WHATSAPP_GRAPH_API_BASE_URL,
    ):
        self.token = token
        self.phone_number_id = phone_number_id
        self.api_version = api_version
        self.base_url = base_url.rstrip("/")
        self.endpoint = f"{self.base_url}/{self.api_version}/{self.phone_number_id}/calls"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def pre_accept_call(self, call_id: str, sdp_answer: str) -> dict:
        """Pre-accept an incoming call with the WebRTC SDP answer."""
        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "pre_accept",
            "session": {
                "sdp_type": "answer",
                "sdp": sdp_answer,
            },
        }
        logger.info("Calling Meta API: pre_accept for call_id=%s", call_id)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
            if resp.status_code >= 400:
                logger.error("Meta pre_accept failed [%d]: %s", resp.status_code, resp.text)
            return resp.json() if resp.content else {"status_code": resp.status_code}

    async def accept_call(self, call_id: str) -> dict:
        """Officially accept the pre-accepted call."""
        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "accept",
        }
        logger.info("Calling Meta API: accept for call_id=%s", call_id)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
            if resp.status_code >= 400:
                logger.error("Meta accept failed [%d]: %s", resp.status_code, resp.text)
            return resp.json() if resp.content else {"status_code": resp.status_code}

    async def terminate_call(self, call_id: str) -> dict:
        """Hang up / terminate the call."""
        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "terminate",
        }
        logger.info("Calling Meta API: terminate for call_id=%s", call_id)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
            return resp.json() if resp.content else {"status_code": resp.status_code}

    async def reject_call(self, call_id: str) -> dict:
        """Reject an incoming call."""
        payload = {
            "messaging_product": "whatsapp",
            "call_id": call_id,
            "action": "reject",
        }
        logger.info("Calling Meta API: reject for call_id=%s", call_id)
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(self.endpoint, headers=self._headers(), json=payload)
            return resp.json() if resp.content else {"status_code": resp.status_code}


class WhatsAppCallSession:
    """Manages a single active WhatsApp VoIP call session."""

    def __init__(
        self,
        call_id: str,
        caller_number: str,
        api_client: WhatsAppCallingClient,
    ):
        self.call_id = call_id
        self.caller_number = caller_number
        self.api_client = api_client
        self.pc: Optional[RTCPeerConnection] = None
        self.output_track: Optional[WhatsAppAudioOutputTrack] = None
        self.openai_ws = None
        self.tasks: list[asyncio.Task] = []
        self.is_closed = False
        self.session_state: dict[str, Any] = {
            "history": [],
            "session_id": f"wa-{call_id}-{int(time.time())}",
            "client_ws": self,
        }
        self.tool_lock = asyncio.Lock()

    async def send_json(self, data: dict) -> None:
        """Duck-typed interface for handle_tool_call logging/transcripts."""
        msg_type = data.get("type")
        if msg_type == "answer":
            logger.info(
                "[WhatsApp VoIP %s] RAG Answer: %s (escalated=%s, ticket=%s)",
                self.call_id,
                data.get("answer", "")[:120],
                data.get("escalated"),
                data.get("ticket_id"),
            )
        elif msg_type == "transcript":
            logger.info(
                "[WhatsApp VoIP %s] %s: %s",
                self.call_id,
                data.get("role"),
                data.get("text"),
            )

    async def start(self, offer_sdp: str) -> str:
        """Set up WebRTC connection, negotiate SDP answer, and connect to OpenAI Live."""
        ice_servers = [RTCIceServer(urls=[WHATSAPP_STUN_SERVER])]
        if WHATSAPP_TURN_SERVER:
            ice_servers.append(
                RTCIceServer(
                    urls=[WHATSAPP_TURN_SERVER],
                    username=WHATSAPP_TURN_USERNAME or None,
                    credential=WHATSAPP_TURN_CREDENTIAL or None,
                )
            )

        rtc_config = RTCConfiguration(iceServers=ice_servers)
        self.pc = RTCPeerConnection(configuration=rtc_config)

        self.output_track = WhatsAppAudioOutputTrack(sample_rate=AUDIO_SAMPLE_RATE)
        self.pc.addTrack(self.output_track)

        @self.pc.on("track")
        def on_track(track):
            logger.info("[WhatsApp VoIP %s] Received track: kind=%s", self.call_id, track.kind)
            if track.kind == "audio":
                task = asyncio.create_task(self._process_incoming_audio(track))
                self.tasks.append(task)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("[WhatsApp VoIP %s] Connection state is %s", self.call_id, self.pc.connectionState)
            if self.pc.connectionState in ["failed", "closed"]:
                await self.close()

        offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
        await self.pc.setRemoteDescription(offer)

        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)

        sdp_answer = self.pc.localDescription.sdp
        if "a=fingerprint:SHA-256" in sdp_answer:
            sdp_answer = sdp_answer.replace("a=fingerprint:SHA-256", "a=fingerprint:sha-256")

        openai_task = asyncio.create_task(self._connect_openai_live())
        self.tasks.append(openai_task)

        return sdp_answer

    async def _connect_openai_live(self) -> None:
        """Connects to OpenAI Realtime GPT-Live-1 WebSocket and handles events."""
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set for WhatsApp VoIP call")
            return

        from websockets.asyncio.client import connect as ws_connect

        system_prompt = build_system_prompt()
        backend_prompt = build_backend_prompt()

        try:
            self.openai_ws = await ws_connect(
                GPT_LIVE_WS_URL,
                additional_headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "User-Agent": USER_AGENT,
                },
                ping_interval=20,
                ping_timeout=180,
            )
            logger.info(
                "[WhatsApp VoIP %s] Connected to GPT-Live-1 model=%s",
                self.call_id,
                effective_live_model(),
            )

            config = build_session_config(
                system_prompt,
                backend_prompt,
                audio_format="audio/pcm",
                sample_rate=AUDIO_SAMPLE_RATE,
            )
            await _send_json(self.openai_ws, {"type": "session.start", "session": config})

            async for raw_msg in self.openai_ws:
                if self.is_closed:
                    break
                data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                event_type = data.get("type", "")

                if event_type == "session.started":
                    logger.info("[WhatsApp VoIP %s] GPT-Live-1 session started: %s", self.call_id, data.get("session", {}).get("id"))
                    await _send_json(self.openai_ws, {
                        "type": "session.commentary.append",
                        "event_id": "greeting",
                        "delegation_id": None,
                        "content": GREETING,
                    })

                elif event_type == "session.output_audio.delta":
                    delta_b64 = data.get("delta", "")
                    if delta_b64 and self.output_track:
                        raw_pcm = base64.b64decode(delta_b64)
                        self.output_track.put_pcm_chunk(raw_pcm)

                elif event_type == "session.input_transcript.done":
                    transcript = data.get("transcript", "")
                    logger.info("[WhatsApp VoIP %s] Caller [%s]: %r", self.call_id, self.caller_number, transcript)

                elif event_type == "session.output_transcript.done":
                    transcript = data.get("transcript", "")
                    logger.info("[WhatsApp VoIP %s] Ahmad (agent): %r", self.call_id, transcript)

                elif event_type == "response.event":
                    inner = data.get("event", {})
                    if inner.get("type") == "response.output_item.done":
                        item = inner.get("item", {})
                        if item.get("type") == "function_call" and item.get("status") == "completed":
                            cid = item.get("call_id", "")
                            fn_name = item.get("name", "")
                            raw_args = item.get("arguments", "{}")
                            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                            logger.info("[WhatsApp VoIP %s] Tool call triggered: %s", self.call_id, fn_name)

                            async def _run_tool(cid=cid, fn_name=fn_name, fn_args=fn_args):
                                async with self.tool_lock:
                                    await handle_tool_call(
                                        self.openai_ws, cid, fn_name, fn_args, self.session_state
                                    )

                            asyncio.create_task(_run_tool())

                elif event_type == "session.closed":
                    logger.info("[WhatsApp VoIP %s] OpenAI session closed", self.call_id)
                    break

        except Exception as exc:
            logger.exception("[WhatsApp VoIP %s] OpenAI Live bridge error: %s", self.call_id, exc)
        finally:
            await self.close()

    async def _process_incoming_audio(self, track: MediaStreamTrack) -> None:
        """Reads audio frames from WhatsApp caller, resamples to 24kHz mono PCM16, and sends to OpenAI."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=AUDIO_SAMPLE_RATE)
        logger.info("[WhatsApp VoIP %s] Started incoming audio processing", self.call_id)

        try:
            while not self.is_closed:
                frame = await track.recv()
                resampled_frames = resampler.resample(frame)
                for rframe in resampled_frames:
                    pcm_bytes = bytes(rframe.planes[0])
                    if pcm_bytes and self.openai_ws:
                        b64_audio = base64.b64encode(pcm_bytes).decode("ascii")
                        await _send_json(self.openai_ws, {
                            "type": "session.input_audio.append",
                            "audio": b64_audio,
                        })
        except Exception as exc:
            if not self.is_closed:
                logger.info("[WhatsApp VoIP %s] Incoming track ended: %s", self.call_id, exc)

    async def close(self) -> None:
        """Cleanly terminate the call session and release all WebRTC & WebSocket resources."""
        if self.is_closed:
            return
        self.is_closed = True
        logger.info("[WhatsApp VoIP %s] Closing session", self.call_id)

        if self.output_track:
            self.output_track.stop()

        for t in self.tasks:
            t.cancel()

        if self.openai_ws:
            try:
                await _send_json(self.openai_ws, {"type": "session.close"})
                await self.openai_ws.close()
            except Exception:
                pass

        if self.pc:
            try:
                await self.pc.close()
            except Exception:
                pass


class WhatsAppWebRTCManager:
    """Singleton manager coordinating all active WhatsApp VoIP call sessions."""

    def __init__(self):
        self.api_client = WhatsAppCallingClient()
        self.active_sessions: dict[str, WhatsAppCallSession] = {}
        self._lock = asyncio.Lock()

    async def start_session(self, call_id: str, caller_number: str, offer_sdp: str) -> str:
        """Starts WebRTC session for call_id and returns negotiated SDP answer."""
        async with self._lock:
            if call_id in self.active_sessions:
                await self.active_sessions[call_id].close()

            session = WhatsAppCallSession(call_id, caller_number, self.api_client)
            self.active_sessions[call_id] = session

        return await session.start(offer_sdp)

    async def handle_inbound_call(self, call_data: dict) -> bool:
        """Handles an incoming call event from Meta's webhook.

        Performs WebRTC SDP negotiation, pre_accept, and accept.
        """
        call_id = call_data.get("id") or call_data.get("call_id")
        caller_number = call_data.get("from", "unknown")
        event = call_data.get("event", "connect")

        offer_sdp = (
            call_data.get("sdp")
            or (call_data.get("session", {}) or {}).get("sdp")
        )

        if not call_id:
            logger.warning("Inbound call event missing call_id: %s", call_data)
            return False

        if event in ["terminate", "hungup", "missed", "rejected"]:
            logger.info("Call termination event received for call_id=%s (event=%s)", call_id, event)
            await self.end_call(call_id)
            return True

        if not offer_sdp:
            logger.warning("Inbound call %s has no SDP offer", call_id)
            return False

        logger.info("Processing inbound call_id=%s from %s", call_id, caller_number)

        async with self._lock:
            if call_id in self.active_sessions:
                await self.active_sessions[call_id].close()

            session = WhatsAppCallSession(call_id, caller_number, self.api_client)
            self.active_sessions[call_id] = session

        try:
            sdp_answer = await session.start(offer_sdp)

            pre_accept_res = await self.api_client.pre_accept_call(call_id, sdp_answer)
            logger.info("pre_accept response for call %s: %s", call_id, pre_accept_res)

            await asyncio.sleep(0.1)
            accept_res = await self.api_client.accept_call(call_id)
            logger.info("accept response for call %s: %s", call_id, accept_res)

            return True
        except Exception as exc:
            logger.exception("Failed to accept inbound call %s: %s", call_id, exc)
            await self.api_client.reject_call(call_id)
            await self.end_call(call_id)
            return False

    async def end_call(self, call_id: str) -> None:
        """Ends an active call session."""
        session = None
        async with self._lock:
            session = self.active_sessions.pop(call_id, None)

        if session:
            await session.close()
            logger.info("Ended call session %s", call_id)

    async def shutdown(self) -> None:
        """Closes all active call sessions gracefully."""
        async with self._lock:
            sessions = list(self.active_sessions.values())
            self.active_sessions.clear()

        for s in sessions:
            await s.close()
        logger.info("Shutdown all WhatsApp VoIP call sessions")


_whatsapp_webrtc_manager: Optional[WhatsAppWebRTCManager] = None


def get_whatsapp_webrtc_manager() -> WhatsAppWebRTCManager:
    global _whatsapp_webrtc_manager
    if _whatsapp_webrtc_manager is None:
        _whatsapp_webrtc_manager = WhatsAppWebRTCManager()
    return _whatsapp_webrtc_manager
