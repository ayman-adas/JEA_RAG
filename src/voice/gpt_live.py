"""GPT-Live-1 WebRTC bridge for the JEA voice assistant.

Browser microphone audio arrives over a WebSocket as PCM 16-bit 24 kHz and is
forwarded to GPT-Live-1. Factual questions are delegated to gpt-6-luna, which
calls the existing RAG chatbot.
"""

import asyncio
import json
import logging
import time
from typing import Any

from src.config import (
    OPENAI_API_KEY,
    effective_delegation_model,
    effective_live_model,
    effective_live_voice,
)
from src.voice.prompts import GREETING, build_backend_prompt, build_system_prompt

logger = logging.getLogger(__name__)

GPT_LIVE_WS_URL = "wss://api.openai.com/v1/live/sessions"
USER_AGENT = "JEA-VoiceAgent/1.0"

ANSWER_TOOL = {
    "type": "function",
    "name": "answer_guild_question",
    "description": (
        "Look up a verified answer from the Jordan Engineers Association knowledge base. "
        "Use this for every question about membership, registration, fees, retirement, "
        "insurance, laws, complaints, offices, or guild services, and when the caller "
        "wants a support ticket. Do not answer those topics without this tool."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "A standalone question in the caller's language, including any "
                    "context required from earlier turns."
                ),
            },
        },
        "required": ["question"],
    },
}

DELEGATION_TOOLS = [ANSWER_TOOL]


def build_session_config(
    system_prompt: str,
    backend_prompt: str,
    audio_format: str = "audio/pcm",
    sample_rate: int = 24000,
) -> dict:
    """Build the GPT-Live-1 session.start payload."""
    return {
        "model": effective_live_model(),
        "instructions": system_prompt,
        "audio": {
            "format": {"type": audio_format, "rate": sample_rate},
            "output": {"voice": effective_live_voice()},
        },
        "delegation": {
            "type": "responses",
            "responses": {
                "model": effective_delegation_model(),
                "instructions": backend_prompt,
                "tools": DELEGATION_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            },
        },
    }


async def _send_json(ws, data: dict) -> None:
    try:
        await ws.send(json.dumps(data))
    except Exception as exc:
        logger.warning("Error sending to OpenAI WS: %s", exc)


def _source_payload(sources: list) -> list[dict]:
    items = []
    for source in (sources or [])[:5]:
        if not isinstance(source, dict):
            continue
        title = source.get("title") or source.get("file_name") or "وثيقة"
        text = source.get("text") or source.get("content") or ""
        items.append({"title": str(title), "text": str(text)[:240]})
    return items


async def handle_tool_call(
    openai_ws,
    call_id: str,
    fn_name: str,
    fn_args: dict,
    session_state: dict,
) -> None:
    """Run the RAG tool and return the verified answer to GPT-Live-1."""
    if fn_name != "answer_guild_question":
        logger.warning("Unknown tool: %s", fn_name)
        await _send_json(openai_ws, {
            "type": "response.item.create",
            "event_id": f"tool_unknown_{call_id}",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"error": f"Unknown tool: {fn_name}"}),
            },
        })
        await _send_json(openai_ws, {"type": "response.create", "event_id": f"continue_{call_id}"})
        return

    question = (fn_args.get("question") or "").strip()
    if not question:
        await _send_json(openai_ws, {
            "type": "response.item.create",
            "event_id": f"tool_empty_{call_id}",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": False, "error": "Missing question"}),
            },
        })
        await _send_json(openai_ws, {"type": "response.create", "event_id": f"continue_{call_id}"})
        return

    logger.info("RAG voice question: %s", question[:300])
    client_ws = session_state["client_ws"]

    try:
        from src.api.routes_chat import get_chatbot
        from src.core.rag_engine import clean_formatting

        history = list(session_state.get("history", [])[-8:])
        session_id = session_state.get("session_id")

        def _run_rag():
            return asyncio.run(
                get_chatbot().answer_question(
                    query=question,
                    history=history,
                    top_k=5,
                    session_id=session_id,
                )
            )

        # Keep the voice socket alive while retrieval and the answer model run.
        result = await asyncio.to_thread(_run_rag)
        answer = clean_formatting(result.get("answer") or "")
        escalated = bool(result.get("is_escalated") or result.get("escalated"))
        ticket_id = result.get("ticket_id")
        sources = _source_payload(result.get("sources") or [])

        session_state.setdefault("history", []).append({"role": "user", "content": question})
        session_state["history"].append({"role": "assistant", "content": answer[:1500]})

        tool_output = {
            "success": True,
            "answer": answer[:2500],
            "escalated": escalated,
            "ticket_id": ticket_id,
            "sources": [item["title"] for item in sources],
        }
        await _send_json(openai_ws, {
            "type": "response.item.create",
            "event_id": f"tool_result_{call_id}",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps(tool_output, ensure_ascii=False),
            },
        })
        await _send_json(openai_ws, {"type": "response.create", "event_id": f"continue_{call_id}"})

        try:
            await client_ws.send_json({
                "type": "answer",
                "question": question,
                "answer": answer,
                "sources": sources,
                "escalated": escalated,
                "ticket_id": ticket_id,
            })
        except Exception as exc:
            logger.info("Could not send answer to client: %s", exc)

        spoken = answer[:1200] if answer else "لا تتوفر إجابة مؤكدة حالياً."
        if escalated and ticket_id:
            spoken += f" تم فتح تذكرة دعم رقم {ticket_id}."

        await asyncio.sleep(0.3)
        await _send_json(openai_ws, {
            "type": "session.commentary.append",
            "event_id": f"speak_{call_id}",
            "delegation_id": None,
            "content": (
                "A verified knowledge-base answer is ready. Speak it now in the caller's "
                "language, in short natural sentences, as Ahmad, a Jordanian man. "
                "If Arabic, use masculine Ammani dialect. Do not mention tools or documents. "
                f"Answer to speak: {spoken}"
            ),
        })
        logger.info("Spoke RAG answer escalated=%s ticket=%s", escalated, ticket_id)

    except Exception as exc:
        logger.exception("RAG voice tool failed: %s", exc)
        await _send_json(openai_ws, {
            "type": "response.item.create",
            "event_id": f"tool_error_{call_id}",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": json.dumps({"success": False, "error": str(exc)}),
            },
        })
        await _send_json(openai_ws, {"type": "response.create", "event_id": f"continue_err_{call_id}"})
        await asyncio.sleep(0.3)
        await _send_json(openai_ws, {
            "type": "session.commentary.append",
            "event_id": f"speak_err_{call_id}",
            "delegation_id": None,
            "content": (
                "The knowledge base could not be reached. Apologize briefly and ask "
                "the caller to try the question again."
            ),
        })


async def handle_webrtc_session(
    client_ws,
    audio_format: str = "audio/pcm",
    sample_rate: int = 24000,
) -> None:
    """Bridge browser PCM audio to GPT-Live-1 and delegate answers to the RAG."""
    openai_ws = None
    session_state: dict[str, Any] = {
        "client_ws": client_ws,
        "history": [],
        "session_id": f"voice-{int(time.time())}",
    }
    tool_lock = asyncio.Lock()

    system_prompt = build_system_prompt()
    backend_prompt = build_backend_prompt()

    try:
        if not OPENAI_API_KEY:
            await client_ws.send_json({
                "type": "status",
                "status": "error",
                "error": "OPENAI_API_KEY is not configured",
            })
            return

        from websockets.asyncio.client import connect as ws_connect

        openai_ws = await ws_connect(
            GPT_LIVE_WS_URL,
            additional_headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "User-Agent": USER_AGENT,
            },
            ping_interval=20,
            ping_timeout=180,
        )
        logger.info(
            "Connected to GPT-Live-1 model=%s delegation=%s",
            effective_live_model(),
            effective_delegation_model(),
        )

        config = build_session_config(system_prompt, backend_prompt, audio_format, sample_rate)
        await _send_json(openai_ws, {"type": "session.start", "session": config})
        await client_ws.send_json({"type": "status", "status": "connected"})

        async def receive_from_client():
            try:
                async for message in client_ws.iter_text():
                    data = json.loads(message)
                    msg_type = data.get("type", "")

                    if msg_type == "audio":
                        audio_b64 = data.get("data", "")
                        if audio_b64:
                            await _send_json(openai_ws, {
                                "type": "session.input_audio.append",
                                "audio": audio_b64,
                            })
                    elif msg_type == "end":
                        logger.info("Client ended voice session")
                        break
            except Exception as exc:
                logger.info("Voice client disconnected: %s", exc)

        async def receive_from_openai():
            msg_count = 0
            try:
                async for raw_msg in openai_ws:
                    msg_count += 1
                    data = json.loads(raw_msg) if isinstance(raw_msg, str) else json.loads(raw_msg.decode())
                    event_type = data.get("type", "")

                    if event_type == "session.started":
                        logger.info("GPT-Live-1 session started: %s", data.get("session", {}).get("id"))
                        greeting = GREETING
                        await _send_json(openai_ws, {
                            "type": "session.commentary.append",
                            "event_id": "greeting",
                            "delegation_id": None,
                            "content": greeting,
                        })

                    elif event_type == "session.closed":
                        logger.info("GPT-Live-1 session closed usage=%s", data.get("usage", {}))
                        break

                    elif event_type == "session.output_audio.delta":
                        delta = data.get("delta", "")
                        if delta:
                            try:
                                await client_ws.send_json({"type": "audio", "data": delta})
                            except Exception:
                                break

                    elif event_type == "session.input_transcript.done":
                        transcript = data.get("transcript", "")
                        logger.info("Caller said: %r", transcript)
                        try:
                            await client_ws.send_json({
                                "type": "transcript",
                                "role": "user",
                                "text": transcript,
                            })
                        except Exception:
                            pass

                    elif event_type == "session.output_transcript.done":
                        transcript = data.get("transcript", "")
                        logger.info("Agent said: %r", transcript)
                        try:
                            await client_ws.send_json({
                                "type": "transcript",
                                "role": "agent",
                                "text": transcript,
                            })
                        except Exception:
                            pass

                    elif event_type == "response.event":
                        inner = data.get("event", {})
                        if inner.get("type") == "response.output_item.done":
                            item = inner.get("item", {})
                            if item.get("type") == "function_call" and item.get("status") == "completed":
                                cid = item.get("call_id", "")
                                fn_name = item.get("name", "")
                                raw_args = item.get("arguments", "{}")
                                if isinstance(raw_args, dict):
                                    fn_args = raw_args
                                else:
                                    try:
                                        fn_args = json.loads(raw_args or "{}")
                                    except (json.JSONDecodeError, TypeError):
                                        fn_args = {}
                                logger.info("Tool call: %s", fn_name)

                                async def _run_tool(cid=cid, fn_name=fn_name, fn_args=fn_args):
                                    async with tool_lock:
                                        await handle_tool_call(
                                            openai_ws, cid, fn_name, fn_args, session_state
                                        )

                                asyncio.create_task(_run_tool())

                    elif event_type == "error":
                        error = data.get("error", {})
                        logger.error("GPT-Live-1 error: %s", error)
                        try:
                            await client_ws.send_json({
                                "type": "status",
                                "status": "error",
                                "error": str(error),
                            })
                        except Exception:
                            pass

                    elif event_type == "session.commentary.appended":
                        logger.info("Commentary accepted")

            except Exception as exc:
                logger.exception("Error receiving from GPT-Live-1: %s", exc)
            finally:
                logger.info("receive_from_openai ended after %d messages", msg_count)

        client_task = asyncio.create_task(receive_from_client())
        openai_task = asyncio.create_task(receive_from_openai())
        done, pending = await asyncio.wait(
            {client_task, openai_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            exc = task.exception() if not task.cancelled() else None
            if exc:
                logger.error("Voice bridge task failed: %s", exc)

    except Exception as exc:
        logger.exception("Voice session error: %s", exc)
        message = str(exc)
        if "401" in message or "invalid_api_key" in message.lower():
            message = (
                "OpenAI rejected the API key. Update OPENAI_API_KEY in JEA_RAG/.env "
                "with a valid key, then restart the server."
            )
        try:
            await client_ws.send_json({"type": "status", "status": "error", "error": message})
        except Exception:
            pass
    finally:
        if openai_ws:
            try:
                await _send_json(openai_ws, {"type": "session.close"})
                await openai_ws.close()
            except Exception:
                pass
        try:
            await client_ws.send_json({"type": "status", "status": "ended"})
        except Exception:
            pass
        logger.info("Voice session ended")
