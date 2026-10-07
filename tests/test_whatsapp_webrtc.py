"""Unit and Integration Tests for WhatsApp VoIP Calling (Option 3 - Native WebRTC)."""

import asyncio
import hashlib
import hmac
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.voice.whatsapp_webrtc import (
    WhatsAppAudioOutputTrack,
    WhatsAppCallingClient,
    get_whatsapp_webrtc_manager,
    AUDIO_SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    FRAME_BYTES,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_whatsapp_webhook_get_verification_success(client):
    """GET webhook verification succeeds when mode=subscribe and token matches."""
    with patch("src.api.whatsapp_call_router.WHATSAPP_WEBHOOK_VERIFY_TOKEN", "my_secret_verify_token"):
        response = client.get(
            "/api/whatsapp/calls",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "my_secret_verify_token",
                "hub.challenge": "1122334455",
            },
        )
        assert response.status_code == 200
        assert response.text == "1122334455"


def test_whatsapp_webhook_get_verification_failure(client):
    """GET webhook verification returns 403 when token does not match."""
    with patch("src.api.whatsapp_call_router.WHATSAPP_WEBHOOK_VERIFY_TOKEN", "correct_token"):
        response = client.get(
            "/api/whatsapp/calls",
            params={
                "hub.mode": "subscribe",
                "hub.verify_token": "wrong_token",
                "hub.challenge": "1122334455",
            },
        )
        assert response.status_code == 403


def test_whatsapp_webhook_signature_validation(client):
    """POST webhook validates HMAC SHA-256 signature when WHATSAPP_APP_SECRET is set."""
    app_secret = "test_meta_app_secret"
    payload = {"entry": []}
    raw_body = json.dumps(payload).encode("utf-8")

    # Correct signature
    valid_sig = "sha256=" + hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()

    with patch("src.api.whatsapp_call_router.WHATSAPP_APP_SECRET", app_secret):
        # Valid signature succeeds
        resp = client.post(
            "/api/whatsapp/calls",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-hub-signature-256": valid_sig},
        )
        assert resp.status_code == 200

        # Invalid signature fails with 401
        bad_resp = client.post(
            "/api/whatsapp/calls",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-hub-signature-256": "sha256=invalid"},
        )
        assert bad_resp.status_code == 401


def test_whatsapp_webhook_post_extracts_call_events(client):
    """POST webhook extracts incoming call offer events and forwards to manager."""
    meta_call_payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "123456",
                "changes": [
                    {
                        "field": "calls",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "987654321",
                            },
                            "calls": [
                                {
                                    "id": "wacid.test_call_999",
                                    "from": "962790000000",
                                    "timestamp": "1710000000",
                                    "event": "connect",
                                    "sdp": "v=0\r\no=- 123 456 IN IP4 0.0.0.0\r\n",
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }

    with patch("src.api.whatsapp_call_router.WHATSAPP_APP_SECRET", ""):
        with patch.object(
            get_whatsapp_webrtc_manager(),
            "handle_inbound_call",
            new_callable=AsyncMock,
        ) as mock_handle:
            mock_handle.return_value = True
            resp = client.post(
                "/api/whatsapp/calls",
                json=meta_call_payload,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("calls_processed") == 1


def test_audio_output_track_framing():
    """WhatsAppAudioOutputTrack accurately outputs 20ms frames and generates silence when idle."""
    async def _run():
        track = WhatsAppAudioOutputTrack(sample_rate=AUDIO_SAMPLE_RATE)
        assert track.kind == "audio"
        assert track.sample_rate == 24000
        assert track.samples_per_frame == SAMPLES_PER_FRAME

        # 1. Idle state -> returns silence frame
        frame = await track.recv()
        assert frame.sample_rate == 24000
        assert frame.samples == SAMPLES_PER_FRAME
        plane_data = bytes(frame.planes[0])
        assert len(plane_data) == FRAME_BYTES
        assert plane_data == b"\x00" * FRAME_BYTES

        # 2. Feed PCM data -> returns fed audio
        custom_audio = b"\x01\x02" * SAMPLES_PER_FRAME
        track.put_pcm_chunk(custom_audio)

        frame2 = await track.recv()
        assert bytes(frame2.planes[0]) == custom_audio

        track.stop()

    asyncio.run(_run())


def test_whatsapp_calling_client_payloads():
    """WhatsAppCallingClient generates correct Graph API payloads for pre_accept, accept, terminate."""
    async def _run():
        client = WhatsAppCallingClient(
            token="test_token",
            phone_number_id="test_phone_id",
            api_version="v21.0",
            base_url="https://graph.facebook.com",
        )
        assert client.endpoint == "https://graph.facebook.com/v21.0/test_phone_id/calls"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'{"success": true}'
            mock_resp.json.return_value = {"success": True}
            mock_post.return_value = mock_resp

            # Test pre_accept
            await client.pre_accept_call("call_123", "v=0\r\no=answer...")
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["action"] == "pre_accept"
            assert kwargs["json"]["call_id"] == "call_123"
            assert kwargs["json"]["session"]["sdp"] == "v=0\r\no=answer..."

            # Test accept
            await client.accept_call("call_123")
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["action"] == "accept"
            assert kwargs["json"]["call_id"] == "call_123"

            # Test terminate
            await client.terminate_call("call_123")
            args, kwargs = mock_post.call_args
            assert kwargs["json"]["action"] == "terminate"
            assert kwargs["json"]["call_id"] == "call_123"

    asyncio.run(_run())
