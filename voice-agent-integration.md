# JEA Voice Agent — Flutter Integration

The voice agent is Ahmad, a Jordanian male assistant for the Jordan Engineers Association. The Flutter app captures the microphone and plays speech. This server bridges that audio to GPT-Live-1 and answers from the guild knowledge base.

The app never talks to OpenAI and never holds `OPENAI_API_KEY`.

## What the agent can answer

Membership documents, subscriptions and fees, health insurance, the retirement fund, professional practice and complaints, engineering offices and companies, accreditation and training, divisions, branches, committees, and general syndicate information.

He speaks Jordanian Arabic by default and switches to English if the caller does. If he has no confirmed answer, or the caller asks for a person, he opens a support ticket. He does not ask for a name or phone number before opening it.

## Run the server

From `JEA_RAG`, with Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Keep `LLM_PROVIDER=openai`. Voice settings:

```
GPT_LIVE_MODEL=gpt-live-1
GPT_LIVE_VOICE=cedar
GPT_LIVE_DELEGATION_MODEL=gpt-6-luna
```

Index the knowledge base once, then start one worker:

```bash
python main.py ingest-texts
python main.py serve --port 8000 --workers 1
```

The first start downloads the embedding, rewriter, and reranker models and can take several minutes. Later starts are faster. Use `--workers 1`. More than one worker loads those models again in each process.

Check:

```bash
curl http://127.0.0.1:8000/api/voice/health
```

```json
{"status":"ok","model":"gpt-live-1","voice":"cedar","delegation_model":"gpt-6-luna"}
```

The browser test page is `http://127.0.0.1:8000/`.

| Device | Base URL |
|---|---|
| iOS Simulator, desktop | `http://127.0.0.1:8000` |
| Android Emulator | `http://10.0.2.2:8000` |
| Physical phone | `http://<your-computer-lan-ip>:8000` |

The phone and the computer must be on the same network. Android needs cleartext HTTP allowed for local development. iOS needs an App Transport Security exception for local HTTP.

## Connect

```
ws://<host>:8000/ws/voice?key=<USER_API_KEY>
```

Use `wss://` when the server is behind HTTPS. `USER_API_KEY` is the value in `.env` (the same key the chat UI sends as `X-API-Key`). A wrong key closes the socket with code `1008` before it is accepted.

Open the socket, start the microphone, and keep sending audio until the user hangs up. One socket is one call.

## Audio

Both directions are raw PCM, not a file and not WAV.

| | |
|---|---|
| Format | signed 16-bit little-endian |
| Rate | 24000 Hz |
| Channels | 1 (mono) |
| Transport | base64 inside a JSON text message |

Record at the device rate, then resample to 24000 Hz before sending. Play received audio as 24000 Hz PCM. Schedule each chunk to start when the previous chunk ends. Starting every chunk at “now” causes overlap and noise.

A useful send size is about 100–200 ms of audio (4096 samples at 24 kHz is about 170 ms).

## App → server

Microphone audio, sent continuously while the call is open:

```json
{"type":"audio","data":"<base64 PCM16>"}
```

Hang up:

```json
{"type":"end"}
```

Then close the socket.

## Server → app

Ready. Start sending audio after this:

```json
{"type":"status","status":"connected"}
```

Agent speech. Decode and queue for playback:

```json
{"type":"audio","data":"<base64 PCM16>"}
```

What was said. Optional for captions:

```json
{"type":"transcript","role":"user","text":"..."}
{"type":"transcript","role":"agent","text":"..."}
```

Verified written answer from the knowledge base. Show this text. `sources` are short document snippets. `ticket_id` is set only when a ticket was opened.

```json
{
  "type": "answer",
  "question": "شو الأوراق المطلوبة للانتساب؟",
  "answer": "...",
  "sources": [{"title": "شروط_تسجيل.md", "text": "..."}],
  "escalated": false,
  "ticket_id": null
}
```

The spoken reply and the `answer` text can differ slightly. Display `answer`. Play `audio`.

Failure. Show `error` and end the call:

```json
{"type":"status","status":"error","error":"..."}
```

Call finished:

```json
{"type":"status","status":"ended"}
```

## Flutter notes

- Request microphone permission before connecting.
- Use a PCM recorder (for example `record` with `pcm16bits`, or `flutter_sound`) and an PCM player that can stream (`just_audio` is a poor fit; `flutter_pcm_sound` or a platform `AudioTrack` / `AVAudioEngine` player is better).
- Convert float or other rates to mono PCM16 at 24000 Hz.
- Base64-encode each chunk and send it as a text WebSocket frame.
- Keep a playback clock: if the next start time is behind the current time, reset it to now, then start the buffer and add its duration.
- Do not send audio before `status: connected`.
- Ignore `transcript` if you only need the spoken call. Use `answer` when you want the written reply and sources on screen.
- The first question in a new server process can take longer while models sit in memory. Later questions are faster, but retrieval still takes several seconds. Keep the socket open and keep playing audio; do not timeout the call at 2–3 seconds.
- Echo cancellation and noise suppression should be on. Play agent audio through the speaker or earpiece, not back into the mic tap.

## Call flow

1. `GET /api/voice/health` to confirm the server is up.
2. Open `ws://<host>:8000/ws/voice?key=...`.
3. Wait for `{"type":"status","status":"connected"}`.
4. Ahmad greets in Arabic. Play the `audio` chunks.
5. Stream microphone PCM until the user stops.
6. On each question, the server may send `answer` (text and sources) and then more `audio` (the spoken reply).
7. Send `{"type":"end"}` and close the socket.
