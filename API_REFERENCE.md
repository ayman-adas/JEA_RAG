# API Reference

Every HTTP endpoint the FastAPI app exposes (`src/api/main.py` + its routers), what each one is for, and what it expects/returns. This is the authoritative endpoint list — the API section in `README.md` predates the ticketing/user-identity changes and is out of date; treat this file as current.

Base URL (local): `http://localhost:8000`

---

## Authentication

All protected routes take the key in a header:

```http
X-API-Key: <key>
```

Two keys exist, set via env vars (`ADMIN_API_KEY`, `USER_API_KEY`), checked in `src/api/dependencies.py`:

| Dependency | Used by | Behavior |
|---|---|---|
| `verify_user_key` | chat/search routes | Accepts either `USER_API_KEY` or `ADMIN_API_KEY`. If **neither** env var is set, the check is skipped entirely (open dev mode) — set at least one before exposing the API beyond localhost. |
| `verify_admin_key` | admin routes | Requires `ADMIN_API_KEY` to be set and matched exactly; always rejects (`403`) otherwise. |

CORS is controlled by `CORS_ORIGINS` (comma-separated list); defaults to `*` if unset ([main.py:34-39](src/api/main.py#L34-L39)).

---

## Chat & Search — `src/api/routes_chat.py` (mounted at `/api`, requires `verify_user_key`)

These are the endpoints an external frontend/backend actually talks to for conversational use.

### `POST /api/chat` (also `GET`)
The main conversational RAG endpoint. Runs the full pipeline: ticket-intake check → semantic cache → hybrid retrieval (dense + BM25) → rerank → confidence gate → answer generation.

**Request body:**
```json
{
  "query": "شو الأوراق المطلوبة للتسجيل بالنقابة؟",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "top_k": 15,
  "user": {
    "user_id": "optional-caller-supplied id",
    "name": "optional",
    "phone": "optional",
    "engineer_number": "optional"
  }
}
```
`history`, `top_k`, and `user` are all optional. `user` should be populated by whatever backend sits in front of this service if it already knows the caller's identity — it's used only for ticket creation (see note below).

**Response body:**
```json
{
  "answer": "...",
  "sources": [{"text": "...", "title": "...", "final_score": 0.87, "...": "..."}],
  "cache_hit": false,
  "metadata": { "...": "profiling data" }
}
```

**Important behavior — this endpoint can silently escalate instead of answering.** If the ticket-intake stage detects an explicit "talk to a human" request, or the RAG pipeline can't produce a confident answer, `answer` will be a ticket-confirmation/apology message instead of a knowledge-base answer, and `sources` will be `[]`. The response shape doesn't change, so a caller can't distinguish "normal answer" from "got escalated" without inspecting the text — there's no dedicated `ticket_id` field or `route` flag in this endpoint's response (unlike the internal pipeline result, which does carry one — see `src/rag_chatbot_engine.py::process_query`). Worth knowing if a frontend wants to render escalations differently.

### `GET /api/query` (also `POST`)
Pure alias for `/api/chat` — same request/response, same behavior, different path. Exists for backward compatibility.

### `POST /api/chat/stream` (also `GET`)
Server-Sent-Events streaming version of `/api/chat`. Same request body and the same ticket-escalation behavior, but the response is `text/event-stream` instead of a single JSON body.

**Event format**, each line prefixed `data: `:
```json
{"type": "meta", "sources": [...], "cache_hit": false, "profiling": {}}
{"type": "token", "token": "..."}
...
{"type": "done", "answer": "full accumulated answer", "text_breakdown": ""}
```
For a ticket escalation, only one `meta` + one `token` (the full message) + `done` are sent — there's no real token-by-token streaming for that path since the message is generated synchronously.

### `POST /api/search` (also `GET`)
Raw hybrid retrieval — dense vector search + BM25 + rerank — **without** LLM answer generation. Useful for debugging retrieval quality or building a "search results" UI separate from chat.

**Request body:**
```json
{ "query": "شروط التقاعد والتأمين الصحي", "top_k": 5 }
```

**Response body:**
```json
{ "query": "...", "results": [{"text": "...", "title": "...", "final_score": 0.91}] }
```

---

## System status — `src/api/main.py` (unauthenticated)

### `GET /api/health`
Liveness/readiness check — confirms the vector store is reachable and returns its chunk count. Returns `503` if the underlying ChromaDB collection can't be reached.
```json
{ "status": "ok", "chunks": 1842 }
```

### `GET /api/stats`
Knowledge-base size metrics — total registered source documents vs. total indexed vector chunks.
```json
{ "total_documents": 76, "total_chunks": 1842 }
```

### `GET /`
Serves the built-in demo chat UI (single-page HTML, embeds the configured API key directly into the page for convenience). **Testing/demo only** — not intended to be the production frontend; kept in the app for internal verification, per team decision.

---

## Admin — `src/api/routes_admin.py` (mounted at `/api/admin`, requires `verify_admin_key`)

Knowledge-base document management and ticket triage — everything a staff/ops tool would call.

### `POST /api/admin/files/upload`
Uploads a new source document (`.pdf`, `.txt`, `.md`) and kicks off ingestion (parsing, chunking, embedding) as a background task. Returns immediately with `202` — ingestion happens async.
```json
{ "status": "processing_started", "filename": "...", "message": "..." }
```

### `DELETE /api/admin/files/{filename}`
Removes a document from the knowledge base (excludes it from retrieval; does not necessarily delete the source file from disk).

### `POST /api/admin/files/re-ingest`
Re-runs ingestion for specific files (`{"files": ["a.pdf", "b.md"]}`) or all files under the data directories (`{"reingest_all": true}`). Also runs as background tasks.

### `GET /api/admin/files`
Lists every document currently registered in the knowledge base, with metadata (hash, status, last-processed time).
```json
{ "documents": [...], "count": 76 }
```

### `GET /api/admin/tickets`
Lists support tickets created by the ticket-intake flow (see `TICKETING_SYSTEM.md` / `ESCALATION_FLOWS.md`). Optional `?status=` query filter (`open`, `in_progress`, `resolved`, `closed`).
```json
{ "tickets": [{"id": 1, "reason": "user_intent", "priority": "high", "status": "open", "...": "..."}], "count": 1 }
```

### `GET /api/admin/tickets/{ticket_id}`
Fetches a single ticket by ID. `404` if it doesn't exist.

### `POST /api/admin/tickets/{ticket_id}/status`
Updates a ticket's status. Body: `{"status": "resolved"}`. Rejects (`400`) any value outside `open | in_progress | resolved | closed`.

---

## Admin dashboard — `src/api/routes_admin_ui.py` (unauthenticated page load)

### `GET /admin`
Serves an HTML dashboard (file upload/re-ingest controls, document list, ticket triage view). The page itself loads without a key, but every action it takes calls the `/api/admin/*` endpoints above, which do require `ADMIN_API_KEY` — the page prompts the operator to paste it in before any admin action works.

---

## Error handling

| Status | Meaning | Typical cause |
|---|---|---|
| `400` | Bad request | Empty `query`, invalid ticket `status` value, unsupported upload file type |
| `403` | Auth failure | Missing/incorrect `X-API-Key` |
| `404` | Not found | Unknown `filename` or `ticket_id` |
| `500` | Internal error | Unhandled exception in the pipeline or storage layer |
| `503` | Dependency unavailable | `/api/health` when the vector store can't be reached |
