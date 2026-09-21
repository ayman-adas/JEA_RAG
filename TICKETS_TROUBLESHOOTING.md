# Tickets Not Appearing — Checklist

Tickets are created and stored correctly on our end (verified in code). If they don't show up on your end, check these:

## 1. Admin key vs user key
`GET /api/admin/tickets` requires `ADMIN_API_KEY` (not `USER_API_KEY`).
- Check: are you sending `X-API-Key: <ADMIN_API_KEY>` on the ticket list request?
- If you get `403` → wrong key.
- If you get `500` → `ADMIN_API_KEY` not set on the server's `.env`.

## 2. Persistent storage
Tickets are stored in `data/storage/registry.db` (SQLite) inside the container.
- Check: does your deployment mount a persistent volume to `/app/data`? (see `docker-compose.yml` — `./data:/app/data`)
- If not mounted, or if you run multiple replicas of this service, each instance has its own separate DB file — a ticket written by one instance won't show up when another instance is queried.

## 3. Endpoint wiring
- Check: does your admin panel actually call **this service's** `GET /api/admin/tickets`? (correct host/port, not a stale/mocked URL)
- There is no webhook/push for new tickets — your side must poll this endpoint.

## 4. Quick DB check (bypasses the API entirely)
```bash
sqlite3 data/storage/registry.db "select id, status, created_at from tickets;"
```
- Rows present → problem is #1 or #3 (API/auth wiring).
- No rows → problem is #2 (writes not landing in shared storage).
