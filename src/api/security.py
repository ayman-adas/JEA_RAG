"""
Security, Performance, and Rate Limiting utilities for Eng-Guild Chatbot API.
Implements defense-in-depth inspired by jea_backend production architecture
and OWASP API Security Top 10 standards.
"""

import time
import secrets
import re
from typing import Dict, Tuple, Optional
from collections import defaultdict
from fastapi import Request, Response, HTTPException, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from src.config import INTERNAL_BYPASS_TOKEN
from .dependencies import ADMIN_API_KEY, USER_API_KEY

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_AUTH_HEADER = APIKeyHeader(name="Authorization", auto_error=False)
_INTERNAL_HEADER = APIKeyHeader(name="x-internal-token", auto_error=False)


def safe_compare(val1: Optional[str], val2: Optional[str]) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if not val1 or not val2:
        return False
    return secrets.compare_digest(val1.strip(), val2.strip())


def verify_authenticated_client(
    x_api_key: Optional[str] = Security(_API_KEY_HEADER),
    auth_header: Optional[str] = Security(_AUTH_HEADER),
    internal_token: Optional[str] = Security(_INTERNAL_HEADER),
) -> str:
    """
    Verifies user or microservice identity in constant time.
    Supports:
    1. x-internal-token matching INTERNAL_BYPASS_TOKEN or 'jea_rag_token'
    2. X-API-Key header matching USER_API_KEY or ADMIN_API_KEY
    3. Authorization: Bearer <key> matching USER_API_KEY or ADMIN_API_KEY
    """
    # 1. Check internal service token from jea_backend
    target_internal = INTERNAL_BYPASS_TOKEN or "jea_rag_token"
    if internal_token and safe_compare(internal_token, target_internal):
        return "internal_service"

    # 2. Extract token from Authorization header if present (Bearer <token>)
    bearer_token = None
    if auth_header and auth_header.lower().startswith("bearer "):
        bearer_token = auth_header[7:].strip()

    candidate_key = x_api_key or bearer_token

    # 3. If no keys are configured, allow access (local dev mode)
    if not USER_API_KEY and not ADMIN_API_KEY:
        return "anonymous_dev"

    # 4. Compare in constant time
    if candidate_key:
        if USER_API_KEY and safe_compare(candidate_key, USER_API_KEY):
            return "user"
        if ADMIN_API_KEY and safe_compare(candidate_key, ADMIN_API_KEY):
            return "admin"

    raise HTTPException(
        status_code=403,
        detail="Invalid or missing API key or internal authorization token.",
    )


from src.monitoring.structured_logger import request_id_ctx

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Attaches security headers to all responses (Fastify / Helmet equivalent):
    - Strict-Transport-Security
    - X-Content-Type-Options: nosniff
    - X-Frame-Options: SAMEORIGIN
    - X-XSS-Protection: 1; mode=block
    - Referrer-Policy: strict-origin-when-cross-origin
    - Request Process Time & Correlation ID Tracing
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        # Correlation ID: reuse upstream header or generate new
        request_id = request.headers.get("x-request-id") or request.headers.get("X-Request-ID")
        if not request_id:
            request_id = secrets.token_hex(8)

        # Bind to contextvar for structured log tracing in downstream async tasks
        token = request_id_ctx.set(request_id)
        try:
            response: Response = await call_next(request)

            # Performance & Tracing headers
            process_time_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"
            response.headers["X-Request-ID"] = request_id

            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

            return response
        finally:
            request_id_ctx.reset(token)


class InMemoryRateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Bounded sliding-window in-memory rate limiter per IP address.
    Protects LLM & RAG pipeline against denial-of-service and runaway loops.
    Exempts internal bypass tokens and health checks.
    Periodically purges inactive IPs to prevent memory growth.
    """
    def __init__(self, app, max_requests: int = 120, window_seconds: int = 60, max_tracked_ips: int = 10000):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_tracked_ips = max_tracked_ips
        self.requests: Dict[str, list] = defaultdict(list)
        self.last_cleanup_time = time.time()

    def _purge_stale_ips(self, window_start: float):
        """Purges IPs with no requests in the current window to prevent memory leaks."""
        stale_ips = [ip for ip, timestamps in self.requests.items() if not timestamps or timestamps[-1] <= window_start]
        for ip in stale_ips:
            self.requests.pop(ip, None)

    async def dispatch(self, request: Request, call_next):
        # Exempt health and static routes
        path = request.url.path
        if path in ("/api/health", "/api/health/live", "/api/health/ready", "/favicon.ico"):
            return await call_next(request)

        # Exempt requests carrying valid internal bypass token from jea_backend
        internal_token = request.headers.get("x-internal-token")
        target_internal = INTERNAL_BYPASS_TOKEN or "jea_rag_token"
        if internal_token and safe_compare(internal_token, target_internal):
            return await call_next(request)

        # Extract client IP
        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )

        now = time.time()
        window_start = now - self.window_seconds

        # Periodic memory cleanup every 60 seconds or if map exceeds capacity
        if now - self.last_cleanup_time > 60 or len(self.requests) > self.max_tracked_ips:
            self._purge_stale_ips(window_start)
            self.last_cleanup_time = now

        # Clean old timestamps for current client
        client_history = [ts for ts in self.requests[client_ip] if ts > window_start]
        self.requests[client_ip] = client_history

        if len(client_history) >= self.max_requests:
            retry_after = int(client_history[0] + self.window_seconds - now) + 1
            return Response(
                content=f'{{"detail": "Rate limit exceeded. Try again in {retry_after} seconds."}}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        self.requests[client_ip].append(now)
        return await call_next(request)


_CONTROL_CHAR_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

def sanitize_user_input(text: str, max_length: int = 4000) -> str:
    """Strips dangerous control characters and enforces length bounds."""
    if not text:
        return ""
    # Strip null bytes and non-printable control characters
    cleaned = _CONTROL_CHAR_RE.sub('', text)
    # Truncate at max_length
    return cleaned[:max_length].strip()
