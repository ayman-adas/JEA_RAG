"""
Centralized HTTP Connection Pool for Eng-Guild Chatbot microservices.
Manages persistent TCP sessions with connection pooling, keepalive, and timeout limits.
"""

import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

# Global singletons
_async_client: Optional[httpx.AsyncClient] = None
_sync_client: Optional[httpx.Client] = None

# Default pool limits
DEFAULT_MAX_CONNECTIONS = 100
DEFAULT_MAX_KEEPALIVE = 20
DEFAULT_TIMEOUT_SECONDS = 5.0


def get_async_http_client(
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
) -> httpx.AsyncClient:
    """Returns or initializes the shared persistent AsyncClient."""
    global _async_client
    if _async_client is None or _async_client.is_closed:
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=30.0,
        )
        _async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=3.0),
            limits=limits,
            follow_redirects=True,
        )
        logger.info("[HTTPClientPool] Initialized shared AsyncClient connection pool.")
    return _async_client


def get_sync_http_client(
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    max_keepalive: int = DEFAULT_MAX_KEEPALIVE,
) -> httpx.Client:
    """Returns or initializes the shared persistent SyncClient."""
    global _sync_client
    if _sync_client is None or _sync_client.is_closed:
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive,
            keepalive_expiry=30.0,
        )
        _sync_client = httpx.Client(
            timeout=httpx.Timeout(timeout, connect=3.0),
            limits=limits,
            follow_redirects=True,
        )
        logger.info("[HTTPClientPool] Initialized shared SyncClient connection pool.")
    return _sync_client


async def close_http_client_pools():
    """Gracefully closes all open HTTP connection pools during application shutdown."""
    global _async_client, _sync_client
    if _async_client and not _async_client.is_closed:
        await _async_client.aclose()
        logger.info("[HTTPClientPool] Shared AsyncClient closed successfully.")
    if _sync_client and not _sync_client.is_closed:
        _sync_client.close()
        logger.info("[HTTPClientPool] Shared SyncClient closed successfully.")
