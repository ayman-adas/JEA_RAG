import sys
import os
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

# Force UTF-8 encoding on Windows console stdout/stderr
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.monitoring.structured_logger import setup_structured_logging

# Configure structured logging
setup_structured_logging(
    level=os.getenv("LOG_LEVEL", "INFO"),
    json_format=os.getenv("LOG_FORMAT", "text").lower() == "json",
)

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Security, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.gzip import GZipMiddleware

from src.core.rag_engine import KnowledgeRetriever
from src.cache_db.document_registry import DocumentRegistry
from src.api.security import SecurityHeadersMiddleware, InMemoryRateLimiterMiddleware
from src.services.http_client_pool import close_http_client_pools
from src.api.middleware import (
    http_exception_handler,
    validation_exception_handler,
    global_unhandled_exception_handler,
)
from src.api.response_envelope import ApiResponseEnvelope
from .routes_chat import router as chat_router
from .routes_admin import router as admin_router
from .routes_admin_ui import router as admin_ui_router
from .dependencies import USER_API_KEY, ADMIN_API_KEY

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
UI_TEMPLATE_PATH = TEMPLATES_DIR / "chat_ui.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Warms up essential singletons on startup and gracefully closes HTTP pools on shutdown.
    """
    logger.info("🚀 Starting Eng-Guild Chatbot API Service (v3.0.0)...")
    try:
        # Pre-warm retriever and document registry in background thread
        await asyncio.to_thread(KnowledgeRetriever)
        await asyncio.to_thread(DocumentRegistry)
        logger.info("✅ Singletons and Vector DB initialized successfully.")
    except Exception as e:
        logger.warning(f"⚠️ Non-fatal warming warning on startup: {e}")

    yield

    logger.info("🛑 Shutting down Eng-Guild Chatbot API Service. Releasing resources...")
    await close_http_client_pools()
    logger.info("✅ All connection pools closed cleanly.")


app = FastAPI(
    title="Guild Knowledge Base RAG Chatbot UI",
    version="3.0.0",
    lifespan=lifespan,
)

# Exception Handlers (Standardized ApiResponse format matching NestJS)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, global_unhandled_exception_handler)

# Middlewares
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    InMemoryRateLimiterMiddleware,
    max_requests=int(os.getenv("RATE_LIMIT_PER_MINUTE", "120")),
    window_seconds=60,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()],
    allow_methods=["POST", "GET", "DELETE", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization", "x-internal-token", "X-Request-ID"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Routes
app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api/admin")
app.include_router(admin_ui_router)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/api/health/live")
def health_live():
    return ApiResponseEnvelope.success_response({"status": "alive"})


@app.get("/api/health/ready")
async def health_ready():
    checks = {}

    def _check_sqlite():
        registry = DocumentRegistry()
        with registry._get_connection() as conn:
            conn.execute("SELECT 1;")
        return "healthy"

    def _check_vector_store():
        retriever = KnowledgeRetriever()
        chunk_count = retriever.collection.count()
        return f"healthy ({chunk_count} chunks)"

    try:
        checks["sqlite"] = await asyncio.to_thread(_check_sqlite)
    except Exception as e:
        checks["sqlite"] = f"unhealthy: {e}"

    try:
        checks["vector_store"] = await asyncio.to_thread(_check_vector_store)
    except Exception as e:
        checks["vector_store"] = f"unhealthy: {e}"

    is_healthy = all("unhealthy" not in v for v in checks.values())
    status_code = 200 if is_healthy else 503
    payload = ApiResponseEnvelope.success_response(
        {"status": "ready" if is_healthy else "degraded", "checks": checks}
    )
    return JSONResponse(content=payload, status_code=status_code)


@app.get("/api/health")
async def health():
    def _count():
        retriever = KnowledgeRetriever()
        return retriever.collection.count()

    try:
        chunk_count = await asyncio.to_thread(_count)
        return ApiResponseEnvelope.success_response({"status": "ok", "chunks": chunk_count})
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    def _fetch_stats():
        registry = DocumentRegistry()
        retriever = KnowledgeRetriever()
        total_docs = len(registry.list_documents())
        total_chunks = retriever.collection.count()
        return {"total_documents": total_docs, "total_chunks": total_chunks}

    stats = await asyncio.to_thread(_fetch_stats)
    return ApiResponseEnvelope.success_response(stats)


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    default_key = USER_API_KEY or ADMIN_API_KEY or ""
    if UI_TEMPLATE_PATH.exists():
        content = UI_TEMPLATE_PATH.read_text(encoding="utf-8")
        return content.replace("{{DEFAULT_USER_KEY}}", default_key)
    return HTMLResponse("<h1>JEA Chatbot API is online. UI template not found.</h1>", status_code=200)
