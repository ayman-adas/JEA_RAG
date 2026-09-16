"""
Global Exception Handlers and Standardized Response Middlewares for FastAPI.
Provides parity with NestJS HttpExceptionFilter and TransformResponseInterceptor.
"""

import logging
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from src.monitoring.structured_logger import request_id_ctx
from src.api.response_envelope import ApiResponseEnvelope

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Catches all standard HTTPExceptions and wraps them in ApiResponseEnvelope format."""
    request_id = request_id_ctx.get() or request.headers.get("x-request-id")
    logger.warning(
        f"[HTTP {exc.status_code}] {request.method} {request.url.path} - {exc.detail} (reqId: {request_id})"
    )
    code = f"HTTP_{exc.status_code}"
    if exc.status_code == status.HTTP_401_UNAUTHORIZED:
        code = "UNAUTHORIZED"
    elif exc.status_code == status.HTTP_403_FORBIDDEN:
        code = "FORBIDDEN"
    elif exc.status_code == status.HTTP_404_NOT_FOUND:
        code = "NOT_FOUND"
    elif exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
        code = "RATE_LIMIT_EXCEEDED"

    envelope = ApiResponseEnvelope.error_response(
        code=code,
        message=str(exc.detail),
        request_id=request_id,
    )
    return JSONResponse(status_code=exc.status_code, content=envelope)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Catches Pydantic schema validation errors and returns structured 422 payload."""
    request_id = request_id_ctx.get() or request.headers.get("x-request-id")
    logger.warning(
        f"[Validation Error] {request.method} {request.url.path} - {exc.errors()} (reqId: {request_id})"
    )
    envelope = ApiResponseEnvelope.error_response(
        code="VALIDATION_FAILED",
        message="Request payload failed schema validation.",
        details=exc.errors(),
        request_id=request_id,
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=envelope)


async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all fallback handler for uncaught server errors (500)."""
    request_id = request_id_ctx.get() or request.headers.get("x-request-id")
    logger.error(
        f"[Unhandled Server Error] {request.method} {request.url.path} - {str(exc)} (reqId: {request_id})",
        exc_info=exc,
    )
    envelope = ApiResponseEnvelope.error_response(
        code="INTERNAL_SERVER_ERROR",
        message="An internal server error occurred while processing the request.",
        details=str(exc) if request.app.debug else None,
        request_id=request_id,
    )
    return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=envelope)
