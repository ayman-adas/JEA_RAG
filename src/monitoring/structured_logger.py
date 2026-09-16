"""
Structured JSON Logger & Tracing Context for Eng-Guild Chatbot Platform.
Outputs machine-readable JSON log events with correlation IDs for ELK / Loki ingestion.
"""

import os
import json
import logging
import contextvars
from datetime import datetime, timezone
from typing import Optional, Any

# ContextVar for distributed tracing correlation ID across async coroutines
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects with ISO 8601 UTC timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "requestId": request_id_ctx.get(),
            "process": os.getpid(),
        }

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        # Include custom extra metadata if supplied
        if hasattr(record, "extra_context") and isinstance(record.extra_context, dict):
            log_obj.update(record.extra_context)

        return json.dumps(log_obj, ensure_ascii=False)


def setup_structured_logging(level: str = "INFO", json_format: bool = False):
    """Configures application root logger with either JSON or standard formatted handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplication
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    handler = logging.StreamHandler()
    if json_format or os.getenv("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d]: %(message)s")
        )

    root_logger.addHandler(handler)
