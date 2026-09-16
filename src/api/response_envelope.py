"""
Standardized API Response Envelope & Exception Formatting for Eng-Guild Chatbot API.
Ensures 100% schema alignment with JEA Enterprise backend (ApiResponseDto standard).
"""

from datetime import datetime, timezone
from typing import Any, Optional, Dict, Generic, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetailDto(BaseModel):
    code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Any] = Field(None, description="Detailed validation or runtime context")


class ApiResponseEnvelope(BaseModel, Generic[T]):
    success: bool = Field(..., description="Indicates request success or failure")
    data: Optional[T] = Field(None, description="Payload data when success=True")
    error: Optional[ErrorDetailDto] = Field(None, description="Error detail when success=False")
    requestId: Optional[str] = Field(None, description="Distributed correlation ID (x-request-id)")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )

    @classmethod
    def success_response(cls, data: Any, request_id: Optional[str] = None) -> Dict[str, Any]:
        return {
            "success": True,
            "data": data,
            "error": None,
            "requestId": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @classmethod
    def error_response(
        cls,
        code: str,
        message: str,
        details: Optional[Any] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "success": False,
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details,
            },
            "requestId": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
