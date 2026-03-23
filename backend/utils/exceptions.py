"""
Custom exceptions for the Country Info AI Agent.
"""

from fastapi import HTTPException, status


# ─── HTTP Exceptions (FastAPI) ─────────────────────────────────────────────────

class NotFoundHTTPException(HTTPException):
    """404 — Resource not found."""

    def __init__(self, detail: str = "The requested resource was not found."):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class BadRequestHTTPException(HTTPException):
    """400 — Bad request."""

    def __init__(self, detail: str = "Bad request."):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class InternalServerHTTPException(HTTPException):
    """500 — Internal server error."""

    def __init__(self, detail: str = "An internal server error occurred."):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail
        )


class RateLimitHTTPException(HTTPException):
    """429 — Too many requests."""

    def __init__(self, detail: str = "Rate limit exceeded. Please try again later."):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail
        )
