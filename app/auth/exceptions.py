from __future__ import annotations

from fastapi import status

from app.exceptions import AppError


class InvalidTokenError(AppError):
    """The bearer token is missing, malformed, expired, or fails verification."""

    status_code = status.HTTP_401_UNAUTHORIZED
    detail = "Invalid or expired authentication token."
