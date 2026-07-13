from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base class for all domain exceptions.

    Domains subclass this and override `status_code` / `detail`; the global
    handler maps them onto HTTP responses so services never import HTTP types.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = status.HTTP_404_NOT_FOUND
    detail = "Resource not found."


class DatabaseUnavailableError(AppError):
    """The database could not be reached."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    detail = "Database is unavailable."


async def app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    """Render an `AppError` as a JSON response."""
    assert isinstance(exc, AppError)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the global exception handlers to the app."""
    handlers: dict[Any, Any] = {AppError: app_error_handler}
    for exc_class, handler in handlers.items():
        app.add_exception_handler(exc_class, handler)
