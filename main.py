"""Entrypoint shim so `uvicorn main:app` resolves to the real app in `app.main`."""

from __future__ import annotations

from app.main import app

__all__ = ["app"]
