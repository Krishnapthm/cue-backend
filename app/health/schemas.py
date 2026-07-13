from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Liveness of the service itself."""

    status: str
    environment: str


class DatabaseHealthResponse(BaseModel):
    """Result of a round-trip query against the database."""

    status: str
    database: str
    server_version: str
    latency_ms: float
