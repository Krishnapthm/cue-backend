from __future__ import annotations

from fastapi import APIRouter, status

from app.config import settings
from app.database import DbSession
from app.health import service
from app.health.schemas import DatabaseHealthResponse, HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
)
async def health() -> dict[str, str]:
    """Report that the service is up. Does not touch the database."""
    return {"status": "ok", "environment": settings.ENVIRONMENT.value}


@router.get(
    "/health/db",
    response_model=DatabaseHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Database connectivity probe",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Database unreachable"},
    },
)
async def database_health(session: DbSession) -> DatabaseHealthResponse:
    """Prove the service can reach Postgres and run a query."""
    return await service.check_database(session)
