from __future__ import annotations

import logging
import time

from sqlalchemy import func, literal, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions import DatabaseUnavailableError
from app.health.schemas import DatabaseHealthResponse

logger = logging.getLogger(__name__)


async def check_database(session: AsyncSession) -> DatabaseHealthResponse:
    """Round-trip a trivial query to prove the database is reachable.

    Args:
        session: An active database session.

    Returns:
        The database name, server version, and round-trip latency.

    Raises:
        DatabaseUnavailableError: If the query fails for any reason.
    """
    started = time.perf_counter()
    try:
        result = await session.execute(
            select(
                literal(1).label("ping"),
                func.current_database().label("database"),
                func.version().label("server_version"),
            )
        )
        row = result.one()
    except SQLAlchemyError:
        logger.exception("Database health check failed")
        raise DatabaseUnavailableError from None

    latency_ms = (time.perf_counter() - started) * 1000
    return DatabaseHealthResponse(
        status="ok",
        database=row.database,
        server_version=row.server_version.split(" on ")[0],
        latency_ms=round(latency_ms, 2),
    )
