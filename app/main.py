from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.chat.router import router as chat_router
from app.config import Environment, settings
from app.database import engine
from app.exceptions import register_exception_handlers
from app.health.router import router as health_router
from app.orders.router import router as orders_router
from app.providers.router import router as providers_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None]:
    """Manage startup and shutdown of app-wide resources."""
    logger.info("Starting cue-api in %s", settings.ENVIRONMENT.value)
    yield
    await engine.dispose()
    logger.info("Database connections closed")


app = FastAPI(
    title="Cue API",
    version="0.1.0",
    lifespan=lifespan,
    # No interactive docs outside local/staging.
    openapi_url=None
    if settings.ENVIRONMENT is Environment.PRODUCTION
    else "/openapi.json",
)

register_exception_handlers(app)
app.include_router(health_router)
app.include_router(providers_router)
app.include_router(chat_router)
app.include_router(orders_router)
