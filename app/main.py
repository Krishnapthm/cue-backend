from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.addresses.router import router as addresses_router
from app.agent.graph import open_compiled_graph
from app.auth.router import router as auth_router
from app.cart.router import router as cart_router
from app.chat.router import router as chat_router
from app.config import Environment, settings
from app.database import engine
from app.exceptions import register_exception_handlers
from app.health.router import router as health_router
from app.orders.router import router as orders_router
from app.pantry.router import router as pantry_router
from app.providers.router import router as providers_router
from app.tags.router import router as tags_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI) -> AsyncGenerator[None]:
    """Manage startup and shutdown of app-wide resources.

    The agent graph is built and compiled **once per process** and stashed on
    `app.state`, backed by a checkpointer connection pool. It used to be
    per-request, which opened a dedicated connection per call; with long-lived
    SSE streams that pinned one connection per open stream and a handful of
    concurrent users exhausted the database (CUE-93).

    Sharing one compiled graph across concurrent requests is safe by
    construction: everything per-run travels in the run config (`thread_id`)
    or in `CueContext`, both supplied per invocation.

    The checkpointer's own tables are **not** provisioned here - that is DDL
    and belongs to deployment, beside `alembic upgrade head`. See
    `scripts/setup_checkpointer.py`.
    """
    logger.info("Starting cue-api in %s", settings.ENVIRONMENT.value)
    async with open_compiled_graph() as graph:
        fastapi_app.state.agent_graph = graph
        logger.info("Agent graph compiled and its checkpointer pool opened")
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

# Native iOS/Android do not enforce CORS, but Expo's web target (react-native-web)
# does, and it is the fastest loop for integration work. `CORS_ALLOW_ORIGINS`
# defaults to empty, so an unconfigured deployment installs the middleware and
# permits nothing rather than shipping wide open.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["Authorization", "Content-Type"],
)

register_exception_handlers(app)
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(providers_router)
app.include_router(chat_router)
app.include_router(orders_router)
app.include_router(pantry_router)
app.include_router(tags_router)
app.include_router(addresses_router)
app.include_router(cart_router)
