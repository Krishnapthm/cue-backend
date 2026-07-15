"""Test-wide environment setup.

`app.config.Settings` and `app.auth.config.AuthSettings` are instantiated
eagerly at import time (module-level singletons), so these env vars must be
set before any `app.*` module is imported - including transitively, via
collection of any test module. OS env vars take precedence over `.env`, so
this always wins over a developer's real `.env`; tests never touch the dev
database.
"""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"
os.environ["AUTH_FIREBASE_PROJECT_ID"] = "cue-test"

import subprocess
from collections.abc import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer


@pytest.fixture(scope="session")
def postgres_url() -> Generator[str]:
    """A real, ephemeral Postgres instance with the schema migrated onto it.

    Never the dev database, never SQLite: a fresh container per test session,
    migrated with the project's own Alembic chain so integration tests run
    against the actual schema.
    """
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            check=True,
            env={**os.environ, "DATABASE_URL": url},
        )
        yield url


@pytest_asyncio.fixture
async def db_session(postgres_url: str) -> AsyncGenerator[AsyncSession]:
    """A request-scoped session bound to the ephemeral test database."""
    engine = create_async_engine(postgres_url)
    session_factory = async_sessionmaker(
        bind=engine, expire_on_commit=False, autoflush=False
    )
    async with session_factory() as session:
        yield session
    await engine.dispose()
