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

from cryptography.fernet import Fernet

os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"
os.environ["AUTH_FIREBASE_PROJECT_ID"] = "cue-test"
os.environ["SWIGGY_CLIENT_ID"] = "test-client-id"
os.environ["SWIGGY_REDIRECT_URI"] = "https://api.cue.test/providers/swiggy/callback"
os.environ["SWIGGY_APP_CALLBACK_DEEP_LINK"] = "cue://swiggy-link"
os.environ["SWIGGY_TOKEN_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

import subprocess
import uuid
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.postgres import PostgresContainer

from app.models.provider import ProviderLink
from app.models.user import User
from app.providers import service as provider_service
from app.providers.constants import PROVIDER

INSTAMART_ACCESS_TOKEN = "at_test_instamart_token"


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


@pytest_asyncio.fixture
async def user(db_session: AsyncSession) -> User:
    """A persisted Cue user, with no Swiggy link."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="user@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


@pytest_asyncio.fixture
async def linked_user(db_session: AsyncSession, user: User) -> User:
    """`user`, with an active Swiggy link decrypting to INSTAMART_ACCESS_TOKEN."""
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=provider_service._encrypt(INSTAMART_ACCESS_TOKEN),
            token_expires_at=datetime.now(UTC) + timedelta(days=5),
            scope="mcp:tools",
            status="active",
        )
    )
    await db_session.commit()
    return user


@dataclass
class InstamartToolCallStub:
    """Configurable stand-in for the Instamart MCP HTTP POST."""

    status_code: int = 200
    json_body: dict[str, Any] = field(
        default_factory=lambda: {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"success": True, "data": {}},
        }
    )
    raises: Exception | None = None
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def configure(
        self,
        *,
        status_code: int = 200,
        result: dict[str, Any] | None = None,
        rpc_error: dict[str, Any] | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.status_code = status_code
        if rpc_error is not None:
            self.json_body = {"jsonrpc": "2.0", "id": 1, "error": rpc_error}
        elif result is not None:
            self.json_body = {"jsonrpc": "2.0", "id": 1, "result": result}
        self.raises = raises

    def record_call(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))


@pytest.fixture
def mock_instamart_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> InstamartToolCallStub:
    """Stub the Instamart MCP HTTP POST so tests never hit the real internet.

    Patches `httpx.AsyncClient` as imported into `app.instamart.client`, so it
    never leaks into the ASGI test client's own HTTP calls against the app.
    Call `.configure(...)` to set the response; `.calls` records outgoing
    request args/kwargs for assertions.
    """
    stub = InstamartToolCallStub()

    class _FakeResponse:
        @property
        def status_code(self) -> int:
            return stub.status_code

        def json(self) -> dict[str, Any]:
            return stub.json_body

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

        async def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
            stub.record_call(*args, **kwargs)
            if stub.raises is not None:
                raise stub.raises
            return _FakeResponse()

    monkeypatch.setattr("app.instamart.client.httpx.AsyncClient", _FakeAsyncClient)
    return stub
