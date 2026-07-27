"""Test-wide environment setup.

`app.config.Settings` and `app.auth.config.AuthSettings` are instantiated
eagerly at import time (module-level singletons), so these env vars must be
set before any `app.*` module is imported - including transitively, via
collection of any test module. OS env vars take precedence over `.env`, so
this always wins over a developer's real `.env`; tests never touch the dev
database.
"""

from __future__ import annotations

import json
import os

from cryptography.fernet import Fernet

os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost/test"
os.environ["AUTH_FIREBASE_PROJECT_ID"] = "cue-test"
os.environ["SWIGGY_CLIENT_ID"] = "test-client-id"
os.environ["SWIGGY_REDIRECT_URI"] = "https://api.cue.test/providers/swiggy/callback"
os.environ["SWIGGY_APP_CALLBACK_DEEP_LINK"] = "cue://swiggy-link"
os.environ["SWIGGY_TOKEN_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
os.environ["AGENT_MODEL_NAME"] = "claude-test-model"
# One allowed browser origin, so the CORS middleware on `app.main.app` has
# something to allow and something to reject. The *default* being empty is
# asserted separately, against the settings class rather than this instance.
os.environ["CORS_ALLOW_ORIGINS"] = '["http://localhost:8081"]'

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
            "result": {"structuredContent": {}},
        }
    )
    raises: Exception | None = None
    # Response encoding. Swiggy answers `application/json` in practice, but
    # MCP permits SSE, so both are stubbable.
    content_type: str = "application/json"
    # Set to answer with a body that isn't the JSON encoding of `json_body`
    # (an SSE frame, or something undecodable).
    raw_body: str | None = None
    # Per-tool-name overrides of the default (status_code, response), for
    # tests exercising a flow that calls more than one tool (e.g. get_cart
    # then checkout) where each call needs a different outcome.
    by_tool: dict[str, tuple[int, dict[str, Any]]] = field(default_factory=dict)
    # Per-search-query overrides, for flows that search several distinct terms
    # in one request (CUE-74's batch tag resolution) and need each term to
    # come back with its own candidates. Keyed on the `query` argument, and
    # consulted before `by_tool`.
    by_query: dict[str, tuple[int, dict[str, Any]]] = field(default_factory=dict)
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

    def configure_text_envelope(
        self,
        envelope: dict[str, Any],
        *,
        tool_name: str | None = None,
        status_code: int = 200,
    ) -> None:
        """Answer with Swiggy's *real* wire shape (CUE-77).

        Swiggy never populates MCP's `structuredContent`; it JSON-encodes its
        own `{success, data, message}` envelope into a text content block.
        `configure`/`configure_tool_result` mock the shape the client used to
        assume, which is why this bug survived 400+ passing tests - use this
        helper for anything that should be proven against production's shape.

        Args:
            envelope: Swiggy's envelope, e.g. `{"success": True, "data": ...}`.
            tool_name: Scope the response to one tool, as per
                `configure_tool_result`; None sets the default response.
            status_code: HTTP status to answer with.
        """
        result = {"content": [{"type": "text", "text": json.dumps(envelope)}]}
        if tool_name is None:
            self.status_code = status_code
            self.json_body = {"jsonrpc": "2.0", "id": 1, "result": result}
        else:
            self.by_tool[tool_name] = (
                status_code,
                {"jsonrpc": "2.0", "id": 1, "result": result},
            )

    def configure_sse(self, *, result: dict[str, Any]) -> None:
        """Answer with the same envelope encoded as a text/event-stream body."""
        self.status_code = 200
        self.json_body = {"jsonrpc": "2.0", "id": 1, "result": result}
        self.content_type = "text/event-stream"
        self.raw_body = f"event: message\ndata: {json.dumps(self.json_body)}\n\n"

    def configure_raw_body(self, body: str, *, status_code: int = 200) -> None:
        """Answer 200 with a body that is not a JSON-RPC envelope at all."""
        self.status_code = status_code
        self.raw_body = body

    def configure_tool_result(
        self, tool_name: str, result: dict[str, Any], *, status_code: int = 200
    ) -> None:
        """Set a distinct outcome for one tool name only."""
        self.by_tool[tool_name] = (
            status_code,
            {"jsonrpc": "2.0", "id": 1, "result": result},
        )

    def configure_tool_status(self, tool_name: str, status_code: int) -> None:
        """Set a distinct HTTP status (e.g. 503) for one tool name only."""
        self.by_tool[tool_name] = (status_code, {"jsonrpc": "2.0", "id": 1})

    def configure_search_query(
        self, query: str, result: dict[str, Any], *, status_code: int = 200
    ) -> None:
        """Set a distinct outcome for one `search_products` query only."""
        self.by_query[query] = (
            status_code,
            {"jsonrpc": "2.0", "id": 1, "result": result},
        )

    def record_call(self, *args: Any, **kwargs: Any) -> None:
        self.calls.append((args, kwargs))

    def tool_calls(self, tool_name: str) -> list[dict[str, Any]]:
        """Every recorded call to one tool, as its arguments mapping."""
        return [
            params["arguments"]
            for _, kwargs in self.calls
            if (params := kwargs.get("json", {}).get("params", {})).get("name")
            == tool_name
        ]

    def response_for(
        self, tool_name: str | None, arguments: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        query = arguments.get("query")
        if isinstance(query, str) and query in self.by_query:
            return self.by_query[query]
        if tool_name is not None and tool_name in self.by_tool:
            return self.by_tool[tool_name]
        return self.status_code, self.json_body


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
        def __init__(self, status_code: int, json_body: dict[str, Any]) -> None:
            self.status_code = status_code
            self._json_body = json_body
            # The client branches on content-type to pick a decoder, so the
            # stub has to carry a real headers mapping.
            self.headers = {"content-type": stub.content_type}

        @property
        def text(self) -> str:
            if stub.raw_body is not None:
                return stub.raw_body
            return json.dumps(self._json_body)

        def json(self) -> dict[str, Any]:
            if stub.raw_body is not None:
                decoded: dict[str, Any] = json.loads(stub.raw_body)
                return decoded
            return self._json_body

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
            params = kwargs.get("json", {}).get("params", {})
            status_code, json_body = stub.response_for(
                params.get("name"), params.get("arguments", {})
            )
            return _FakeResponse(status_code, json_body)

    monkeypatch.setattr("app.instamart.client.httpx.AsyncClient", _FakeAsyncClient)
    return stub
