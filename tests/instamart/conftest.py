from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provider import ProviderLink
from app.models.user import User
from app.providers import service as provider_service
from app.providers.constants import PROVIDER

ACCESS_TOKEN = "at_test_instamart_token"

DEFAULT_RESULT: dict[str, Any] = {"success": True, "data": {}}


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
    """`user`, with an active Swiggy link decrypting to ACCESS_TOKEN."""
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=provider_service._encrypt(ACCESS_TOKEN),
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
            "result": dict(DEFAULT_RESULT),
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
