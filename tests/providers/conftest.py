from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provider import ProviderLink
from app.models.user import User
from app.providers import service
from app.providers.constants import PROVIDER

DEFAULT_TOKEN_PAYLOAD = {
    "access_token": "at_test_token",
    "token_type": "Bearer",
    "expires_in": 432000,
    "scope": "mcp:tools mcp:resources mcp:prompts",
}


@pytest_asyncio.fixture
async def user(db_session: AsyncSession) -> User:
    """A persisted Cue user to link a Swiggy account to."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="user@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


@pytest_asyncio.fixture
async def other_user(db_session: AsyncSession) -> User:
    """A second Cue user, for cross-user authorization checks."""
    new_user = User(
        firebase_uid=f"firebase-uid-{uuid.uuid4()}", email="other@example.com"
    )
    db_session.add(new_user)
    await db_session.commit()
    await db_session.refresh(new_user)
    return new_user


@pytest_asyncio.fixture
async def active_link(db_session: AsyncSession, user: User) -> ProviderLink:
    """An already-linked, active Swiggy provider link for `user`."""
    link = ProviderLink(
        user_id=user.id,
        provider=PROVIDER,
        access_token_ct=b"ciphertext",
        token_expires_at=datetime.now(UTC) + timedelta(days=5),
        scope="mcp:tools",
        status="active",
    )
    db_session.add(link)
    await db_session.commit()
    await db_session.refresh(link)
    return link


@dataclass
class SwiggyTokenEndpointStub:
    """Configurable stand-in for Swiggy's `/auth/token` endpoint.

    Calling the stub configures the response (defaults to a successful
    exchange); `requests` records every outbound request that reached it, so
    tests can assert on what was actually sent to Swiggy.
    """

    status_code: int = int(httpx.codes.OK)
    payload: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_TOKEN_PAYLOAD))
    requests: list[httpx.Request] = field(default_factory=list)

    def __call__(
        self, *, status_code: int = int(httpx.codes.OK), **payload: Any
    ) -> None:
        self.status_code = status_code
        self.payload = {**DEFAULT_TOKEN_PAYLOAD, **payload}

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status_code, json=self.payload)

    @property
    def request_bodies(self) -> list[dict[str, Any]]:
        """The JSON body of every request the stub received, in order."""
        bodies: list[dict[str, Any]] = [
            json.loads(request.content) for request in self.requests
        ]
        return bodies


@pytest.fixture
def mock_swiggy_token_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> SwiggyTokenEndpointStub:
    """Stub Swiggy's `/auth/token` endpoint so tests never hit the real internet.

    Replaces `httpx` as bound in `app.providers.service` with a namespace whose
    `AsyncClient` is wired to a `MockTransport`, so the real request-building
    code still runs (and can be asserted on) while nothing leaves the process.
    Scoped to that one module, so it never leaks into the ASGI test client's
    own HTTP calls against the app.
    """
    stub = SwiggyTokenEndpointStub()

    class _MockTransportAsyncClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(transport=httpx.MockTransport(stub.handle), **kwargs)

    monkeypatch.setattr(
        service,
        "httpx",
        SimpleNamespace(
            AsyncClient=_MockTransportAsyncClient,
            HTTPError=httpx.HTTPError,
            codes=httpx.codes,
        ),
    )
    return stub
