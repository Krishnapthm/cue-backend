from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

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


@pytest.fixture
def mock_swiggy_token_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., None]:
    """Stub Swiggy's `/auth/token` endpoint so tests never hit the real internet.

    Patches `app.providers.service._exchange_code` directly rather than the
    global `httpx.AsyncClient`, so it never leaks into the ASGI test client's
    own HTTP calls against the app.

    Returns a setter the test can call to configure the response; defaults to
    a successful exchange.
    """
    response_status: int = httpx.codes.OK
    response_json = dict(DEFAULT_TOKEN_PAYLOAD)

    def _configure(*, status_code: int = httpx.codes.OK, **payload: object) -> None:
        nonlocal response_status, response_json
        response_status = status_code
        response_json = {**DEFAULT_TOKEN_PAYLOAD, **payload}

    async def _fake_exchange_code(**kwargs: object) -> httpx.Response:
        return httpx.Response(
            response_status,
            json=response_json,
            request=httpx.Request("POST", "https://mcp.swiggy.com/auth/token"),
        )

    monkeypatch.setattr(service, "_exchange_code", _fake_exchange_code)
    return _configure
