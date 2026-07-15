from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from urllib.parse import parse_qs, urlparse

import httpx
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.main import app
from app.models.user import User


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient]:
    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test", follow_redirects=False
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authed_client(
    client: httpx.AsyncClient, user: User
) -> AsyncGenerator[httpx.AsyncClient]:
    from app.auth.dependencies import current_user

    async def override_current_user() -> User:
        return user

    app.dependency_overrides[current_user] = override_current_user
    yield client


async def test_authorize_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.post("/providers/swiggy/authorize")

    assert response.status_code == 401


async def test_authorize_returns_a_swiggy_consent_url(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post("/providers/swiggy/authorize")

    assert response.status_code == 200
    body = response.json()
    assert body["authorize_url"].startswith("https://mcp.swiggy.com/auth/authorize?")


async def test_callback_redirects_to_the_app_on_success(
    authed_client: httpx.AsyncClient,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    authorize_response = await authed_client.post("/providers/swiggy/authorize")
    state = parse_qs(urlparse(authorize_response.json()["authorize_url"]).query)[
        "state"
    ][0]

    response = await authed_client.get(
        "/providers/swiggy/callback", params={"code": "auth-code", "state": state}
    )

    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("cue://swiggy-link")
    assert "swiggy_link=success" in location


async def test_callback_redirects_to_the_app_on_invalid_state(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(
        "/providers/swiggy/callback",
        params={"code": "auth-code", "state": "never-issued"},
    )

    assert response.status_code == 307
    location = response.headers["location"]
    assert location.startswith("cue://swiggy-link")
    assert "swiggy_link=error" in location


async def test_callback_redirects_to_the_app_when_swiggy_reports_an_error(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get(
        "/providers/swiggy/callback",
        params={"error": "access_denied", "state": "whatever"},
    )

    assert response.status_code == 307
    assert "swiggy_link=error" in response.headers["location"]
