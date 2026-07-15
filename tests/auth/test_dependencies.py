"""End-to-end dependency chain: header -> verified claims -> stable Cue user."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable

import httpx
import pytest_asyncio
from fastapi import APIRouter, FastAPI
from httpx import ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.dependencies import CurrentUser
from app.database import get_session
from app.exceptions import register_exception_handlers


def _build_test_app() -> FastAPI:
    """A throwaway app with one CurrentUser-gated route, to exercise the dependency."""
    app = FastAPI()
    register_exception_handlers(app)
    router = APIRouter()

    @router.get("/whoami")
    async def whoami(user: CurrentUser) -> dict[str, int]:
        return {"user_id": user.id}

    app.include_router(router)
    return app


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[httpx.AsyncClient]:
    app = _build_test_app()

    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = override_get_session
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def test_whoami_rejects_missing_authorization_header(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/whoami")

    assert response.status_code == 401


async def test_whoami_rejects_invalid_token(client: httpx.AsyncClient) -> None:
    response = await client.get(
        "/whoami", headers={"Authorization": "Bearer not-a-jwt"}
    )

    assert response.status_code == 401


async def test_whoami_provisions_and_returns_a_stable_user_id_for_a_valid_token(
    client: httpx.AsyncClient,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    token = sign_firebase_token(sub="firebase-uid-e2e", email="e2e@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    first = await client.get("/whoami", headers=headers)
    second = await client.get("/whoami", headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    # Same firebase_uid resolves to the same Cue user id every time - no
    # duplicate rows created on repeat sign-ins.
    assert first.json()["user_id"] == second.json()["user_id"]
