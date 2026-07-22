"""`GET /auth/me` and the CORS middleware (CUE-59).

These go through the real `current_user` dependency rather than overriding it:
the point of the endpoint is that it exercises (and can be used to observe)
the provisioning upsert, so an override would test nothing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.database import get_session
from app.main import app
from app.models.user import User

ALLOWED_ORIGIN = "http://localhost:8081"
DISALLOWED_ORIGIN = "https://evil.example"


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
async def concurrent_client(postgres_url: str) -> AsyncGenerator[httpx.AsyncClient]:
    """A client whose requests each get their own database session.

    The shared `db_session` fixture hands every request the *same* session,
    which is fine for sequential tests but cannot model two requests racing -
    SQLAlchemy rejects concurrent operations on one session. This override
    mirrors production, where each request opens its own session.
    """
    engine = create_async_engine(postgres_url)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async def override_get_session() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()
    await engine.dispose()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def test_me_returns_the_signed_in_user(
    client: httpx.AsyncClient,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    token = sign_firebase_token(
        sub="firebase-uid-me", email="me@example.com", name="Me Myself"
    )

    response = await client.get("/auth/me", headers=_auth(token))

    assert response.status_code == 200
    body = response.json()
    assert body["email"] == "me@example.com"
    assert body["display_name"] == "Me Myself"
    assert isinstance(body["id"], int)
    # firebase_uid is the identity, not payload - it is never echoed back.
    assert "firebase_uid" not in body


async def test_me_provisions_the_user_row_on_first_call(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    uid = "firebase-uid-brand-new"
    assert await db_session.scalar(select(User).where(User.firebase_uid == uid)) is None

    response = await client.get(
        "/auth/me", headers=_auth(sign_firebase_token(sub=uid, email="new@example.com"))
    )

    assert response.status_code == 200
    persisted = await db_session.scalar(select(User).where(User.firebase_uid == uid))
    assert persisted is not None
    assert persisted.id == response.json()["id"]


async def test_me_refreshes_email_and_display_name_on_later_calls(
    client: httpx.AsyncClient,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    uid = "firebase-uid-renamed"
    first = await client.get(
        "/auth/me",
        headers=_auth(
            sign_firebase_token(sub=uid, email="old@example.com", name="Old")
        ),
    )
    second = await client.get(
        "/auth/me",
        headers=_auth(
            sign_firebase_token(sub=uid, email="new@example.com", name="New")
        ),
    )

    assert first.json()["id"] == second.json()["id"]
    assert second.json()["email"] == "new@example.com"
    assert second.json()["display_name"] == "New"


async def test_concurrent_first_calls_for_one_uid_provision_one_row(
    concurrent_client: httpx.AsyncClient,
    db_session: AsyncSession,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    # The upsert in `current_user` is conflict-aware; this proves it, now that
    # an endpoint makes the race easy to hit directly.
    uid = "firebase-uid-raced"
    headers = _auth(sign_firebase_token(sub=uid, email="raced@example.com"))

    responses = await asyncio.gather(
        concurrent_client.get("/auth/me", headers=headers),
        concurrent_client.get("/auth/me", headers=headers),
    )

    assert [r.status_code for r in responses] == [200, 200]
    rows = (
        await db_session.scalars(select(User).where(User.firebase_uid == uid))
    ).all()
    assert len(rows) == 1
    assert {r.json()["id"] for r in responses} == {rows[0].id}


async def test_me_without_authorization_header_is_401(
    client: httpx.AsyncClient,
) -> None:
    response = await client.get("/auth/me")

    # 401, not HTTPBearer's default 403, and in the standard detail envelope.
    assert response.status_code == 401
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "token",
    ["not-a-jwt", "eyJhbGciOiJIUzI1NiJ9.e30.sig"],
    ids=["malformed", "unsigned-by-firebase"],
)
async def test_me_with_a_bad_token_is_401(
    client: httpx.AsyncClient, mock_jwks_fetch: None, token: str
) -> None:
    response = await client.get("/auth/me", headers=_auth(token))

    assert response.status_code == 401


async def test_me_with_an_expired_token_is_401(
    client: httpx.AsyncClient,
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
) -> None:
    response = await client.get(
        "/auth/me", headers=_auth(sign_firebase_token(exp=1_000_000))
    )

    assert response.status_code == 401


async def test_preflight_from_an_allowed_origin_gets_cors_headers(
    client: httpx.AsyncClient,
) -> None:
    response = await client.options(
        "/auth/me",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ALLOWED_ORIGIN
    assert "authorization" in response.headers["access-control-allow-headers"].lower()


async def test_preflight_from_a_disallowed_origin_gets_no_cors_headers(
    client: httpx.AsyncClient,
) -> None:
    response = await client.options(
        "/auth/me",
        headers={
            "Origin": DISALLOWED_ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


def test_cors_allow_origins_defaults_to_empty() -> None:
    # Fail closed: an unconfigured deployment installs the middleware but
    # permits nothing, rather than shipping wide open by accident.
    assert Settings.model_fields["CORS_ALLOW_ORIGINS"].default == []
