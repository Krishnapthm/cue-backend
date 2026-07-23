from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_session
from app.main import app
from app.models.provider import OAuthTransaction, ProviderLink
from app.models.user import User
from app.providers import service
from app.providers.constants import PROVIDER
from tests.providers.conftest import SwiggyTokenEndpointStub


async def _get_link(session: AsyncSession, user: User) -> ProviderLink | None:
    """Read the user's Swiggy link straight from the database."""
    stmt = select(ProviderLink).where(
        ProviderLink.user_id == user.id, ProviderLink.provider == PROVIDER
    )
    return (await session.execute(stmt)).scalar_one_or_none()


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


async def _issue_state(authed_client: httpx.AsyncClient) -> str:
    """Run POST /authorize and return the `state` it minted."""
    response = await authed_client.post("/providers/swiggy/authorize")
    assert response.status_code == 200
    state: str = parse_qs(urlparse(response.json()["authorize_url"]).query)["state"][0]
    return state


async def test_authorize_returns_the_registered_redirect_uri(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post("/providers/swiggy/authorize")

    assert response.status_code == 200
    assert (
        response.json()["redirect_uri"]
        == "https://api.cue.test/providers/swiggy/callback"
    )


async def test_post_callback_requires_authentication(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/providers/swiggy/callback", json={"code": "auth-code", "state": "some-state"}
    )

    assert response.status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"state": "some-state"},
        {"code": "auth-code"},
        {"code": "", "state": "some-state"},
        {"code": "auth-code", "state": ""},
        {},
    ],
)
async def test_post_callback_rejects_a_missing_or_empty_field(
    authed_client: httpx.AsyncClient, body: dict[str, str]
) -> None:
    response = await authed_client.post("/providers/swiggy/callback", json=body)

    assert response.status_code == 422


async def test_post_callback_links_the_provider_and_returns_connected(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    state = await _issue_state(authed_client)

    response = await authed_client.post(
        "/providers/swiggy/callback", json={"code": "auth-code", "state": state}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "connected"}

    txn = await db_session.get(OAuthTransaction, state)
    assert txn is not None
    assert txn.consumed_at is not None

    link = await _get_link(db_session, user)
    assert link is not None
    assert link.status == "active"


async def test_post_callback_sends_the_persisted_redirect_uri_to_swiggy(
    authed_client: httpx.AsyncClient,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    state = await _issue_state(authed_client)

    await authed_client.post(
        "/providers/swiggy/callback", json={"code": "auth-code", "state": state}
    )

    body = mock_swiggy_token_endpoint.request_bodies[0]
    assert body["redirect_uri"] == "https://api.cue.test/providers/swiggy/callback"
    assert body["code"] == "auth-code"
    assert body["grant_type"] == "authorization_code"


async def test_post_callback_rejects_a_replayed_code_and_keeps_the_link(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    state = await _issue_state(authed_client)
    body = {"code": "auth-code", "state": state}
    first = await authed_client.post("/providers/swiggy/callback", json=body)
    assert first.status_code == 200
    link = await _get_link(db_session, user)
    assert link is not None
    original_token_ct = link.access_token_ct

    response = await authed_client.post("/providers/swiggy/callback", json=body)

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid or expired authorization state."}

    link = await _get_link(db_session, user)
    assert link is not None
    await db_session.refresh(link)
    assert link.status == "active"
    assert link.access_token_ct == original_token_ct


async def test_post_callback_rejects_an_unknown_state(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.post(
        "/providers/swiggy/callback",
        json={"code": "auth-code", "state": "never-issued"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid or expired authorization state."}


async def test_post_callback_rejects_an_expired_state(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
) -> None:
    db_session.add(
        OAuthTransaction(
            state="expired-state-post-callback",
            user_id=user.id,
            provider=PROVIDER,
            code_verifier_ct=b"ciphertext",
            redirect_uri="https://api.cue.test/providers/swiggy/callback",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    await db_session.commit()

    response = await authed_client.post(
        "/providers/swiggy/callback",
        json={"code": "auth-code", "state": "expired-state-post-callback"},
    )

    assert response.status_code == 400
    assert await _get_link(db_session, user) is None


async def test_post_callback_rejects_a_state_belonging_to_another_user(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    other_user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    """The signed-in caller must own the transaction, or anyone holding a
    `state` could bind their Swiggy account onto another Cue user."""
    db_session.add(
        OAuthTransaction(
            state="other-users-state",
            user_id=other_user.id,
            provider=PROVIDER,
            code_verifier_ct=b"ciphertext",
            redirect_uri="https://api.cue.test/providers/swiggy/callback",
            expires_at=datetime.now(UTC) + timedelta(seconds=600),
        )
    )
    await db_session.commit()

    response = await authed_client.post(
        "/providers/swiggy/callback",
        json={"code": "auth-code", "state": "other-users-state"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid or expired authorization state."}
    assert mock_swiggy_token_endpoint.requests == []
    assert await _get_link(db_session, user) is None
    assert await _get_link(db_session, other_user) is None


async def test_post_callback_returns_502_when_swiggy_rejects_the_exchange(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    mock_swiggy_token_endpoint(status_code=400)
    state = await _issue_state(authed_client)

    response = await authed_client.post(
        "/providers/swiggy/callback", json={"code": "auth-code", "state": state}
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Failed to exchange authorization code with Swiggy."
    }
    assert await _get_link(db_session, user) is None


async def test_post_callback_reconnects_a_user_whose_link_expired(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    active_link: ProviderLink,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    """Relinking upserts the existing row rather than tripping the unique index."""
    await service.mark_link_expired(db_session, user.id)
    status_before = await authed_client.get("/providers/swiggy/status")
    assert status_before.json() == {"status": "reconnect_needed"}
    state = await _issue_state(authed_client)

    response = await authed_client.post(
        "/providers/swiggy/callback", json={"code": "auth-code", "state": state}
    )

    assert response.status_code == 200
    assert response.json() == {"status": "connected"}

    stmt = select(ProviderLink).where(
        ProviderLink.user_id == user.id, ProviderLink.provider == PROVIDER
    )
    links = (await db_session.execute(stmt)).scalars().all()
    assert len(links) == 1
    assert links[0].status == "active"


async def test_status_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.get("/providers/swiggy/status")

    assert response.status_code == 401


async def test_status_is_not_connected_when_never_linked(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.get("/providers/swiggy/status")

    assert response.status_code == 200
    assert response.json() == {"status": "not_connected"}


async def test_status_is_connected_after_a_successful_link(
    authed_client: httpx.AsyncClient,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    authorize_response = await authed_client.post("/providers/swiggy/authorize")
    state = parse_qs(urlparse(authorize_response.json()["authorize_url"]).query)[
        "state"
    ][0]
    await authed_client.get(
        "/providers/swiggy/callback", params={"code": "auth-code", "state": state}
    )

    response = await authed_client.get("/providers/swiggy/status")

    assert response.status_code == 200
    assert response.json() == {"status": "connected"}


async def test_unlink_requires_authentication(client: httpx.AsyncClient) -> None:
    response = await client.delete("/providers/swiggy")

    assert response.status_code == 401


async def test_unlink_removes_the_link_and_status_reads_not_connected(
    authed_client: httpx.AsyncClient,
    db_session: AsyncSession,
    user: User,
    active_link: ProviderLink,
) -> None:
    response = await authed_client.delete("/providers/swiggy")

    assert response.status_code == 204

    status_response = await authed_client.get("/providers/swiggy/status")
    assert status_response.json() == {"status": "not_connected"}


async def test_unlink_is_idempotent_when_already_not_connected(
    authed_client: httpx.AsyncClient,
) -> None:
    response = await authed_client.delete("/providers/swiggy")

    assert response.status_code == 204
