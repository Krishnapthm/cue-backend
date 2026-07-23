from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provider import OAuthTransaction, ProviderLink
from app.models.user import User
from app.providers import service
from app.providers.config import provider_settings
from app.providers.constants import PROVIDER
from app.providers.exceptions import InvalidOAuthStateError, SwiggyTokenExchangeError
from app.providers.schemas import ProviderStatus
from tests.providers.conftest import SwiggyTokenEndpointStub


async def test_create_authorization_returns_a_swiggy_consent_url_with_pkce_params(
    db_session: AsyncSession, user: User
) -> None:
    response = await service.create_authorization(db_session, user)

    parsed = urlparse(response.authorize_url)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "mcp.swiggy.com"
    assert parsed.path == "/auth/authorize"
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert "code_challenge" in query
    assert "state" in query


async def test_create_authorization_persists_an_encrypted_verifier_for_the_state(
    db_session: AsyncSession, user: User
) -> None:
    response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(response.authorize_url).query)["state"][0]

    txn = await db_session.get(OAuthTransaction, state)

    assert txn is not None
    assert txn.user_id == user.id
    assert txn.provider == PROVIDER
    assert txn.consumed_at is None
    # The verifier is never stored in the clear.
    assert txn.code_verifier_ct != b""


async def test_complete_authorization_links_the_provider_on_success(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    authorize_response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(authorize_response.authorize_url).query)["state"][0]

    await service.complete_authorization(
        db_session, state=state, code="auth-code", user_id=user.id
    )

    stmt = select(ProviderLink).where(
        ProviderLink.user_id == user.id, ProviderLink.provider == PROVIDER
    )
    result = await db_session.execute(stmt)
    link = result.scalar_one()
    assert link.status == "active"
    assert link.token_expires_at > datetime.now(UTC)
    assert link.access_token_ct != b""


async def test_complete_authorization_consumes_the_state_so_it_cannot_replay(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    authorize_response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(authorize_response.authorize_url).query)["state"][0]
    await service.complete_authorization(
        db_session, state=state, code="auth-code", user_id=user.id
    )

    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(
            db_session, state=state, code="auth-code", user_id=user.id
        )


async def test_complete_authorization_rejects_an_unknown_state(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(
            db_session, state="never-issued", code="auth-code", user_id=None
        )


async def test_complete_authorization_rejects_an_expired_state(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        OAuthTransaction(
            state="expired-state",
            user_id=user.id,
            provider=PROVIDER,
            code_verifier_ct=b"ciphertext",
            redirect_uri="https://api.cue.test/providers/swiggy/callback",
            expires_at=datetime.now(UTC) - timedelta(seconds=1),
        )
    )
    await db_session.commit()

    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(
            db_session, state="expired-state", code="auth-code", user_id=user.id
        )


async def test_complete_authorization_raises_on_swiggy_token_exchange_failure(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    mock_swiggy_token_endpoint(status_code=400)
    authorize_response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(authorize_response.authorize_url).query)["state"][0]

    with pytest.raises(SwiggyTokenExchangeError):
        await service.complete_authorization(
            db_session, state=state, code="auth-code", user_id=user.id
        )


async def test_complete_authorization_upserts_on_relink_after_expiry(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    """A second successful link for the same user replaces the row in place."""
    first = await service.create_authorization(db_session, user)
    first_state = parse_qs(urlparse(first.authorize_url).query)["state"][0]
    await service.complete_authorization(
        db_session, state=first_state, code="code-1", user_id=user.id
    )

    second = await service.create_authorization(db_session, user)
    second_state = parse_qs(urlparse(second.authorize_url).query)["state"][0]
    await service.complete_authorization(
        db_session, state=second_state, code="code-2", user_id=user.id
    )

    stmt = select(ProviderLink).where(
        ProviderLink.user_id == user.id, ProviderLink.provider == PROVIDER
    )
    result = await db_session.execute(stmt)
    links = result.scalars().all()
    assert len(links) == 1
    assert links[0].status == "active"


async def test_create_authorization_returns_the_configured_redirect_uri(
    db_session: AsyncSession, user: User
) -> None:
    response = await service.create_authorization(db_session, user)

    assert response.redirect_uri == provider_settings.REDIRECT_URI
    query = parse_qs(urlparse(response.authorize_url).query)
    assert query["redirect_uri"] == [response.redirect_uri]


async def test_complete_authorization_rejects_a_state_owned_by_another_user(
    db_session: AsyncSession,
    user: User,
    other_user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    """A signed-in user cannot redeem a state another Cue user's flow issued."""
    authorize_response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(authorize_response.authorize_url).query)["state"][0]

    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(
            db_session, state=state, code="auth-code", user_id=other_user.id
        )

    assert mock_swiggy_token_endpoint.requests == []
    assert await service.get_link(db_session, user.id) is None
    assert await service.get_link(db_session, other_user.id) is None


async def test_complete_authorization_replays_the_persisted_redirect_uri(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: SwiggyTokenEndpointStub,
) -> None:
    """Swiggy requires the token call's redirect_uri to match the authorize call."""
    authorize_response = await service.create_authorization(db_session, user)
    state = parse_qs(urlparse(authorize_response.authorize_url).query)["state"][0]
    txn = await db_session.get(OAuthTransaction, state)
    assert txn is not None

    await service.complete_authorization(
        db_session, state=state, code="auth-code", user_id=user.id
    )

    body = mock_swiggy_token_endpoint.request_bodies[0]
    assert body["redirect_uri"] == txn.redirect_uri
    assert body["grant_type"] == "authorization_code"
    assert body["code"] == "auth-code"


async def test_get_link_returns_none_when_not_linked(
    db_session: AsyncSession, user: User
) -> None:
    link = await service.get_link(db_session, user.id)

    assert link is None


async def test_get_link_returns_the_row_when_linked(
    db_session: AsyncSession, user: User, active_link: ProviderLink
) -> None:
    link = await service.get_link(db_session, user.id)

    assert link is not None
    assert link.id == active_link.id


async def test_get_decrypted_access_token_returns_none_when_not_linked(
    db_session: AsyncSession, user: User
) -> None:
    token = await service.get_decrypted_access_token(db_session, user.id)

    assert token is None


async def test_get_decrypted_access_token_decrypts_an_active_link(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=service._encrypt("at_live_token"),
            token_expires_at=datetime.now(UTC) + timedelta(days=5),
            scope="mcp:tools",
            status="active",
        )
    )
    await db_session.commit()

    token = await service.get_decrypted_access_token(db_session, user.id)

    assert token == "at_live_token"


async def test_get_decrypted_access_token_returns_none_when_status_expired(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=service._encrypt("at_live_token"),
            token_expires_at=datetime.now(UTC) + timedelta(days=5),
            scope="mcp:tools",
            status="expired",
        )
    )
    await db_session.commit()

    token = await service.get_decrypted_access_token(db_session, user.id)

    assert token is None


async def test_get_decrypted_access_token_returns_none_when_clock_expired(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=service._encrypt("at_live_token"),
            token_expires_at=datetime.now(UTC) - timedelta(seconds=1),
            scope="mcp:tools",
            status="active",
        )
    )
    await db_session.commit()

    token = await service.get_decrypted_access_token(db_session, user.id)

    assert token is None


async def test_mark_link_expired_flips_status_without_touching_other_columns(
    db_session: AsyncSession, user: User, active_link: ProviderLink
) -> None:
    await service.mark_link_expired(db_session, user.id)

    stmt = select(ProviderLink).where(ProviderLink.id == active_link.id)
    result = await db_session.execute(stmt)
    link = result.scalar_one()
    assert link.status == "expired"
    assert link.access_token_ct == active_link.access_token_ct
    assert link.token_expires_at == active_link.token_expires_at


async def test_mark_link_expired_is_a_noop_when_no_link_exists(
    db_session: AsyncSession, user: User
) -> None:
    # No provider_link row for this user; must not raise.
    await service.mark_link_expired(db_session, user.id)

    link = await service.get_link(db_session, user.id)
    assert link is None


@pytest.mark.parametrize("status_code", [401, 419])
async def test_record_provider_response_marks_expired_on_recoverable_status(
    db_session: AsyncSession,
    user: User,
    active_link: ProviderLink,
    status_code: int,
) -> None:
    await service.record_provider_response(db_session, user.id, status_code)

    link = await service.get_link(db_session, user.id)
    assert link is not None
    assert link.status == "expired"


@pytest.mark.parametrize("status_code", [200, 403, 500])
async def test_record_provider_response_leaves_link_active_on_other_statuses(
    db_session: AsyncSession,
    user: User,
    active_link: ProviderLink,
    status_code: int,
) -> None:
    await service.record_provider_response(db_session, user.id, status_code)

    link = await service.get_link(db_session, user.id)
    assert link is not None
    assert link.status == "active"


async def test_get_link_status_is_not_connected_when_never_linked(
    db_session: AsyncSession, user: User
) -> None:
    link_status = await service.get_link_status(db_session, user.id)

    assert link_status == ProviderStatus.NOT_CONNECTED


async def test_get_link_status_is_connected_for_an_active_unexpired_link(
    db_session: AsyncSession, user: User, active_link: ProviderLink
) -> None:
    link_status = await service.get_link_status(db_session, user.id)

    assert link_status == ProviderStatus.CONNECTED


async def test_get_link_status_is_reconnect_needed_after_recovery_ladder_expiry(
    db_session: AsyncSession, user: User, active_link: ProviderLink
) -> None:
    await service.mark_link_expired(db_session, user.id)

    link_status = await service.get_link_status(db_session, user.id)

    assert link_status == ProviderStatus.RECONNECT_NEEDED


async def test_get_link_status_is_reconnect_needed_once_the_token_clock_expires(
    db_session: AsyncSession, user: User
) -> None:
    db_session.add(
        ProviderLink(
            user_id=user.id,
            provider=PROVIDER,
            access_token_ct=b"ciphertext",
            token_expires_at=datetime.now(UTC) - timedelta(seconds=1),
            scope="mcp:tools",
            status="active",
        )
    )
    await db_session.commit()

    link_status = await service.get_link_status(db_session, user.id)

    assert link_status == ProviderStatus.RECONNECT_NEEDED


async def test_unlink_removes_the_link_row(
    db_session: AsyncSession, user: User, active_link: ProviderLink
) -> None:
    await service.unlink(db_session, user.id)

    link = await service.get_link(db_session, user.id)
    assert link is None


async def test_unlink_is_a_noop_when_not_linked(
    db_session: AsyncSession, user: User
) -> None:
    # Must not raise for a user who was never linked, or already unlinked.
    await service.unlink(db_session, user.id)

    link = await service.get_link(db_session, user.id)
    assert link is None
