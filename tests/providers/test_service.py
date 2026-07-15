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
from app.providers.constants import PROVIDER
from app.providers.exceptions import InvalidOAuthStateError, SwiggyTokenExchangeError


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

    await service.complete_authorization(db_session, state=state, code="auth-code")

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
    await service.complete_authorization(db_session, state=state, code="auth-code")

    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(db_session, state=state, code="auth-code")


async def test_complete_authorization_rejects_an_unknown_state(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(InvalidOAuthStateError):
        await service.complete_authorization(
            db_session, state="never-issued", code="auth-code"
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
            db_session, state="expired-state", code="auth-code"
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
        await service.complete_authorization(db_session, state=state, code="auth-code")


async def test_complete_authorization_upserts_on_relink_after_expiry(
    db_session: AsyncSession,
    user: User,
    mock_swiggy_token_endpoint: Callable[..., None],
) -> None:
    """A second successful link for the same user replaces the row in place."""
    first = await service.create_authorization(db_session, user)
    first_state = parse_qs(urlparse(first.authorize_url).query)["state"][0]
    await service.complete_authorization(db_session, state=first_state, code="code-1")

    second = await service.create_authorization(db_session, user)
    second_state = parse_qs(urlparse(second.authorize_url).query)["state"][0]
    await service.complete_authorization(db_session, state=second_state, code="code-2")

    stmt = select(ProviderLink).where(
        ProviderLink.user_id == user.id, ProviderLink.provider == PROVIDER
    )
    result = await db_session.execute(stmt)
    links = result.scalars().all()
    assert len(links) == 1
    assert links[0].status == "active"
