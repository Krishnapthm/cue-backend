"""Swiggy OAuth 2.1 + PKCE authorize/callback flow (R2.1/R2.5, CUE-7)."""

from __future__ import annotations

import base64
import hashlib
import logging
import secrets
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx
from cryptography.fernet import Fernet
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.provider import OAuthTransaction, ProviderLink
from app.models.user import User
from app.providers.config import provider_settings
from app.providers.constants import (
    CODE_CHALLENGE_METHOD,
    OAUTH_TRANSACTION_TTL_SECONDS,
    PROVIDER,
    SWIGGY_AUTHORIZE_URL,
    SWIGGY_SCOPE,
    SWIGGY_TOKEN_URL,
    TOKEN_ENCRYPTION_KEY_VERSION,
)
from app.providers.exceptions import (
    InvalidOAuthStateError,
    ProviderNotConfiguredError,
    SwiggyTokenExchangeError,
)
from app.providers.schemas import AuthorizeResponse

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Build the cipher from the configured key, or raise if unconfigured."""
    if provider_settings.TOKEN_ENCRYPTION_KEY is None:
        raise ProviderNotConfiguredError
    return Fernet(provider_settings.TOKEN_ENCRYPTION_KEY)


def _encrypt(plaintext: str) -> bytes:
    """Encrypt a secret for storage in an `*_ct` column."""
    return _get_fernet().encrypt(plaintext.encode())


def _decrypt(ciphertext: bytes) -> str:
    """Decrypt a secret previously stored by `_encrypt`."""
    return _get_fernet().decrypt(ciphertext).decode()


def _generate_pkce_pair() -> tuple[str, str]:
    """Return a (code_verifier, code_challenge) pair per RFC 7636 S256."""
    verifier = secrets.token_urlsafe(32)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


async def create_authorization(session: AsyncSession, user: User) -> AuthorizeResponse:
    """Start the Swiggy OAuth 2.1 + PKCE flow for `user` (R2.1).

    Args:
        session: An active database session.
        user: The Cue user linking their Swiggy account.

    Returns:
        The Swiggy consent URL the client should open.

    Raises:
        ProviderNotConfiguredError: If Swiggy OAuth is not configured on
            this deployment (e.g. local dev without a registered app).
    """
    client_id = provider_settings.CLIENT_ID
    redirect_uri = provider_settings.REDIRECT_URI
    if client_id is None or redirect_uri is None:
        raise ProviderNotConfiguredError

    verifier, challenge = _generate_pkce_pair()
    state = secrets.token_urlsafe(32)
    now = datetime.now(UTC)

    session.add(
        OAuthTransaction(
            state=state,
            user_id=user.id,
            provider=PROVIDER,
            code_verifier_ct=_encrypt(verifier),
            key_version=TOKEN_ENCRYPTION_KEY_VERSION,
            redirect_uri=redirect_uri,
            expires_at=now + timedelta(seconds=OAUTH_TRANSACTION_TTL_SECONDS),
        )
    )
    await session.commit()

    query = urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": CODE_CHALLENGE_METHOD,
            "state": state,
            "scope": SWIGGY_SCOPE,
        }
    )
    return AuthorizeResponse(authorize_url=f"{SWIGGY_AUTHORIZE_URL}?{query}")


async def _exchange_code(
    *, code: str, verifier: str, redirect_uri: str
) -> httpx.Response:
    """POST the authorization_code grant to Swiggy's token endpoint."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await client.post(
            SWIGGY_TOKEN_URL,
            json={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": redirect_uri,
            },
        )


async def complete_authorization(
    session: AsyncSession, *, state: str, code: str
) -> None:
    """Exchange a Swiggy authorization code for an access token and link it (R2.1).

    Args:
        session: An active database session.
        state: The state parameter Swiggy echoed back. Validated against the
            transaction this same backend created, which is the CSRF check:
            an attacker cannot guess an unconsumed, unexpired state value.
        code: The single-use authorization code Swiggy issued.

    Raises:
        InvalidOAuthStateError: If `state` is unknown, expired, or already used.
        SwiggyTokenExchangeError: If Swiggy rejects or fails the code exchange.
    """
    now = datetime.now(UTC)
    txn = await session.get(OAuthTransaction, state)
    if (
        txn is None
        or txn.provider != PROVIDER
        or txn.consumed_at is not None
        or txn.expires_at < now
    ):
        logger.warning("Rejected Swiggy OAuth callback with invalid state")
        raise InvalidOAuthStateError

    verifier = _decrypt(txn.code_verifier_ct)

    try:
        response = await _exchange_code(
            code=code, verifier=verifier, redirect_uri=txn.redirect_uri
        )
    except httpx.HTTPError as exc:
        logger.warning("Swiggy token exchange request failed: %s", exc)
        raise SwiggyTokenExchangeError from exc

    if response.status_code != httpx.codes.OK:
        logger.warning(
            "Swiggy token exchange rejected with status %s", response.status_code
        )
        raise SwiggyTokenExchangeError

    payload = response.json()
    access_token: str = payload["access_token"]
    expires_in: int = payload["expires_in"]
    scope: str = payload["scope"]

    insert_stmt = pg_insert(ProviderLink).values(
        user_id=txn.user_id,
        provider=PROVIDER,
        access_token_ct=_encrypt(access_token),
        key_version=TOKEN_ENCRYPTION_KEY_VERSION,
        token_expires_at=now + timedelta(seconds=expires_in),
        scope=scope,
        status="active",
        linked_at=now,
    )
    upsert_stmt = insert_stmt.on_conflict_do_update(
        constraint="uq_provider_link_user_provider",
        set_={
            "access_token_ct": insert_stmt.excluded.access_token_ct,
            "key_version": insert_stmt.excluded.key_version,
            "token_expires_at": insert_stmt.excluded.token_expires_at,
            "scope": insert_stmt.excluded.scope,
            "status": insert_stmt.excluded.status,
            "linked_at": insert_stmt.excluded.linked_at,
        },
    )
    await session.execute(upsert_stmt)
    txn.consumed_at = now
    await session.commit()
