from __future__ import annotations

import time
from collections.abc import Callable

import pytest
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from app.auth import service
from app.auth.exceptions import InvalidTokenError


async def test_verify_firebase_id_token_accepts_a_valid_token(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    token = sign_firebase_token(
        sub="firebase-uid-valid", email="valid@example.com", name="Valid User"
    )

    claims = await service.verify_firebase_id_token(token)

    assert claims.sub == "firebase-uid-valid"
    assert claims.email == "valid@example.com"
    assert claims.name == "Valid User"


async def test_verify_firebase_id_token_rejects_expired_token(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    now = int(time.time())
    token = sign_firebase_token(exp=now - 60, iat=now - 3600)

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_wrong_audience(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    token = sign_firebase_token(aud="someone-elses-project")

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_wrong_issuer(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    token = sign_firebase_token(iss="https://securetoken.google.com/some-other-project")

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_bad_signature(
    mock_jwks_fetch: None,
    sign_firebase_token: Callable[..., str],
    other_private_key: RSAPrivateKey,
) -> None:
    # Signed with a key that never appears in the published JWKS.
    token = sign_firebase_token(key=other_private_key)

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_missing_kid(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    token = sign_firebase_token(kid=None)

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_unrecognized_kid(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    token = sign_firebase_token(kid="some-rotated-out-key")

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_future_issued_at(
    mock_jwks_fetch: None, sign_firebase_token: Callable[..., str]
) -> None:
    now = int(time.time())
    token = sign_firebase_token(iat=now + 3600)

    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token(token)


async def test_verify_firebase_id_token_rejects_malformed_token(
    mock_jwks_fetch: None,
) -> None:
    with pytest.raises(InvalidTokenError):
        await service.verify_firebase_id_token("not-a-jwt")
