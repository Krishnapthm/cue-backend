from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from jwt.algorithms import RSAAlgorithm

from app.auth.config import auth_settings

_KID = "test-key-1"


@pytest.fixture(scope="session")
def firebase_private_key() -> RSAPrivateKey:
    """The RSA key the test JWKS advertises as `test-key-1`."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def other_private_key() -> RSAPrivateKey:
    """A second key never published in the test JWKS, for bad-signature tests."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="session")
def firebase_jwks(firebase_private_key: RSAPrivateKey) -> dict[str, Any]:
    """The JWKS document Google would serve for the test signing key."""
    public_jwk = RSAAlgorithm.to_jwk(firebase_private_key.public_key(), as_dict=True)
    public_jwk["kid"] = _KID
    public_jwk["alg"] = "RS256"
    public_jwk["use"] = "sig"
    return {"keys": [public_jwk]}


@pytest.fixture
def mock_jwks_fetch(
    monkeypatch: pytest.MonkeyPatch, firebase_jwks: dict[str, Any]
) -> None:
    """Point `app.auth.service`'s JWKS fetch at the fixture keyset instead of Google."""
    from app.auth import service

    async def _fake_fetch() -> dict[str, Any]:
        return firebase_jwks

    monkeypatch.setattr(service, "_fetch_jwks", _fake_fetch)
    service._cached_keys.clear()
    service._cached_at = 0.0


@pytest.fixture
def sign_firebase_token(firebase_private_key: RSAPrivateKey) -> Callable[..., str]:
    """Return a callable that mints a Firebase-shaped ID token, claims overridable."""

    def _sign(
        *,
        sub: str | None = None,
        aud: str = auth_settings.FIREBASE_PROJECT_ID,
        iss: str = auth_settings.firebase_issuer,
        exp: int | None = None,
        iat: int | None = None,
        auth_time: int | None = None,
        email: str = "user@example.com",
        email_verified: bool = True,
        name: str | None = "Test User",
        kid: str | None = _KID,
        key: RSAPrivateKey | None = None,
    ) -> str:
        now = int(time.time())
        payload = {
            "sub": sub if sub is not None else f"firebase-uid-{uuid.uuid4()}",
            "aud": aud,
            "iss": iss,
            "iat": iat if iat is not None else now,
            "exp": exp if exp is not None else now + 3600,
            "auth_time": auth_time if auth_time is not None else now,
            "email": email,
            "email_verified": email_verified,
            "name": name,
        }
        headers = {"kid": kid} if kid is not None else {}
        return jwt.encode(
            payload,
            key=key or firebase_private_key,
            algorithm="RS256",
            headers=headers,
        )

    return _sign
