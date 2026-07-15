"""Firebase ID token verification (R2.0, CUE-6)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from jwt.algorithms import RSAAlgorithm
from pydantic import ValidationError

from app.auth.config import auth_settings
from app.auth.constants import (
    FIREBASE_ALGORITHM,
    FIREBASE_JWKS_URL,
    JWKS_CACHE_TTL_SECONDS,
)
from app.auth.exceptions import InvalidTokenError
from app.auth.schemas import FirebaseClaims

logger = logging.getLogger(__name__)

_cache_lock = asyncio.Lock()
_cached_keys: dict[str, RSAPublicKey] = {}
_cached_at: float = 0.0


async def _fetch_jwks() -> dict[str, Any]:
    """Fetch Google's current signing keys for Firebase ID tokens."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        response = await client.get(FIREBASE_JWKS_URL)
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        return data


async def _get_signing_key(kid: str) -> RSAPublicKey:
    """Return the RSA public key matching `kid`, refreshing the cache if stale.

    Args:
        kid: The `kid` header claim from the token being verified.

    Returns:
        The RSA public key to verify the token's signature against.

    Raises:
        InvalidTokenError: If no known key matches `kid`, even after a refresh.
    """
    global _cached_at

    async with _cache_lock:
        stale = time.monotonic() - _cached_at > JWKS_CACHE_TTL_SECONDS
        if stale or kid not in _cached_keys:
            jwks = await _fetch_jwks()
            _cached_keys.clear()
            for jwk in jwks["keys"]:
                public_key = RSAAlgorithm.from_jwk(json.dumps(jwk))
                # JWKS is a public keyset; never carries a private key.
                assert isinstance(public_key, RSAPublicKey)
                _cached_keys[jwk["kid"]] = public_key
            _cached_at = time.monotonic()

        try:
            return _cached_keys[kid]
        except KeyError:
            raise InvalidTokenError("Unrecognized signing key.") from None


async def verify_firebase_id_token(token: str) -> FirebaseClaims:
    """Validate a Firebase ID token's signature, issuer, audience, and expiry.

    Args:
        token: The raw JWT from the `Authorization: Bearer` header.

    Returns:
        The verified claims, including the stable Firebase `sub` (uid).

    Raises:
        InvalidTokenError: If the token is malformed, expired, or otherwise
            fails verification.
    """
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as exc:
        raise InvalidTokenError("Token header is malformed.") from exc

    kid = header.get("kid")
    if not kid:
        raise InvalidTokenError("Token is missing a key id.")

    key = await _get_signing_key(kid)

    try:
        payload = jwt.decode(
            token,
            key=key,
            algorithms=[FIREBASE_ALGORITHM],
            audience=auth_settings.FIREBASE_PROJECT_ID,
            issuer=auth_settings.firebase_issuer,
        )
    except jwt.PyJWTError as exc:
        raise InvalidTokenError from exc

    # PyJWT only checks `exp`; Firebase additionally requires `iat` and
    # `auth_time` (when present) to be in the past.
    now = time.time()
    issued_at = payload.get("iat")
    auth_time = payload.get("auth_time")
    if not isinstance(issued_at, int | float) or issued_at > now:
        raise InvalidTokenError("Token issued-at time is invalid.")
    if auth_time is not None and (
        not isinstance(auth_time, int | float) or auth_time > now
    ):
        raise InvalidTokenError("Token auth time is invalid.")

    try:
        return FirebaseClaims.model_validate(payload)
    except ValidationError as exc:
        raise InvalidTokenError("Token claims are malformed.") from exc
