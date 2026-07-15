"""Fixed Firebase Auth verification parameters (not environment-specific)."""

from __future__ import annotations

FIREBASE_JWKS_URL = (
    "https://www.googleapis.com/service_accounts/v1/jwk/"
    "securetoken@system.gserviceaccount.com"
)
FIREBASE_ALGORITHM = "RS256"

# Google rotates these keys infrequently; re-fetch at most this often unless an
# unrecognized `kid` forces an early refresh.
JWKS_CACHE_TTL_SECONDS = 3600
