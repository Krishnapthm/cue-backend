"""Fixed Swiggy OAuth parameters (not environment-specific)."""

from __future__ import annotations

PROVIDER = "swiggy"

SWIGGY_BASE_URL = "https://mcp.swiggy.com"
SWIGGY_AUTHORIZE_URL = f"{SWIGGY_BASE_URL}/auth/authorize"
SWIGGY_TOKEN_URL = f"{SWIGGY_BASE_URL}/auth/token"
SWIGGY_SCOPE = "mcp:tools"
CODE_CHALLENGE_METHOD = "S256"

# Window to complete the Swiggy consent screen and be redirected back before
# the state is treated as expired. Well above Swiggy's 120s auth-code TTL
# since the user interacts with Swiggy's own page for part of this window.
OAUTH_TRANSACTION_TTL_SECONDS = 600

# provider_link.key_version / oauth_transaction.key_version: only one
# encryption key is configured today; the column exists for future rotation.
TOKEN_ENCRYPTION_KEY_VERSION = 1

# 401 (expired/invalid access token) and 419 (session revoked) are the only
# Swiggy statuses recoverable via the recovery ladder (R2.5), and only by
# re-running OAuth authorize - Swiggy MCP v1.0 issues no refresh token.
RECOVERABLE_PROVIDER_STATUS_CODES = frozenset({401, 419})
