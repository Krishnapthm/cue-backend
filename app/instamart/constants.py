"""Fixed Swiggy Instamart MCP parameters (not environment-specific).

Docs: https://mcp.swiggy.com/builders/docs/reference/instamart/,
https://mcp.swiggy.com/builders/docs/reference/errors.md
"""

from __future__ import annotations

INSTAMART_MCP_ENDPOINT = "https://mcp.swiggy.com/im"

TOOL_GET_ADDRESSES = "get_addresses"
TOOL_CREATE_ADDRESS = "create_address"
TOOL_DELETE_ADDRESS = "delete_address"
TOOL_SEARCH_PRODUCTS = "search_products"

DEFAULT_SEARCH_OFFSET = 0

REQUEST_TIMEOUT_SECONDS = 10.0

# HTTP statuses documented as auth failure - token expired (401) or the
# Swiggy-side session revoked (419) - recoverable only via a fresh OAuth
# authorize (R2.5). Matches app.providers.constants.RECOVERABLE_PROVIDER_STATUS_CODES.
AUTH_FAILURE_STATUS_CODES = frozenset({401, 419})

# JSON-RPC error code documented for auth failure at the protocol level,
# distinct from an HTTP 401 (errors reference: "-32001").
JSONRPC_AUTH_ERROR_CODE = -32001
