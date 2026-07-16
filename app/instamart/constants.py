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
TOOL_UPDATE_CART = "update_cart"
TOOL_GET_CART = "get_cart"
TOOL_CHECKOUT = "checkout"
TOOL_GET_ORDERS = "get_orders"
TOOL_GET_ORDER_DETAILS = "get_order_details"
TOOL_YOUR_GO_TO_ITEMS = "your_go_to_items"

DEFAULT_SEARCH_OFFSET = 0
# Matches get_orders' own documented default.
DEFAULT_GET_ORDERS_COUNT = 10
# get_orders' documented maximum; higher requests are clamped client-side.
MAX_GET_ORDERS_COUNT = 20
# get_orders' own tool default is "DASH" (food delivery), which is wrong for
# Cue - we always send this explicitly instead of relying on Swiggy's default.
DEFAULT_ORDER_TYPE = "INSTAMART"

REQUEST_TIMEOUT_SECONDS = 10.0

# HTTP statuses documented as auth failure - token expired (401) or the
# Swiggy-side session revoked (419) - recoverable only via a fresh OAuth
# authorize (R2.5). Matches app.providers.constants.RECOVERABLE_PROVIDER_STATUS_CODES.
AUTH_FAILURE_STATUS_CODES = frozenset({401, 419})

# JSON-RPC error code documented for auth failure at the protocol level,
# distinct from an HTTP 401 (errors reference: "-32001").
JSONRPC_AUTH_ERROR_CODE = -32001
