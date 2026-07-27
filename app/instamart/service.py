"""Instamart tool wrappers: addresses (CUE-10), search_products (CUE-11),
cart (CUE-16), checkout (CUE-19), order history + details (CUE-13),
track_order (CUE-14).

Every call resolves the user's live Swiggy access token first (R2.5): no
usable token means "not linked or reconnect needed", which is routed through
`InstamartAuthError` exactly like a live auth failure - never a raw failure.
A live 401/419 from Swiggy additionally marks the link expired so the next
call short-circuits the same way instead of re-hitting Swiggy with a token
we now know is dead.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import client
from app.instamart.constants import (
    DEFAULT_GET_ORDERS_COUNT,
    DEFAULT_ORDER_TYPE,
    DEFAULT_SEARCH_OFFSET,
    MAX_GET_ORDERS_COUNT,
    TOOL_CHECKOUT,
    TOOL_CREATE_ADDRESS,
    TOOL_DELETE_ADDRESS,
    TOOL_GET_ADDRESSES,
    TOOL_GET_CART,
    TOOL_GET_ORDER_DETAILS,
    TOOL_GET_ORDERS,
    TOOL_SEARCH_PRODUCTS,
    TOOL_TRACK_ORDER,
    TOOL_UPDATE_CART,
    TOOL_YOUR_GO_TO_ITEMS,
)
from app.instamart.exceptions import InstamartAuthError
from app.instamart.schemas import (
    Address,
    Cart,
    CartItemInput,
    CheckoutResult,
    CreateAddressRequest,
    GoToItem,
    OrderDetails,
    OrderSummary,
    OrderTracking,
    PreferenceSignal,
    Product,
)
from app.providers import service as provider_service


async def resolve_access_token(session: AsyncSession, user_id: int) -> str:
    """Return the user's live Swiggy access token, or fail like a dead session.

    Exposed for callers that fan several tool calls out concurrently: a single
    `AsyncSession` cannot be used from more than one coroutine at a time, so
    the token is resolved once, serially, and the concurrent leg is then pure
    HTTP via the `*_with_token` wrappers.

    Args:
        session: An active database session.
        user_id: The user whose Swiggy link to read.

    Returns:
        The decrypted access token.

    Raises:
        InstamartAuthError: If the account is not linked, or the link is
            expired - indistinguishable to the caller, and recoverable only
            by a fresh OAuth authorize.
    """
    token = await provider_service.get_decrypted_access_token(session, user_id)
    if token is None:
        raise InstamartAuthError
    return token


async def _call_authenticated(
    session: AsyncSession, user_id: int, tool_name: str, arguments: dict[str, Any]
) -> Any:
    """Call an Instamart tool with the user's token, applying the recovery ladder."""
    token = await resolve_access_token(session, user_id)
    try:
        return await client.call_tool(token, tool_name, arguments)
    except InstamartAuthError:
        await provider_service.mark_link_expired(session, user_id)
        raise


async def get_addresses(session: AsyncSession, user_id: int) -> list[Address]:
    """Return the user's saved Swiggy delivery addresses (R3.1 smoke test)."""
    data = await _call_authenticated(session, user_id, TOOL_GET_ADDRESSES, {})
    # The envelope key holding the list isn't pinned by Swiggy's docs; accept
    # either a bare list or one nested under "addresses".
    raw_addresses = data.get("addresses", []) if isinstance(data, dict) else data or []
    return [Address.model_validate(item) for item in raw_addresses]


async def create_address(
    session: AsyncSession, user_id: int, request: CreateAddressRequest
) -> Address:
    """Create a new Swiggy delivery address for the user (R3.1)."""
    arguments = request.model_dump(by_alias=True, exclude_none=True)
    data = await _call_authenticated(session, user_id, TOOL_CREATE_ADDRESS, arguments)
    # Same envelope ambiguity as get_addresses: accept the address nested
    # under "address" or returned as the data payload directly.
    raw_address = data.get("address", data) if isinstance(data, dict) else data
    return Address.model_validate(raw_address)


async def delete_address(session: AsyncSession, user_id: int, address_id: str) -> None:
    """Delete a saved Swiggy delivery address.

    Irreversible on Swiggy's side; confirming with the user before calling
    this is the caller's responsibility.
    """
    await _call_authenticated(
        session, user_id, TOOL_DELETE_ADDRESS, {"addressId": address_id}
    )


async def search_products(
    session: AsyncSession,
    user_id: int,
    *,
    address_id: str,
    query: str,
    offset: int = DEFAULT_SEARCH_OFFSET,
) -> list[Product]:
    """Search Instamart products scoped to `address_id` (R4.1).

    Search is address-scoped: results (and stock) depend on the selected
    delivery address. No results is not an error - it returns an empty list
    so the substitution path (R4.4) can treat "nothing found" the same way
    it treats "found, but the preferred item was out of stock".
    """
    data = await _call_authenticated(
        session,
        user_id,
        TOOL_SEARCH_PRODUCTS,
        _search_arguments(address_id=address_id, query=query, offset=offset),
    )
    return _parse_products(data)


async def search_products_with_token(
    access_token: str,
    *,
    address_id: str,
    query: str,
    offset: int = DEFAULT_SEARCH_OFFSET,
) -> list[Product]:
    """`search_products` for a caller that already holds the access token.

    Identical to `search_products` in request shape and parsing, but touches
    no database session, so several of these can run concurrently under one
    request (CUE-74's batch tag resolution). The auth recovery ladder is the
    caller's to apply: an `InstamartAuthError` raised here has not marked the
    provider link expired.

    Args:
        access_token: A token from `resolve_access_token`.
        address_id: The delivery address results are scoped to.
        query: Free text - a bare pantry slug is a valid query.
        offset: Result offset for paging.

    Returns:
        The parsed candidates; empty when Swiggy has nothing, which is not an
        error.
    """
    data = await client.call_tool(
        access_token,
        TOOL_SEARCH_PRODUCTS,
        _search_arguments(address_id=address_id, query=query, offset=offset),
    )
    return _parse_products(data)


def _search_arguments(*, address_id: str, query: str, offset: int) -> dict[str, Any]:
    """Build `search_products` arguments, spelled once for both wrappers."""
    return {"addressId": address_id, "query": query, "offset": offset}


def _parse_products(data: Any) -> list[Product]:
    """Parse a `search_products` payload into candidates."""
    # The envelope key holding the list isn't pinned by Swiggy's docs; accept
    # either a bare list or one nested under "products".
    raw_products = data.get("products", []) if isinstance(data, dict) else data or []
    return [Product.model_validate(item) for item in raw_products]


async def update_cart(
    session: AsyncSession, user_id: int, *, address_id: str, items: list[CartItemInput]
) -> Cart:
    """Replace the entire Swiggy cart in one write (R5.1).

    `items` is always the full composed cart, never a delta: Swiggy's
    `update_cart` replaces whatever cart exists, so there is no partial or
    incremental variant of this call.
    """
    data = await _call_authenticated(
        session,
        user_id,
        TOOL_UPDATE_CART,
        {
            "selectedAddressId": address_id,
            "items": [item.model_dump(by_alias=True) for item in items],
        },
    )
    # The envelope isn't pinned by Swiggy's docs; accept the cart nested
    # under "cart" or returned as the data payload directly.
    raw_cart = data.get("cart", data) if isinstance(data, dict) else data
    return Cart.model_validate(raw_cart or {})


async def get_cart(session: AsyncSession, user_id: int) -> Cart:
    """Read the server cart - the source of truth before confirm and checkout (R5.2)."""
    data = await _call_authenticated(session, user_id, TOOL_GET_CART, {})
    raw_cart = data.get("cart", data) if isinstance(data, dict) else data
    return Cart.model_validate(raw_cart or {})


async def checkout(
    session: AsyncSession, user_id: int, *, address_id: str
) -> CheckoutResult:
    """Place a COD order against the current server cart (R6.1).

    Non-idempotent: this creates and confirms a real order in one operation.
    A transport-level failure here (`InstamartTransportError`) means the
    outcome is genuinely unknown, not "failed" - callers must reconcile via
    `get_orders` (R6.3) before ever calling this again, never blindly retry.
    """
    data = await _call_authenticated(
        session, user_id, TOOL_CHECKOUT, {"addressId": address_id}
    )
    raw_order = data.get("order", data) if isinstance(data, dict) else data
    return CheckoutResult.model_validate(raw_order or {})


async def get_orders(
    session: AsyncSession,
    user_id: int,
    *,
    count: int = DEFAULT_GET_ORDERS_COUNT,
    order_type: str = DEFAULT_ORDER_TYPE,
    active_only: bool = False,
) -> list[OrderSummary]:
    """List the user's recent Instamart orders (R6.3 reconciliation, R10.1 history).

    `order_type` defaults to "INSTAMART" and is always sent explicitly -
    get_orders' own tool default is "DASH" (food delivery), which is wrong
    for Cue. `count` is clamped to `MAX_GET_ORDERS_COUNT` client-side before
    it is sent; Swiggy's documented maximum is 20.
    """
    clamped_count = min(count, MAX_GET_ORDERS_COUNT)
    data = await _call_authenticated(
        session,
        user_id,
        TOOL_GET_ORDERS,
        {
            "count": clamped_count,
            "orderType": order_type,
            "activeOnly": active_only,
        },
    )
    # The envelope key holding the list isn't pinned by Swiggy's docs; accept
    # either a bare list or one nested under "orders".
    raw_orders = data.get("orders", []) if isinstance(data, dict) else data or []
    return [OrderSummary.model_validate(item) for item in raw_orders]


async def get_order_details(
    session: AsyncSession, user_id: int, order_id: str
) -> OrderDetails:
    """Fetch full detail for a single order, including line items (R10.2).

    An order id that doesn't belong to the caller (or doesn't exist) comes
    back as `success: false`, which `client.call_tool` already turns into
    `InstamartDomainError` before this function ever sees the payload - no
    extra handling is needed here for that case.
    """
    data = await _call_authenticated(
        session, user_id, TOOL_GET_ORDER_DETAILS, {"orderId": order_id}
    )
    raw_order = data.get("order", data) if isinstance(data, dict) else data
    return OrderDetails.model_validate(raw_order or {})


async def get_go_to_items(
    session: AsyncSession, user_id: int, address_id: str, offset: int = 0
) -> list[GoToItem]:
    """your_go_to_items wrapper (R4.3 preference bootstrap)."""
    data = await _call_authenticated(
        session,
        user_id,
        TOOL_YOUR_GO_TO_ITEMS,
        {"addressId": address_id, "offset": offset},
    )
    # The envelope key holding the list isn't pinned by Swiggy's docs; accept
    # either a bare list or one nested under "items".
    raw_items = data.get("items", []) if isinstance(data, dict) else data or []
    return [GoToItem.model_validate(item) for item in raw_items]


async def track_order(
    session: AsyncSession, user_id: int, order_id: str, *, lat: float, lng: float
) -> OrderTracking:
    """Fetch live tracking state for a single order (CUE-14).

    `lat`/`lng` are the caller's current location, forwarded to Swiggy so it
    can compute delivery-partner distance/ETA; the poll-rate floor that
    protects Swiggy from being hammered lives in `app.orders.service`, not
    here - this wrapper always makes a live call when invoked.
    """
    data = await _call_authenticated(
        session,
        user_id,
        TOOL_TRACK_ORDER,
        {"orderId": order_id, "lat": lat, "lng": lng},
    )
    # Neither the envelope key nor the request arg names for track_order are
    # pinned by Swiggy's docs; accept the tracking payload nested under
    # "tracking" or returned as the data payload directly.
    raw = data.get("tracking", data) if isinstance(data, dict) else data
    return OrderTracking.model_validate(raw or {})


def normalize_preferences(items: list[GoToItem]) -> dict[str, PreferenceSignal]:
    """Map category/ingredient name -> the most-ordered spinId/brand.

    Each `GoToItem`'s variants are assumed most-ordered-first, so
    `variants[0]` is the preferred variant; items with no variants are
    skipped rather than raising. The first occurrence of a given
    `product_name` wins, so a later, less-ordered duplicate never overwrites
    the most-ordered signal.
    """
    preferences: dict[str, PreferenceSignal] = {}
    for item in items:
        if not item.variants:
            continue
        preferences.setdefault(
            item.product_name,
            PreferenceSignal(spin_id=item.variants[0].spin_id, brand=item.brand),
        )
    return preferences
