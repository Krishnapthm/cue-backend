"""Order tracking service: track_order wrapper with poll-rate discipline (CUE-14).

Swiggy's track_order is a live call; a client polling this endpoint too
aggressively (e.g. a tight UI refresh loop) would otherwise hammer Swiggy on
every poll. The floor below is enforced here, in the service layer, so no
client-side behavior can bypass it - the router and any future caller of
`get_tracking` gets the same protection for free.
"""

from __future__ import annotations

import time

from sqlalchemy.ext.asyncio import AsyncSession

from app.instamart import service as instamart_service
from app.instamart.schemas import OrderSummary
from app.orders.constants import MIN_POLL_INTERVAL_SECONDS
from app.orders.schemas import (
    OrderDetailsResponse,
    OrderLineItem,
    OrderListItem,
    OrderStatus,
    TrackingResponse,
    TrackingStatus,
)

# Raw Swiggy status tokens (lowercased) that indicate a terminal, delivered
# order. Matched case-insensitively against `OrderTracking.status`.
_DELIVERED_STATUS_TOKENS = frozenset({"delivered", "completed", "order_delivered"})

# Raw tokens for an order that is terminal but never arrived.
_CANCELLED_STATUS_TOKENS = frozenset({"cancelled", "canceled", "refunded", "failed"})

# Raw tokens for an order that has left the store. Checked before the
# preparing fallback so the Orders list and the tracking screen can label the
# two live stages apart instead of collapsing them into one "active".
_OUT_FOR_DELIVERY_STATUS_TOKENS = frozenset(
    {"out_for_delivery", "out for delivery", "dispatched", "shipped", "on_the_way"}
)

# Process-local cache: (user_id, order_id) -> (last_live_call_monotonic, response).
# This is NOT shared across instances - in a multi-instance deployment the
# 10s floor only holds per-process. Holding the floor cluster-wide would
# require a shared store (e.g. Redis) keyed the same way.
_tracking_cache: dict[tuple[int, str], tuple[float, TrackingResponse]] = {}


def _now() -> float:
    """Monotonic clock, indirected so tests can monkeypatch it to advance
    time without sleeping."""
    return time.monotonic()


def _reset_tracking_cache() -> None:
    """Clear the tracking cache. Exposed for test isolation between cases."""
    _tracking_cache.clear()


def _map_status(raw: str) -> TrackingStatus:
    """Map Swiggy's raw status string onto the closed `TrackingStatus` set.

    Lowercases `raw` and checks it against a documented set of
    delivered-indicating tokens (`_DELIVERED_STATUS_TOKENS`), matching on
    exact token or substring. Any status not recognized as delivered
    defaults to `ACTIVE` - a poller should keep polling until the order is
    unambiguously delivered, so treating an unknown status as terminal would
    be the unsafe direction to guess wrong in.
    """
    normalized = raw.lower()
    if any(token in normalized for token in _DELIVERED_STATUS_TOKENS):
        return TrackingStatus.DELIVERED
    return TrackingStatus.ACTIVE


def _map_order_status(raw: str) -> OrderStatus:
    """Map Swiggy's raw status string onto the closed `OrderStatus` set.

    Checked most-specific first: cancelled and delivered are terminal and
    must never be mistaken for a live order, then out-for-delivery, then
    everything else falls back to `PREPARING`.

    The fallback direction is deliberate and matches `_map_status`: an
    unrecognized status is far more likely to be a stage of a live order
    than a terminal one, and guessing "live" only costs a tracking screen
    that shows an early stage, whereas guessing "terminal" would strand a
    real in-flight order on a static detail page with no way to track it.
    """
    normalized = raw.lower()
    for tokens, status in (
        (_CANCELLED_STATUS_TOKENS, OrderStatus.CANCELLED),
        (_DELIVERED_STATUS_TOKENS, OrderStatus.DELIVERED),
        (_OUT_FOR_DELIVERY_STATUS_TOKENS, OrderStatus.OUT_FOR_DELIVERY),
    ):
        if any(token in normalized for token in tokens):
            return status
    return OrderStatus.PREPARING


def _item_names(summary: OrderSummary) -> list[str]:
    """Best-effort product names off `get_orders`' untyped `items` dicts.

    Swiggy doesn't pin the key, so try the plausible spellings per entry and
    drop entries that carry no usable name at all rather than emitting a
    blank row into the list frame's item summary.
    """
    names: list[str] = []
    for item in summary.items:
        if not isinstance(item, dict):
            continue
        for key in ("productName", "product_name", "name", "displayName"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.append(value.strip())
                break
    return names


async def list_orders(session: AsyncSession, user_id: int) -> list[OrderListItem]:
    """Return the user's recent Instamart orders, newest first (R10.1).

    A thin projection over `instamart_service.get_orders`: it maps the raw
    status onto the closed `OrderStatus` set and flattens each order's items
    down to product names, which is all the Orders list frame renders.
    Ordering is Swiggy's own - the tool returns recent orders first and there
    is no reliable timestamp to re-sort on.

    Raises:
        InstamartAuthError: Not linked, or the link is expired (propagates
            uncaught; the global `AppError` handler maps it to 401).
    """
    summaries = await instamart_service.get_orders(session, user_id)
    return [
        OrderListItem(
            order_id=summary.order_id,
            status=_map_order_status(summary.status),
            placed_at=summary.placed_at,
            items=_item_names(summary),
            total=summary.total,
        )
        for summary in summaries
    ]


async def get_order(
    session: AsyncSession, user_id: int, order_id: str
) -> OrderDetailsResponse:
    """Return one order's line items and bill breakdown (R10.2).

    Raises:
        InstamartAuthError: Not linked, or the link is expired (-> 401).
        InstamartDomainError: Swiggy reported `success: false` for this order
            - it doesn't exist, or belongs to another user. Propagates
            uncaught so the global handler maps it to 422 rather than a raw
            500; the app renders its error state off that.
    """
    details = await instamart_service.get_order_details(session, user_id, order_id)
    return OrderDetailsResponse(
        order_id=details.order_id,
        status=_map_order_status(details.status),
        placed_at=details.placed_at,
        items=[
            OrderLineItem(
                product_name=item.product_name,
                quantity=item.quantity,
                price=item.price,
            )
            for item in details.items
        ],
        item_total=details.item_total,
        delivery_fee=details.delivery_fee,
        handling_fee=details.handling_fee,
        grand_total=details.grand_total,
    )


async def get_tracking(
    session: AsyncSession, user_id: int, order_id: str, *, lat: float, lng: float
) -> TrackingResponse:
    """Enforce >=10s between live Swiggy calls per (user_id, order_id); serve
    the cached TrackingResponse within the window instead of re-hitting Swiggy.

    `lat`/`lng` are forwarded to Swiggy on a live call but never affect the
    cache key or the poll floor - only `(user_id, order_id)` does.

    Raises:
        InstamartAuthError: Not linked, or the link is expired (propagates
            from `instamart_service.track_order` uncaught; the global
            `AppError` handler maps it to 401).
        InstamartDomainError: Swiggy reported `success: false` for this
            order (e.g. it belongs to another user, or is too old to
            track). Propagates uncaught so the global handler maps it to
            422 instead of this surfacing as a raw 500. Failures are never
            cached, so the next call retries live.
    """
    cache_key = (user_id, order_id)
    cached = _tracking_cache.get(cache_key)
    if cached is not None:
        last_live_call, cached_response = cached
        if _now() - last_live_call < MIN_POLL_INTERVAL_SECONDS:
            return cached_response

    tracking = await instamart_service.track_order(
        session, user_id, order_id, lat=lat, lng=lng
    )
    response = TrackingResponse(
        order_id=tracking.order_id,
        status=_map_status(tracking.status),
        eta=tracking.eta,
        delivery_partner_location=tracking.delivery_partner_location,
    )
    _tracking_cache[cache_key] = (_now(), response)
    return response
