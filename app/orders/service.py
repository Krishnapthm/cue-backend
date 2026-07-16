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
from app.orders.constants import MIN_POLL_INTERVAL_SECONDS
from app.orders.schemas import TrackingResponse, TrackingStatus

# Raw Swiggy status tokens (lowercased) that indicate a terminal, delivered
# order. Matched case-insensitively against `OrderTracking.status`.
_DELIVERED_STATUS_TOKENS = frozenset({"delivered", "completed", "order_delivered"})

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
