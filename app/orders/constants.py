"""Order tracking constants (CUE-14)."""

from __future__ import annotations

# Minimum time between two LIVE Swiggy track_order calls for the same
# (user_id, order_id), enforced by app.orders.service.get_tracking. A poller
# hitting this endpoint faster than this window is served the cached result
# instead of hammering Swiggy.
MIN_POLL_INTERVAL_SECONDS = 10
