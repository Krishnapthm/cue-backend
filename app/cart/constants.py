"""Cart composition invariants (R5.4, CUE-16) and cart-API constants (CUE-80)."""

from __future__ import annotations

from decimal import Decimal

# Swiggy Instamart's minimum order value (R5.4). Purely a reporting threshold
# - `compose_cart` and the cart endpoints (CUE-80) both write regardless of
# where the cart sits against it; a cart is allowed to sit below the minimum
# while it is being filled, and checkout is where the floor actually applies.
MINIMUM_ORDER_VALUE = Decimal("99.00")

# Reason reported for a line Swiggy accepted the write for, but which is
# absent from the cart it read back. Swiggy answers `success: true` and
# silently drops such items rather than failing the call, so the read-back
# diff is the only way to notice.
DROPPED_BY_SWIGGY_REASON = (
    "Swiggy accepted the request but the item is not in the resulting cart; "
    "it is most likely out of stock or undeliverable to this address."
)

# Reason reported for a line that was already in the cart before this write,
# and is absent afterwards, despite this write never asking for its removal.
# Distinct from DROPPED_BY_SWIGGY_REASON (a *new* line Swiggy would not add):
# this is Swiggy evicting an unrelated, previously-accepted line during a
# merge write - most likely a cart size limit - and the client needs a
# different message for a line the user did not just try to add.
EVICTED_BY_SWIGGY_REASON = (
    "Swiggy dropped this item from the cart while adding others; it is most "
    "likely a cart size limit. Add it again if you still want it."
)
