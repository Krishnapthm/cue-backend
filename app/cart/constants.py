"""Cart composition invariants (R5.4, CUE-16) and cart-API constants (CUE-80)."""

from __future__ import annotations

from decimal import Decimal

# Swiggy Instamart's minimum order value (R5.4). Enforced client-side before
# ever calling update_cart, so we never write a cart that can't check out.
# Deliberately *not* enforced by the cart endpoints (CUE-80): a cart is
# allowed to sit below the minimum while it is being filled; checkout is
# where the floor applies.
MINIMUM_ORDER_VALUE = Decimal("99.00")

# Reason reported for a line Swiggy accepted the write for, but which is
# absent from the cart it read back. Swiggy answers `success: true` and
# silently drops such items rather than failing the call, so the read-back
# diff is the only way to notice.
DROPPED_BY_SWIGGY_REASON = (
    "Swiggy accepted the request but the item is not in the resulting cart; "
    "it is most likely out of stock or undeliverable to this address."
)
