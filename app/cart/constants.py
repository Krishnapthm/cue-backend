"""Cart composition invariants (R5.4, CUE-16)."""

from __future__ import annotations

from decimal import Decimal

# Swiggy Instamart's minimum order value (R5.4). Enforced client-side before
# ever calling update_cart, so we never write a cart that can't check out.
MINIMUM_ORDER_VALUE = Decimal("99.00")
