"""Variant selection (CUE-15) and cart composition (CUE-16) schemas.

`SelectedVariant`'s fields mirror `app.models.cart.CartPlanItem` 1:1 (same
`match_status` values, same optionality) so CUE-16 can persist a selection
without any further translation - CUE-15 decides *what* to buy; CUE-16
decides how it becomes a `CartPlan` row and a Swiggy `update_cart` call.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel

from app.instamart.schemas import Cart


class MatchStatus(StrEnum):
    """Mirrors `cart_plan_item.match_status`'s CHECK constraint values."""

    MATCHED = "matched"
    SUBSTITUTED = "substituted"
    UNAVAILABLE = "unavailable"


class Ingredient(BaseModel):
    """One recipe ingredient to resolve to a purchasable Instamart variant."""

    name: str
    quantity: Decimal | None = None
    unit: str | None = None
    # The go-to preference signal (R4.3): a brand name the user favors for
    # this ingredient, if known. Sourcing this value is out of scope here.
    preferred_brand: str | None = None


class SelectedVariant(BaseModel):
    """The single best variant chosen for one `Ingredient` (R4.2)."""

    ingredient_name: str
    ingredient_qty: Decimal | None = None
    ingredient_unit: str | None = None
    match_status: MatchStatus
    spin_id: str | None = None
    product_name: str | None = None
    pack_size: str | None = None
    unit_price: Decimal | None = None
    # Purchasable pack count for the needed quantity (R4.5), None when
    # `unavailable` or when quantity math wasn't computable.
    quantity: int | None = None
    # Amount left over past the need, in the pack's base unit (g or ml),
    # once `quantity` packs are bought. None when quantity math wasn't
    # computable; 0 means an exact fit.
    overage: Decimal | None = None
    selection_reason: str


class ComposeCartResult(BaseModel):
    """Outcome of composing a `CartPlan` (R5.1/R5.4, CUE-16).

    `cart` is the get_cart read-back (R5.2) and is only present when the
    minimum order value was met - below it, the plan is still recorded (so
    recompose history stays debuggable) but `update_cart` is never called,
    since there is nothing checkout-able yet.
    """

    plan_id: int
    subtotal: Decimal
    minimum_order_value: Decimal
    below_minimum: bool
    shortfall: Decimal
    cart: Cart | None = None
