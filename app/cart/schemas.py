"""Variant selection (CUE-15), cart composition (CUE-16) and the cart API
(CUE-80) schemas.

`SelectedVariant`'s fields mirror `app.models.cart.CartPlanItem` 1:1 (same
`match_status` values, same optionality) so CUE-16 can persist a selection
without any further translation - CUE-15 decides *what* to buy; CUE-16
decides how it becomes a `CartPlan` row and a Swiggy `update_cart` call.

The CUE-80 request/response models are the app-facing contract and are
spelled camelCase on the wire (`spinId`, `addressId`), matching Swiggy's own
naming for the same fields so the scan screen never has to translate between
two spellings of one id.
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from app.instamart.schemas import Cart, ProductRating


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
    image_url: str | None = None
    rating: ProductRating | None = None
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


class CartItemRequest(BaseModel):
    """One line the client wants in the cart, keyed by Swiggy `spinId`.

    On `POST /cart/items` `quantity` is a *delta*: it is added to whatever
    the cart already holds for this `spin_id`. Setting an absolute quantity
    is `PATCH /cart/items/{spin_id}`.
    """

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId", min_length=1)
    quantity: int = Field(gt=0)


class AddCartItemsRequest(BaseModel):
    """Body of `POST /cart/items`.

    `address_id` is required because Swiggy's `update_cart` requires
    `selectedAddressId` - stock and deliverability are address-scoped, so
    there is no address-free way to write a cart.
    """

    model_config = ConfigDict(populate_by_name=True)

    address_id: str = Field(alias="addressId", min_length=1)
    items: list[CartItemRequest] = Field(min_length=1)


class UpdateCartItemQuantityRequest(BaseModel):
    """Body of `PATCH /cart/items/{spin_id}` - the line's absolute quantity.

    `quantity` is `gt=0`: removing a line is `DELETE`, not a PATCH to zero,
    so there is exactly one way to express each intent.
    """

    model_config = ConfigDict(populate_by_name=True)

    address_id: str = Field(alias="addressId", min_length=1)
    quantity: int = Field(gt=0)


class RejectedCartItem(BaseModel):
    """One line Swiggy would not take, reported per-item rather than as a 5xx.

    The scan screen keeps exactly these rows on screen, so the client has to
    be told *which* item failed and why - a blanket error on the whole batch
    would lose that.
    """

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId")
    quantity: int
    reason: str


class CartMutationResult(BaseModel):
    """The outcome of any mutating cart call (CUE-80).

    Always carries the resulting cart, so the client never has to follow a
    write with a read, and splits the batch into what landed and what did
    not. A non-empty `rejected` is a normal, successful (200) response.
    """

    cart: Cart
    added: list[CartItemRequest] = Field(default_factory=list)
    rejected: list[RejectedCartItem] = Field(default_factory=list)
