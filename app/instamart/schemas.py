"""Instamart tool request/response schemas (CUE-10, CUE-11).

Field names for `create_address`'s request are exactly as documented at
https://mcp.swiggy.com/builders/docs/reference/instamart/create_address.md.
Swiggy's response payloads are not field-documented, so `Address` mirrors an
observed live `get_addresses` response rather than the request it round-trips:
the read model is much narrower than the write model (one flattened
`addressLine`, no city/postalCode/coordinates). `ProductVariant.spin_id` is
likewise inferred rather than literally spelled out on `search_products`' own
page, but is cross-confirmed by `update_cart`'s documented request shape
(`items: [{spinId, quantity}]`) - cart operations are variant-level and
addressed by spinId (R4.1/R4.2).
"""

from __future__ import annotations

import logging
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class AddressCategory(StrEnum):
    """Swiggy's `addressCategory` enum (create_address)."""

    HOME = "HOME"
    WORK = "WORK"
    OFFICE = "OFFICE"
    FRIENDS_AND_FAMILY = "FRIENDS_AND_FAMILY"
    OTHER = "OTHER"

    @classmethod
    def _missing_(cls, value: object) -> AddressCategory:
        """Coerce Swiggy's read-side spelling onto the documented enum.

        `create_address` takes SCREAMING_SNAKE ("FRIENDS_AND_FAMILY") but
        `get_addresses` echoes categories back title-cased ("Home", "Other"),
        so the two sides of the same field disagree on casing. Anything that
        still doesn't match degrades to `OTHER` rather than failing the whole
        list: the category only drives which icon the sheet renders, and the
        user's own label lives in `address_tag`.
        """
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_").replace("&", "AND")
            for member in cls:
                if member.value == normalized:
                    return member
        logger.warning("Unrecognized Swiggy addressCategory %r; using OTHER", value)
        return cls.OTHER


class Address(BaseModel):
    """A saved Swiggy delivery address, as returned by `get_addresses`.

    Swiggy's read model is far narrower than `create_address`'s request: the
    address comes back as a single flattened `addressLine` (with the account
    holder's name prefixed), and city, postal code, locality and
    latitude/longitude are all withheld - the coordinates deliberately, for
    privacy. `phoneNumber` arrives masked (e.g. "****0324").
    """

    model_config = ConfigDict(populate_by_name=True)

    id: str
    address_line: str = Field(alias="addressLine")
    phone_number: str | None = Field(default=None, alias="phoneNumber")
    address_category: AddressCategory | None = Field(
        default=None, alias="addressCategory"
    )
    address_tag: str | None = Field(default=None, alias="addressTag")


class CreateAddressRequest(BaseModel):
    """Request to create a new Swiggy delivery address."""

    model_config = ConfigDict(populate_by_name=True)

    full_address: str = Field(alias="fullAddress", min_length=1)
    address_line: str = Field(alias="addressLine", min_length=1)
    address_line2: str | None = Field(default=None, alias="addressLine2")
    city: str = Field(min_length=1)
    postal_code: str = Field(alias="postalCode", min_length=1)
    latitude: float
    longitude: float
    address_category: AddressCategory = Field(alias="addressCategory")
    user_name: str = Field(alias="userName", min_length=1)
    user_phone: str = Field(alias="userPhone", min_length=1)
    locality: str | None = None
    address_tag: str | None = Field(default=None, alias="addressTag")
    receiver_name: str | None = Field(default=None, alias="receiverName")
    receiver_phone: str | None = Field(default=None, alias="receiverPhone")


class ProductRating(BaseModel):
    """A variation's rating as supplied by Instamart search results.

    Swiggy sends both fields as display strings, including already-rounded
    counts such as ``"51.5k"``. Keep that object as-is so cart/search
    responses do not fabricate precision or force clients to reverse a local
    rename.
    """

    model_config = ConfigDict(populate_by_name=True)

    value: str
    count: str


def _flatten_price_value(value: Any) -> Any:
    """Reduce Swiggy's nested price object to the price actually charged.

    Live responses send `price` as `{mrp, offerPrice, unitLevelPrice}`, but
    every consumer (variant selection, cart line totals, CUE-15) treats
    `price` as a single Decimal. `offerPrice` is what the user pays, so it
    wins; `mrp` is the fallback when nothing is discounted. A plain scalar
    passes through untouched. Shared by `ProductVariant` and `CartLineItem`:
    both are documented as nesting price the same way.
    """
    if isinstance(value, dict):
        offer_price = value.get("offerPrice", value.get("offer_price"))
        return offer_price if offer_price is not None else value.get("mrp")
    return value


class ProductVariant(BaseModel):
    """One purchasable variant of a search_products candidate (R4.1, R4.2).

    Cart operations (update_cart) address items by `spin_id`, never by
    product id - a variant is the unit selection and quantity math (CUE-15)
    reason about.

    Field names mirror an observed live `search_products` response (CUE-77).
    The earlier guesses `packSize`/`inStock` never appear on the wire; Swiggy
    sends `quantityDescription` and `isInStockAndAvailable`. Both spellings
    are accepted so payloads shaped like the old assumption still parse.
    """

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId")
    pack_size: str | None = Field(
        default=None,
        validation_alias=AliasChoices("quantityDescription", "packSize", "pack_size"),
    )
    price: Decimal | None = None
    in_stock: bool = Field(
        default=True,
        validation_alias=AliasChoices("isInStockAndAvailable", "inStock", "in_stock"),
    )
    # Both fields were observed on individual `variations`, which makes their
    # association with the cart-addressable spin_id explicit. Rating is null
    # for some otherwise purchasable variations.
    image_url: str | None = Field(default=None, alias="imageUrl")
    rating: ProductRating | None = None

    @field_validator("price", mode="before")
    @classmethod
    def _flatten_price(cls, value: Any) -> Any:
        return _flatten_price_value(value)


class Product(BaseModel):
    """A search_products candidate, with its purchasable variants (R4.1).

    Field names mirror an observed live response (CUE-77): the variant list
    arrives under `variations` and the name under `displayName`. The previous
    `variants`/`name` guesses meant every product parsed with an empty
    variant list, leaving nothing addressable by `spinId` to put in a cart.
    """

    model_config = ConfigDict(populate_by_name=True)

    product_id: str | None = Field(default=None, alias="productId")
    name: str | None = Field(
        default=None, validation_alias=AliasChoices("displayName", "name")
    )
    brand: str | None = None
    variants: list[ProductVariant] = Field(
        default_factory=list,
        validation_alias=AliasChoices("variations", "variants"),
    )


class CartItemInput(BaseModel):
    """One `update_cart` line item, exactly as Swiggy's docs specify it."""

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId")
    quantity: int = Field(gt=0)


class CartLineItem(BaseModel):
    """One line of the server cart (get_cart / update_cart's response).

    Field names beyond `spin_id`/`quantity` (the request-side contract) are
    not documented; the rest parse defensively and are optional.
    """

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId")
    quantity: int
    price: Decimal | None = None

    @field_validator("price", mode="before")
    @classmethod
    def _flatten_price(cls, value: Any) -> Any:
        return _flatten_price_value(value)

    # Swiggy spells the line's name differently in different payloads -
    # `search_products` alone answers with `displayName` where the cart docs say
    # `productName` - and a line that parses unnamed reaches the user as a
    # placeholder word. Accept every spelling seen rather than one of them.
    product_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "productName", "displayName", "product_name", "name", "itemName"
        ),
        # Kept explicitly: the response the app parses is camelCase, and a
        # validation alias alone would serialize this back out as `product_name`.
        serialization_alias="productName",
    )
    image_url: str | None = Field(default=None, alias="imageUrl")
    rating: ProductRating | None = None


class Cart(BaseModel):
    """The server cart - the source of truth before confirm and checkout (R5.2).

    Swiggy's docs confirm `availablePaymentMethods` and prose-describe "a
    pricing breakdown and totals" and "minimum order requirements"; the exact
    keys for those aren't spelled out, so `total`/`minimum_order_value` parse
    defensively and are optional.
    """

    model_config = ConfigDict(populate_by_name=True)

    items: list[CartLineItem] = Field(default_factory=list)
    total: Decimal | None = None
    minimum_order_value: Decimal | None = Field(default=None, alias="minimumOrderValue")
    available_payment_methods: list[str] = Field(
        default_factory=list, alias="availablePaymentMethods"
    )


class CheckoutResult(BaseModel):
    """Result of a successful `checkout` call (R6.1).

    Swiggy's docs confirm `checkout` "creates order and confirms payment in
    a single operation" but don't spell out response field names; `order_id`
    parses defensively from the most likely key and may be absent, which
    callers must treat as an unconfirmed outcome, not a placed order.
    """

    model_config = ConfigDict(populate_by_name=True)

    order_id: str | None = Field(default=None, alias="orderId")
    total: Decimal | None = None


class OrderSummary(BaseModel):
    """One entry from get_orders.

    Used both to check whether a checkout landed after a transport failure
    (R6.3 reconciliation) and as the Order-History list shape (R10.1) -
    `get_orders` is the single source for both. `items`/`address` are
    tool-specific and not further typed until a screen needs more than
    passthrough display.

    `placed_at` and `total` are the two fields the Order-History list frame
    needs beyond id/status. Swiggy's docs don't pin either field's name for
    get_orders, so both accept the plausible spellings and stay optional -
    a payload carrying neither still validates, and the list renders without
    them rather than failing the whole request.
    """

    model_config = ConfigDict(populate_by_name=True)

    order_id: str = Field(alias="orderId")
    status: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    address: dict[str, Any] | None = None
    placed_at: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "placed_at", "placedAt", "orderedTime", "orderTime", "createdAt"
        ),
    )
    total: Decimal | None = Field(
        default=None,
        validation_alias=AliasChoices("total", "grandTotal", "orderTotal"),
    )


class OrderLineItem(BaseModel):
    """One line item of an order, from get_order_details."""

    model_config = ConfigDict(populate_by_name=True)

    product_name: str = Field(alias="productName")
    quantity: int
    price: Decimal


class OrderDetails(BaseModel):
    """Full detail of a single order, from get_order_details (R10.2)."""

    model_config = ConfigDict(populate_by_name=True)

    order_id: str = Field(alias="orderId")
    status: str
    items: list[OrderLineItem]
    # Same defensive, optional parse as OrderSummary.placed_at - the detail
    # frame titles the screen with the order's date.
    placed_at: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "placed_at", "placedAt", "orderedTime", "orderTime", "createdAt"
        ),
    )
    item_total: Decimal | None = Field(default=None, alias="itemTotal")
    delivery_fee: Decimal | None = Field(default=None, alias="deliveryFee")
    handling_fee: Decimal | None = Field(default=None, alias="handlingFee")
    grand_total: Decimal | None = Field(default=None, alias="grandTotal")


class GoToItemVariant(ProductVariant):
    """One variation of a your_go_to_items entry (R4.3 preference bootstrap).

    your_go_to_items and search_products return byte-identical variation
    objects, so this shares `ProductVariant`'s parsing rather than restating
    it - one place to fix when Swiggy's field names move. The tool is assumed
    to return each item's variations most-ordered-first, so `variants[0]` is
    the preferred variant.
    """


class GoToItem(BaseModel):
    """One product the user has previously ordered (your_go_to_items, R4.3).

    Field names mirror an observed live response (CUE-77): the name arrives
    as `displayName` and the list as `variations`. The earlier
    `productName`/`variants` guesses are still accepted so existing callers
    and fixtures keep parsing.
    """

    model_config = ConfigDict(populate_by_name=True)

    product_name: str = Field(
        validation_alias=AliasChoices("displayName", "productName", "product_name")
    )
    brand: str | None = None
    category: str | None = None
    variants: list[GoToItemVariant] = Field(
        default_factory=list,
        validation_alias=AliasChoices("variations", "variants"),
    )


class OrderTracking(BaseModel):
    """Live tracking state for a single order, from track_order (CUE-14).

    Swiggy's docs don't pin the exact response field names for this tool;
    `order_id`/`status` mirror the shape every other order-scoped response
    uses (`orderId`, `status`), and `eta`/`delivery_partner_location` parse
    defensively and are optional so a partial payload still validates.
    """

    model_config = ConfigDict(populate_by_name=True)

    order_id: str = Field(alias="orderId")
    status: str
    eta: str | None = None
    delivery_partner_location: dict[str, float] | None = Field(
        default=None, alias="deliveryPartnerLocation"
    )


class PreferenceSignal(BaseModel):
    """A normalized preference signal consumed by variant selection (R4.3)."""

    spin_id: str
    brand: str | None = None
