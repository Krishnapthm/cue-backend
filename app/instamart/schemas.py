"""Instamart tool request/response schemas (CUE-10, CUE-11).

Field names for `create_address`'s request are exactly as documented at
https://mcp.swiggy.com/builders/docs/reference/instamart/create_address.md.
Swiggy's response payloads are not field-documented; `Address` mirrors the
request fields it round-trips plus `addressId`, which `delete_address`'s docs
confirm is present on every saved address. `ProductVariant.spin_id` is
likewise inferred rather than literally spelled out on `search_products`' own
page, but is cross-confirmed by `update_cart`'s documented request shape
(`items: [{spinId, quantity}]`) - cart operations are variant-level and
addressed by spinId (R4.1/R4.2).
"""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AddressCategory(StrEnum):
    """Swiggy's `addressCategory` enum (create_address)."""

    HOME = "HOME"
    WORK = "WORK"
    OFFICE = "OFFICE"
    FRIENDS_AND_FAMILY = "FRIENDS_AND_FAMILY"
    OTHER = "OTHER"


class Address(BaseModel):
    """A saved Swiggy delivery address (get_addresses).

    Latitude/longitude are withheld by Swiggy for privacy and are never
    present here.
    """

    model_config = ConfigDict(populate_by_name=True)

    address_id: str = Field(alias="addressId")
    full_address: str = Field(alias="fullAddress")
    address_line: str = Field(alias="addressLine")
    address_line2: str | None = Field(default=None, alias="addressLine2")
    city: str
    postal_code: str = Field(alias="postalCode")
    locality: str | None = None
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


class ProductVariant(BaseModel):
    """One purchasable variant of a search_products candidate (R4.1, R4.2).

    Cart operations (update_cart) address items by `spin_id`, never by
    product id - a variant is the unit selection and quantity math (CUE-15)
    reason about. `pack_size`/`price`/`in_stock` are optional: if Swiggy's
    actual keys differ from these, a variant still parses with them unset
    rather than failing the whole search.
    """

    model_config = ConfigDict(populate_by_name=True)

    spin_id: str = Field(alias="spinId")
    pack_size: str | None = Field(default=None, alias="packSize")
    price: Decimal | None = None
    in_stock: bool = Field(default=True, alias="inStock")


class Product(BaseModel):
    """A search_products candidate, with its purchasable variants (R4.1)."""

    model_config = ConfigDict(populate_by_name=True)

    product_id: str | None = Field(default=None, alias="productId")
    name: str | None = None
    brand: str | None = None
    variants: list[ProductVariant] = Field(default_factory=list)
