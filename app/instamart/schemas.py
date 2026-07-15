"""Instamart tool request/response schemas (CUE-10).

Field names for `create_address`'s request are exactly as documented at
https://mcp.swiggy.com/builders/docs/reference/instamart/create_address.md.
Swiggy's response payloads are not field-documented; `Address` mirrors the
request fields it round-trips plus `addressId`, which `delete_address`'s docs
confirm is present on every saved address.
"""

from __future__ import annotations

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
