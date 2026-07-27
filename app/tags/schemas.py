"""NFC tag resolution request/response schemas (CUE-74).

Request fields are camelCase (`addressId`, `tagUid`) to match the rest of the
address-scoped surface the client already calls; responses are snake_case,
like every other Cue response model. `populate_by_name` is on, so a client may
send either spelling.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from app.tags.constants import (
    DEFAULT_TAP_QUANTITY,
    MAX_ADDRESS_ID_LENGTH,
    MAX_SPIN_ID_LENGTH,
    MAX_TAG_TEXT_LENGTH,
    MAX_TAG_UID_LENGTH,
    MAX_TAP_QUANTITY,
    MAX_TAPS_PER_BATCH,
    MIN_TAP_QUANTITY,
    TagOutcome,
)

AddressId = Annotated[str, Field(min_length=1, max_length=MAX_ADDRESS_ID_LENGTH)]
SpinId = Annotated[str, Field(min_length=1, max_length=MAX_SPIN_ID_LENGTH)]
TagUid = Annotated[str, Field(min_length=1, max_length=MAX_TAG_UID_LENGTH)]
TagText = Annotated[str, Field(min_length=1, max_length=MAX_TAG_TEXT_LENGTH)]
TapQuantity = Annotated[int, Field(ge=MIN_TAP_QUANTITY, le=MAX_TAP_QUANTITY)]


class TagTap(BaseModel):
    """One scanned sticker.

    `text` is the bare slug physically written on the tag - "sugar", "haldi" -
    and is all the tag can ever carry, which is why resolution happens
    server-side against live Swiggy data.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    tag_uid: TagUid = Field(alias="tagUid")
    text: TagText
    quantity: TapQuantity = DEFAULT_TAP_QUANTITY


class TagResolveBatchRequest(BaseModel):
    """Body of `POST /pantry/tags/resolve-batch`.

    The whole finished scan, in one call: no backend round-trip happens while
    the user is actually tapping jars.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    address_id: AddressId = Field(alias="addressId")
    taps: list[TagTap] = Field(min_length=1, max_length=MAX_TAPS_PER_BATCH)


class TagResolution(BaseModel):
    """One resolved tap, carrying enough to build a real cart line.

    Every product field is null when `outcome` is `unresolved`; `spin_id` is
    present on every other outcome, because `update_cart` addresses items by
    `spinId` and a result without one is not orderable.
    """

    model_config = ConfigDict(from_attributes=True)

    tag_uid: str
    text: str
    outcome: TagOutcome
    spin_id: str | None = None
    product_id: str | None = None
    product_name: str | None = None
    refill_size: str | None = None
    unit_price: Decimal | None = None
    in_stock: bool | None = None
    pantry_item_id: int | None = None
    quantity: int


class TagResolveBatchResponse(BaseModel):
    """One entry per tap, in request order."""

    results: list[TagResolution]


class TagResolveRequest(BaseModel):
    """Body of `POST /pantry/tags/resolve` - one tapped sticker (CUE-79).

    Flat rather than a nested tap: this is the hot path during a scan, called
    once per tap, and there is only ever one.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    address_id: AddressId = Field(alias="addressId")
    tag_uid: TagUid = Field(alias="tagUid")
    text: TagText
    quantity: TapQuantity = DEFAULT_TAP_QUANTITY

    def as_tap(self) -> TagTap:
        """View this request as the `TagTap` the shared resolution path takes.

        Single-tap and batch resolution run the identical code below this
        point; converting here is what keeps that true.
        """
        return TagTap(tag_uid=self.tag_uid, text=self.text, quantity=self.quantity)


class TagCandidate(BaseModel):
    """One alternative the same search turned up (CUE-79).

    Purchasable by construction - in stock and priced - so `spin_id` and
    `unit_price` are non-optional here even though they are nullable on
    `TagResolution`, which also has to describe an `unresolved` tap.
    `product_id` is carried because `PATCH /pantry/tags/{tag_uid}` takes it:
    picking an alternative is a re-bind, and the app should not need a second
    lookup to perform one.
    """

    model_config = ConfigDict(from_attributes=True)

    spin_id: str
    product_id: str | None = None
    product_name: str | None = None
    refill_size: str | None = None
    unit_price: Decimal


class TagResolveResponse(TagResolution):
    """Body of `POST /pantry/tags/resolve` - one resolution, plus alternates.

    Extends `TagResolution` rather than restating it, so the single-tap and
    batch payloads cannot drift; `candidates` is the only addition, and it is
    deliberately absent from the batch response, which nothing renders a
    picker from.

    On a `cached` outcome, no search runs - searching purely to fill this
    list would turn a free rescan into a paid one - so `candidates` holds
    only the single bound variant rather than a full alternates list.
    Clients should always have at least one entry to render the picker from.
    """

    candidates: list[TagCandidate] = Field(default_factory=list)


class TagBindingUpdate(BaseModel):
    """Body of `PATCH /pantry/tags/{tag_uid}` - re-bind to a chosen variant.

    A correction path, off every hot path: the user has picked a specific
    variant, so this writes it as given rather than re-ranking. `spin_id` is
    required because a binding that cannot be ordered is not worth storing.
    """

    model_config = ConfigDict(populate_by_name=True, str_strip_whitespace=True)

    spin_id: SpinId = Field(alias="spinId")
    product_id: str | None = Field(default=None, alias="productId")
    product_name: str | None = Field(default=None, alias="productName")
    refill_size: str | None = Field(default=None, alias="refillSize")
    unit_price: Decimal | None = Field(default=None, alias="unitPrice", ge=0)
    address_id: AddressId = Field(alias="addressId")


class TagBindingResponse(BaseModel):
    """A stored binding, as returned by the correction endpoints."""

    model_config = ConfigDict(from_attributes=True)

    tag_uid: str
    tag_text: str
    spin_id: str
    product_id: str | None
    product_name: str | None
    refill_size: str | None
    unit_price: Decimal | None
    address_id: str
    pantry_item_id: int | None
    last_used_at: datetime
