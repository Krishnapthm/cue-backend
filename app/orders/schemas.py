"""Order tracking and order-history API schemas (CUE-14, CUE-41/CUE-42)."""

from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel


class OrderStatus(StrEnum):
    """Closed-set order status surfaced by the order-history endpoints.

    Wider than `TrackingStatus` in both directions: history has to render
    orders that never reached delivery (`CANCELLED`), and the Orders list
    and tracking screens both label the two live stages separately rather
    than collapsing them into "active". Mapped from Swiggy's free-form raw
    status string by `app.orders.service._map_order_status` - never passed
    through verbatim.

    `PREPARING` and `OUT_FOR_DELIVERY` are the non-terminal values: the app
    routes those into live tracking and the other two into the static
    detail screen.
    """

    PREPARING = "preparing"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class OrderListItem(BaseModel):
    """One row of `GET /orders` (R10.1).

    `items` carries product names only. The list frame shows a one-line
    item summary ("Paneer, Butter, Tomatoes + 5 more"); composing that line
    is presentation, so it belongs to the app, not the API.
    """

    order_id: str
    status: OrderStatus
    placed_at: str | None
    items: list[str]
    total: Decimal | None


class OrderLineItem(BaseModel):
    """One line item of `GET /orders/{order_id}` (R10.2)."""

    product_name: str
    quantity: int
    price: Decimal


class OrderDetailsResponse(BaseModel):
    """Response for `GET /orders/{order_id}` (R10.2)."""

    order_id: str
    status: OrderStatus
    placed_at: str | None
    items: list[OrderLineItem]
    item_total: Decimal | None
    delivery_fee: Decimal | None
    handling_fee: Decimal | None
    grand_total: Decimal | None


class TrackingStatus(StrEnum):
    """Closed-set tracking status surfaced to callers.

    Mapped from Swiggy's free-form raw status string by
    `app.orders.service._map_status` - never passed through verbatim.
    """

    ACTIVE = "active"
    DELIVERED = "delivered"


class TrackingResponse(BaseModel):
    """Response for `GET /orders/{order_id}/track`."""

    order_id: str
    status: TrackingStatus
    eta: str | None
    delivery_partner_location: dict[str, float] | None
