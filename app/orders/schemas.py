"""Order tracking API schemas (CUE-14)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


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
