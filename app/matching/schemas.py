"""Substitution result schema (R4.4, CUE-17).

`SubstitutionResult` maps 1:1 onto the substituted-row columns of
`cart_plan_item` (see `app/models/cart.py`) - the caller persists it there
directly, so its shape and constraints must never drift from that table.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from app.instamart.schemas import ProductRating


class SubstitutionResult(BaseModel):
    """A proposed alternative variant for an out-of-stock preferred item.

    `reason` is capped at 1000 chars to match `cart_plan_item`'s
    `selection_reason_length` CHECK constraint - it can never violate that
    constraint by construction.
    """

    spin_id: str
    product_name: str
    pack_size: str | None
    unit_price: Decimal
    quantity: int
    image_url: str | None = None
    rating: ProductRating | None = None
    reason: str = Field(max_length=1000)
