"""cart_plans and cart_plan_items (R4.2/R4.4/R4.5/R5.1, CUE-15/16/17).

We persist the plan, not the cart - Swiggy owns the cart (R5.2). Prices here are
snapshots at compose time, for display and reasoning only; the bill of record is
always get_cart.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin


class CartPlan(CreatedAtMixin, Base):
    __tablename__ = "cart_plan"
    __table_args__ = (
        CheckConstraint("char_length(address_id) <= 100", name="address_id_length"),
        Index("ix_cart_plan_session", "session_id"),
        # R3.3 as a database guarantee: at most one live plan per session.
        # Address change supersedes the old plan and inserts a new one; plans are
        # append-only so the superseded history keeps recomposes debuggable.
        Index(
            "uq_cart_plan_live",
            "session_id",
            unique=True,
            postgresql_where=text("superseded_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False
    )
    address_id: Mapped[str] = mapped_column(Text, nullable=False)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CartPlanItem(CreatedAtMixin, Base):
    """One row per NEEDED ingredient, matched or not. Cart lines are the subset
    where spin_id IS NOT NULL; unmatched rows are the match-rate denominator."""

    __tablename__ = "cart_plan_item"
    __table_args__ = (
        CheckConstraint(
            "char_length(ingredient_name) <= 200", name="ingredient_name_length"
        ),
        CheckConstraint(
            "char_length(ingredient_unit) <= 20", name="ingredient_unit_length"
        ),
        CheckConstraint(
            "match_status IN ('matched', 'substituted', 'unavailable')",
            name="match_status_allowed",
        ),
        CheckConstraint("char_length(spin_id) <= 100", name="spin_id_length"),
        CheckConstraint("char_length(product_name) <= 300", name="product_name_length"),
        CheckConstraint("char_length(pack_size) <= 50", name="pack_size_length"),
        CheckConstraint("unit_price >= 0", name="unit_price_nonneg"),
        CheckConstraint("quantity > 0", name="quantity_positive"),
        CheckConstraint(
            "char_length(selection_reason) <= 1000", name="selection_reason_length"
        ),
        # A resolved item must carry a purchasable variant.
        CheckConstraint(
            "match_status = 'unavailable'"
            " OR (spin_id IS NOT NULL AND quantity IS NOT NULL"
            " AND unit_price IS NOT NULL)",
            name="resolved",
        ),
        # R4.4: a substitution is never silent. The reason is not optional.
        CheckConstraint(
            "match_status <> 'substituted' OR selection_reason IS NOT NULL",
            name="substitution_reason",
        ),
        Index("ix_cart_plan_item_plan", "plan_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    plan_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("cart_plan.id", ondelete="CASCADE"), nullable=False
    )
    ingredient_name: Mapped[str] = mapped_column(Text, nullable=False)
    ingredient_qty: Mapped[Decimal | None] = mapped_column(Numeric(10, 3))
    ingredient_unit: Mapped[str | None] = mapped_column(Text)
    # matched -> auto-mapped, in stock. substituted -> preferred unavailable,
    # alternative proposed WITH a reason. unavailable -> nothing purchasable;
    # row still exists as the denominator.
    match_status: Mapped[str] = mapped_column(Text, nullable=False)
    spin_id: Mapped[str | None] = mapped_column(Text)
    product_name: Mapped[str | None] = mapped_column(Text)
    pack_size: Mapped[str | None] = mapped_column(Text)
    unit_price: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    quantity: Mapped[int | None] = mapped_column(Integer)
    selection_reason: Mapped[str | None] = mapped_column(Text)
    # R4.6 round-it-out items: fold into the same plan and the same single
    # update_cart write, but must not pollute the match-rate denominator.
    is_addon: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
