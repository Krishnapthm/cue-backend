"""orders (R6.1/R6.3, CUE-19) - Cue-side checkout record, not an order mirror.

Order History (R10.1/R10.2) does NOT read this table - it reads get_orders. This
table exists for idempotency verification of a non-idempotent checkout.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Numeric,
    Text,
    Uuid,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base
from app.models.mixins import CreatedAtMixin, UpdatedAtMixin


class Order(CreatedAtMixin, UpdatedAtMixin, Base):
    __tablename__ = "order"
    __table_args__ = (
        CheckConstraint("char_length(address_id) <= 100", name="address_id_length"),
        CheckConstraint(
            "status IN ('placing', 'placed', 'failed', 'unknown')",
            name="status_allowed",
        ),
        CheckConstraint(
            "char_length(swiggy_order_id) <= 100", name="swiggy_order_id_length"
        ),
        CheckConstraint("total >= 0", name="total_nonneg"),
        CheckConstraint(
            "status <> 'placed'"
            " OR (swiggy_order_id IS NOT NULL AND placed_at IS NOT NULL)",
            name="placed_has_id",
        ),
        Index("ix_order_user", "user_id", text("checkout_started_at DESC")),
        Index("ix_order_session", "session_id"),
        Index("ix_order_plan", "plan_id"),
        Index(
            "uq_order_swiggy_order_id",
            "swiggy_order_id",
            unique=True,
            postgresql_where=text("swiggy_order_id IS NOT NULL"),
        ),
        # R6.3 as a database guarantee: one checkout in flight per user. A
        # concurrent second checkout hits a constraint violation instead of
        # placing a real duplicate.
        Index(
            "uq_order_one_placing",
            "user_id",
            unique=True,
            postgresql_where=text("status = 'placing'"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("chat_session.id", ondelete="CASCADE"), nullable=False
    )
    # ON DELETE RESTRICT, not CASCADE: a real placed order must never lose its
    # line-item provenance to a cascade.
    plan_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("cart_plan.id", ondelete="RESTRICT"), nullable=False
    )
    address_id: Mapped[str] = mapped_column(Text, nullable=False)
    # placing -> row inserted BEFORE calling checkout (guarded by
    # uq_orders_one_placing). placed -> success, swiggy_order_id recorded.
    # failed -> definitive 4xx. unknown -> 5xx/timeout, order MAY exist; never
    # retry, resolve via get_orders.
    status: Mapped[str] = mapped_column(Text, nullable=False)
    swiggy_order_id: Mapped[str | None] = mapped_column(Text)
    total: Mapped[Decimal | None] = mapped_column(Numeric(10, 2))
    checkout_started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("now()")
    )
    placed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
