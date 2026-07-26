"""Checkout + idempotency safety (R6.1/R6.3, CUE-19).

Swiggy's `checkout` creates and confirms a COD order in one non-idempotent
operation (COD only in v1, R6.1). A transport-level failure (5xx, timeout, a
network error) leaves the true outcome unknown - the request may have
reached Swiggy and placed the order anyway. `place_order` never retries in
that case: it records the Cue-side `Order` row as `unknown`, checks Swiggy's
own recent order history via `get_orders` (R6.3), and returns both - the
caller decides what to do next, since Swiggy gives us no idempotency key to
match a specific order with certainty.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.cart.exceptions import CartNotCheckoutableError, CheckoutInProgressError
from app.instamart import service as instamart_service
from app.instamart.exceptions import (
    InstamartAuthError,
    InstamartDomainError,
    InstamartTransportError,
)
from app.instamart.schemas import OrderSummary
from app.models.order import Order
from app.pantry import service as pantry_service

logger = logging.getLogger(__name__)


async def _stamp_pantry(
    session: AsyncSession, user_id: int, plan_id: int, placed_at: datetime
) -> None:
    """Record that a placed order restocked the user's matching staples.

    Runs after the order has already been committed, in its own transaction,
    and swallows database failures: this is bookkeeping for the Pantry
    screen's "Bought 3 days ago" line, and losing it must never turn a
    successfully placed order into an error the caller sees.
    """
    try:
        stamped = await pantry_service.stamp_last_bought(
            session, user_id, plan_id, placed_at
        )
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        logger.exception(
            "Failed to stamp last_bought_at for user %s, plan %s; the order is "
            "placed regardless.",
            user_id,
            plan_id,
        )
        return
    logger.info("Stamped last_bought_at on %s pantry item(s).", stamped)


async def place_order(
    session: AsyncSession,
    user_id: int,
    chat_session_id: uuid.UUID,
    plan_id: int,
    address_id: str,
) -> tuple[Order, list[OrderSummary]]:
    """Place a COD order against the server-confirmed cart (R6.1).

    Returns the persisted `Order` row - `swiggy_order_id` is the reference
    the tracking screen consumes once `status == "placed"` - plus Swiggy's
    recent orders, populated only when `status == "unknown"` so the caller
    can inspect what Swiggy shows before ever attempting checkout again.

    Raises:
        CartNotCheckoutableError: The server cart (get_cart, R5.2) is empty;
            `checkout` is never called against nothing.
        CheckoutInProgressError: Another checkout is already `placing` for
            this user; never place two orders concurrently.
    """
    cart = await instamart_service.get_cart(session, user_id)
    if not cart.items:
        raise CartNotCheckoutableError

    order = Order(
        user_id=user_id,
        session_id=chat_session_id,
        plan_id=plan_id,
        address_id=address_id,
        status="placing",
    )
    session.add(order)
    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise CheckoutInProgressError from exc

    try:
        result = await instamart_service.checkout(
            session, user_id, address_id=address_id
        )
    except (InstamartAuthError, InstamartDomainError):
        # Swiggy told us definitively that this attempt did not place an
        # order - no ambiguity, no reconciliation needed.
        order.status = "failed"
        await session.commit()
        return order, []
    except InstamartTransportError:
        recent_orders = await instamart_service.get_orders(session, user_id)
        order.status = "unknown"
        await session.commit()
        return order, recent_orders

    if result.order_id is None:
        # success:true but no parseable order id: the DB's placed_has_id
        # constraint requires one, and we can't honestly call this placed.
        recent_orders = await instamart_service.get_orders(session, user_id)
        order.status = "unknown"
        await session.commit()
        return order, recent_orders

    order.status = "placed"
    order.swiggy_order_id = result.order_id
    order.total = result.total
    order.placed_at = datetime.now(UTC)
    await session.commit()
    await _stamp_pantry(session, user_id, plan_id, order.placed_at)
    return order, []
