from __future__ import annotations

from fastapi import status

from app.exceptions import AppError


class CartNotCheckoutableError(AppError):
    """The server cart (get_cart, R5.2) is empty; there is nothing to check out."""

    status_code = status.HTTP_409_CONFLICT
    detail = "Cart is empty; nothing to check out."


class CartItemNotFoundError(AppError):
    """The addressed `spin_id` is not a line of the caller's server cart (CUE-80).

    Raised by the PATCH/DELETE item routes rather than silently creating the
    line: a quantity change against something the cart does not hold is a
    client bug, not an add.
    """

    status_code = status.HTTP_404_NOT_FOUND
    detail = "No such item in the cart."


class CheckoutInProgressError(AppError):
    """Another checkout is already `placing` for this user (R6.3).

    DB-enforced by `uq_order_one_placing`: at most one order may be in
    flight per user, so a concurrent second checkout is rejected here
    rather than risking a duplicate order.
    """

    status_code = status.HTTP_409_CONFLICT
    detail = "A checkout is already in progress."
