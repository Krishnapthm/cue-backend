"""Order history and tracking API endpoints (CUE-14, CUE-41/CUE-42)."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.orders import service
from app.orders.schemas import OrderDetailsResponse, OrderListItem, TrackingResponse

router = APIRouter(prefix="/orders", tags=["orders"])


@router.get(
    "",
    response_model=list[OrderListItem],
    status_code=status.HTTP_200_OK,
    summary="The user's recent Instamart orders",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Not authenticated, or Swiggy session expired."
        },
    },
)
async def list_orders(user: CurrentUser, session: DbSession) -> list[OrderListItem]:
    """Return recent orders, newest first, for the Orders list (R10.1).

    An empty list is a normal result, not an error - it is exactly what the
    Orders-empty frame renders.
    """
    return await service.list_orders(session, user.id)


@router.get(
    "/{order_id}",
    response_model=OrderDetailsResponse,
    status_code=status.HTTP_200_OK,
    summary="Line items and bill breakdown for one order",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Not authenticated, or Swiggy session expired."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "No such order, or it is not the caller's."
        },
    },
)
async def get_order(
    order_id: str, user: CurrentUser, session: DbSession
) -> OrderDetailsResponse:
    """Return one order's line items and bill breakdown (R10.2)."""
    return await service.get_order(session, user.id, order_id)


@router.get(
    "/{order_id}/track",
    response_model=TrackingResponse,
    status_code=status.HTTP_200_OK,
    summary="Live tracking state for one order",
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "description": "Not authenticated, or Swiggy session expired."
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Swiggy could not track this order (not the "
            "caller's order, or too old to track)."
        },
    },
)
async def track(
    order_id: str, lat: float, lng: float, user: CurrentUser, session: DbSession
) -> TrackingResponse:
    """Return live status, ETA, and delivery-partner location for an order.

    Repeated polls within 10s of the last live Swiggy call are served a
    cached result (`app.orders.service.get_tracking`); the poll floor is
    enforced in the service, not here.
    """
    return await service.get_tracking(session, user.id, order_id, lat=lat, lng=lng)
