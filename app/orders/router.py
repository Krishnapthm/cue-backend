"""Order tracking API endpoints (CUE-14)."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.orders import service
from app.orders.schemas import TrackingResponse

router = APIRouter(prefix="/orders", tags=["orders"])


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
