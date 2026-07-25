"""Saved delivery address API endpoints (CUE-66).

Swiggy is the source of truth for addresses; Cue stores none of its own. This
router is therefore a pure HTTP surface over the `app.instamart.service`
wrappers (CUE-10) - there is no projection or invariant to enforce in
between, so it calls them directly rather than through a pass-through service
of its own.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.instamart import service as instamart_service
from app.instamart.schemas import Address, CreateAddressRequest

router = APIRouter(prefix="/addresses", tags=["addresses"])

_SWIGGY_AUTH_RESPONSE = {
    "description": "Not authenticated, or Swiggy session expired.",
}


@router.get(
    "",
    response_model=list[Address],
    status_code=status.HTTP_200_OK,
    summary="The user's saved Swiggy delivery addresses",
    responses={status.HTTP_401_UNAUTHORIZED: _SWIGGY_AUTH_RESPONSE},
)
async def list_addresses(user: CurrentUser, session: DbSession) -> list[Address]:
    """Return every saved delivery address for the account (R3.1).

    An empty list is a normal result, not an error - it is exactly what the
    empty address sheet renders.
    """
    return await instamart_service.get_addresses(session, user.id)


@router.post(
    "",
    response_model=Address,
    status_code=status.HTTP_201_CREATED,
    summary="Save a new delivery address",
    responses={
        status.HTTP_401_UNAUTHORIZED: _SWIGGY_AUTH_RESPONSE,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "Swiggy rejected the address."
        },
    },
)
async def create_address(
    request: CreateAddressRequest, user: CurrentUser, session: DbSession
) -> Address:
    """Create a delivery address and return it as saved by Swiggy (R3.1)."""
    return await instamart_service.create_address(session, user.id, request)


@router.delete(
    "/{address_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a saved delivery address",
    responses={
        status.HTTP_401_UNAUTHORIZED: _SWIGGY_AUTH_RESPONSE,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "No such address, or it is not the caller's."
        },
    },
)
async def delete_address(
    address_id: str, user: CurrentUser, session: DbSession
) -> None:
    """Delete one saved delivery address."""
    await instamart_service.delete_address(session, user.id, address_id)
