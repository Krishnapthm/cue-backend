"""Cart API endpoints (CUE-80).

The app talks to Cue and Cue talks to Swiggy, never the other way around:
the Swiggy OAuth access token lives server-side (`app.providers.service`
owns it and its expiry ladder), and the transport is MCP JSON-RPC with an
envelope that CUE-77 proved is easy to parse wrong. Both are reasons this
surface exists rather than the device calling Swiggy itself.

Every route is Firebase-authenticated and scoped to the caller: the cart is
addressed by the user's own Swiggy session, never by an id in the path, so
there is no cart another user could name.

The mutating routes take an `addressId` because Swiggy's `update_cart`
requires `selectedAddressId` - stock and deliverability are address-scoped.
All the read-merge-write logic lives in `service.py`; these handlers only do
HTTP.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from app.auth.dependencies import CurrentUser
from app.cart import service
from app.cart.schemas import (
    AddCartItemsRequest,
    CartMutationResult,
    UpdateCartItemQuantityRequest,
)
from app.database import DbSession
from app.instamart.schemas import Cart

router = APIRouter(prefix="/cart", tags=["cart"])

# Error responses shared by several routes, spelled once so the documented
# contract cannot drift between endpoints that answer the same way.
_Responses = dict[int | str, dict[str, Any]]

_SWIGGY_ERRORS: _Responses = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": "Not authenticated, or the Swiggy link is expired - the "
        "link is marked expired and the user must reconnect."
    },
    status.HTTP_502_BAD_GATEWAY: {"description": "Swiggy Instamart was unreachable."},
}
_ITEM_NOT_FOUND: _Responses = {
    status.HTTP_404_NOT_FOUND: {"description": "The cart holds no such `spinId`."}
}

_ADDRESS_ID_QUERY = Query(
    alias="addressId",
    min_length=1,
    description="Swiggy delivery address the cart is written against.",
)


@router.get(
    "",
    response_model=Cart,
    status_code=status.HTTP_200_OK,
    summary="The caller's current Swiggy server cart",
    responses={**_SWIGGY_ERRORS},
)
async def read_cart(user: CurrentUser, session: DbSession) -> Cart:
    """Return the server cart as Swiggy currently holds it (R5.2).

    An empty cart is a normal state, not an error.
    """
    return await service.get_cart(session, user.id)


@router.post(
    "/items",
    response_model=CartMutationResult,
    status_code=status.HTTP_200_OK,
    summary="Add items to the cart, preserving what it already holds",
    responses={
        **_SWIGGY_ERRORS,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "`items` is empty, or a `quantity` is not positive."
        },
    },
)
async def add_cart_items(
    request: AddCartItemsRequest, user: CurrentUser, session: DbSession
) -> CartMutationResult:
    """Add the given lines to the cart and return the resulting cart.

    `quantity` is a delta: a `spinId` already in the cart has its quantity
    increased, never overwritten. Existing cart contents are always kept.

    A line Swiggy refuses comes back in `rejected` with a reason while the
    rest of the batch still lands - a partly-accepted batch is a 200, so the
    scan screen can keep exactly the failed rows on screen.
    """
    return await service.add_items(
        session, user.id, address_id=request.address_id, items=request.items
    )


@router.patch(
    "/items/{spin_id}",
    response_model=CartMutationResult,
    status_code=status.HTTP_200_OK,
    summary="Set one cart line's quantity",
    responses={
        **_SWIGGY_ERRORS,
        **_ITEM_NOT_FOUND,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "`quantity` is not positive - use DELETE to remove a line."
        },
    },
)
async def update_cart_item(
    spin_id: str,
    request: UpdateCartItemQuantityRequest,
    user: CurrentUser,
    session: DbSession,
) -> CartMutationResult:
    """Set `spin_id`'s quantity to an absolute value and return the cart."""
    return await service.set_item_quantity(
        session,
        user.id,
        address_id=request.address_id,
        spin_id=spin_id,
        quantity=request.quantity,
    )


@router.delete(
    "/items/{spin_id}",
    response_model=CartMutationResult,
    status_code=status.HTTP_200_OK,
    summary="Remove one line from the cart",
    responses={**_SWIGGY_ERRORS, **_ITEM_NOT_FOUND},
)
async def delete_cart_item(
    spin_id: str,
    user: CurrentUser,
    session: DbSession,
    address_id: Annotated[str, _ADDRESS_ID_QUERY],
) -> CartMutationResult:
    """Remove `spin_id` from the cart and return the resulting cart.

    Returns 200 with the cart rather than 204: the client would otherwise
    have to follow every delete with a read to redraw totals.
    """
    return await service.remove_item(
        session, user.id, address_id=address_id, spin_id=spin_id
    )
