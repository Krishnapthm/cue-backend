"""Pantry API endpoints (CUE-69).

Backs the Pantry screen (CUE-71) and the NFC restock scan (CUE-72). Every
route is scoped to the Firebase-authenticated caller: the list query filters
by user, and the single-item routes resolve `{item_id}` through
`owned_pantry_item`, which 404s on anything that is not the caller's.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.pantry import service
from app.pantry.dependencies import OwnedPantryItem
from app.pantry.models import PantryItem
from app.pantry.schemas import PantryItemCreate, PantryItemResponse, PantryItemUpdate

router = APIRouter(prefix="/pantry", tags=["pantry"])

# Error responses shared by several routes, spelled once so the documented
# contract cannot drift between endpoints that answer the same way.
_Responses = dict[int | str, dict[str, Any]]

_UNAUTHENTICATED: _Responses = {
    status.HTTP_401_UNAUTHORIZED: {"description": "Not authenticated."}
}
_NOT_FOUND: _Responses = {
    status.HTTP_404_NOT_FOUND: {
        "description": "No such pantry item, or it is not the caller's."
    }
}
_INVALID_BODY: _Responses = {
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": "`level` is outside 0-3, `category` is not one of the "
        "fixed set, or the body changes nothing."
    }
}


@router.get(
    "",
    response_model=list[PantryItemResponse],
    status_code=status.HTTP_200_OK,
    summary="The caller's pantry, in category display order",
    responses={**_UNAUTHENTICATED},
)
async def list_pantry(user: CurrentUser, session: DbSession) -> list[PantryItem]:
    """Return the caller's pantry items, grouped in category order.

    An empty list is the normal day-one state, not an error - the Pantry
    screen has to be correct with zero items.
    """
    return await service.list_items(session, user.id)


@router.post(
    "",
    response_model=PantryItemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add a pantry item, or update the one already using that name",
    responses={**_UNAUTHENTICATED, **_INVALID_BODY},
)
async def add_pantry_item(
    request: PantryItemCreate, user: CurrentUser, session: DbSession
) -> PantryItem:
    """Add a staple to the caller's pantry, defaulting `level` to full.

    Adding a name the pantry already holds - compared case- and
    whitespace-insensitively - updates that item instead of creating a
    duplicate or erroring, so the client can add freely without checking
    first. That update is still reported as 201.
    """
    return await service.upsert_item(session, user.id, request)


@router.patch(
    "/{item_id}",
    response_model=PantryItemResponse,
    status_code=status.HTTP_200_OK,
    summary="Partially update a pantry item, in practice its stock level",
    responses={
        **_UNAUTHENTICATED,
        **_NOT_FOUND,
        **_INVALID_BODY,
        status.HTTP_409_CONFLICT: {
            "description": "The new name is already used by another item."
        },
    },
)
async def update_pantry_item(
    request: PantryItemUpdate,
    item: OwnedPantryItem,
    session: DbSession,
) -> PantryItem:
    """Write the fields the body carries, leaving the rest untouched.

    This is the hot path: the Pantry screen sends `{"level": n}` on every
    slider release.
    """
    return await service.update_item(session, item, request)


@router.delete(
    "/{item_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a pantry item",
    responses={**_UNAUTHENTICATED, **_NOT_FOUND},
)
async def delete_pantry_item(item: OwnedPantryItem, session: DbSession) -> Response:
    """Delete one of the caller's pantry items."""
    await service.delete_item(session, item)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
