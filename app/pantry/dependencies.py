from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.pantry import service
from app.pantry.models import PantryItem


async def owned_pantry_item(
    item_id: int, user: CurrentUser, session: DbSession
) -> PantryItem:
    """Resolve `{item_id}` to a pantry item the caller owns.

    Loads and authorizes in one step, so no route handler ever sees an item
    id it has not already proven belongs to the signed-in user.

    Raises:
        PantryItemNotFoundError: If no such item exists for this user.
    """
    return await service.get_item(session, user.id, item_id)


OwnedPantryItem = Annotated[PantryItem, Depends(owned_pantry_item)]
