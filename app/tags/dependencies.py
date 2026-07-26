from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.models.tag import TagBinding
from app.tags import service


async def owned_tag_binding(
    tag_uid: str, user: CurrentUser, session: DbSession
) -> TagBinding:
    """Resolve `{tag_uid}` to a binding the caller owns.

    Loads and authorizes in one step, so no route handler ever sees a tag UID
    it has not already proven is bound by the signed-in user.

    Raises:
        TagBindingNotFoundError: If the caller has no binding for that tag.
    """
    return await service.get_binding(session, user.id, tag_uid)


OwnedTagBinding = Annotated[TagBinding, Depends(owned_tag_binding)]
