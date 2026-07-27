"""NFC tag API endpoints (CUE-74).

`resolve-batch` is the only route on the scan path, and it is called once,
after the user has finished tapping - never per tap. The correction routes
are off every hot path.

Every route is scoped to the Firebase-authenticated caller: batch resolution
reads and writes only that user's bindings, and `{tag_uid}` resolves through
`owned_tag_binding`, which 404s on a tag the caller has not bound.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

from app.auth.dependencies import CurrentUser
from app.database import DbSession
from app.models.tag import TagBinding
from app.tags import service
from app.tags.dependencies import OwnedTagBinding
from app.tags.schemas import (
    TagBindingResponse,
    TagBindingUpdate,
    TagResolveBatchRequest,
    TagResolveBatchResponse,
)

router = APIRouter(prefix="/pantry/tags", tags=["tags"])

_Responses = dict[int | str, dict[str, Any]]

_UNAUTHENTICATED: _Responses = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": "Not authenticated, or Swiggy session expired."
    }
}
_NOT_FOUND: _Responses = {
    status.HTTP_404_NOT_FOUND: {
        "description": "No binding for that tag, or it is not the caller's."
    }
}


@router.post(
    "/resolve-batch",
    response_model=TagResolveBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve a finished NFC scan to orderable Instamart variants",
    responses={
        **_UNAUTHENTICATED,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "`taps` is empty, over the batch ceiling, or a tap "
            "is missing its uid or text."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "Swiggy could not be reached."},
    },
)
async def resolve_batch(
    request: TagResolveBatchRequest, user: CurrentUser, session: DbSession
) -> TagResolveBatchResponse:
    """Resolve every scanned tag in one call, in request order.

    A slug Swiggy has nothing for comes back with outcome `unresolved` inside
    a normal 200 - one dead slug never fails the other nine.
    """
    results = await service.resolve_batch(session, user.id, request)
    return TagResolveBatchResponse(results=results)


@router.patch(
    "/{tag_uid}",
    response_model=TagBindingResponse,
    status_code=status.HTTP_200_OK,
    summary="Re-bind a tag to a different variant",
    responses={**_UNAUTHENTICATED, **_NOT_FOUND},
)
async def update_tag_binding(
    request: TagBindingUpdate, binding: OwnedTagBinding, session: DbSession
) -> TagBinding:
    """Point a sticker at the variant the user actually wants.

    The correction path for a slug that ranked onto the wrong pack or brand;
    the chosen variant is stored as given, not re-ranked.
    """
    return await service.rebind(session, binding, request)


@router.delete(
    "/{tag_uid}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Unbind a tag",
    responses={**_UNAUTHENTICATED, **_NOT_FOUND},
)
async def delete_tag_binding(binding: OwnedTagBinding, session: DbSession) -> Response:
    """Forget the binding, so the next scan of that sticker resolves afresh."""
    await service.unbind(session, binding)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
