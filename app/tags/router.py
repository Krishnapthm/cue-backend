"""NFC tag API endpoints (CUE-74, CUE-79).

`resolve` is the scan hot path, called once per tap: the app lands the row
immediately as a skeleton and fills it in when this answers, so nothing
blocks the next tap and the user sees the real product while the jar is still
in their hand. `resolve-batch` is the reconciliation call at Add to cart -
the finished set goes up, and any tap that failed or happened offline gets a
second chance before the cart write. Both are cheap to keep: binding is an
upsert keyed on `(user_id, tag_uid)`, so a tag already resolved per-tap comes
back `cached` from the batch at no upstream cost. The correction routes are
off every hot path.

Every route is scoped to the Firebase-authenticated caller: resolution reads
and writes only that user's bindings, and `{tag_uid}` resolves through
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
    TagResolveRequest,
    TagResolveResponse,
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
    "/resolve",
    response_model=TagResolveResponse,
    status_code=status.HTTP_200_OK,
    summary="Resolve one tapped NFC tag to an orderable Instamart variant",
    responses={
        **_UNAUTHENTICATED,
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "description": "The tap is missing its uid, text or address."
        },
        status.HTTP_502_BAD_GATEWAY: {"description": "Swiggy could not be reached."},
    },
)
async def resolve(
    request: TagResolveRequest, user: CurrentUser, session: DbSession
) -> TagResolveResponse:
    """Resolve a single tap, and offer the alternatives it beat.

    A slug Swiggy has nothing for comes back with outcome `unresolved` inside
    a normal 200. `candidates` is empty on a `cached` outcome - no search ran,
    so there are no alternatives to offer, and the picker should be hidden
    rather than treated as an error.
    """
    return await service.resolve_one(session, user.id, request)


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
